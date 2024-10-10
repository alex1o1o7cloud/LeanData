import Mathlib

namespace jansen_family_has_three_children_l1751_175153

/-- Represents the Jansen family structure -/
structure JansenFamily where
  mother_age : ℝ
  father_age : ℝ
  grandfather_age : ℝ
  num_children : ℕ
  children_total_age : ℝ

/-- The Jansen family satisfies the given conditions -/
def is_valid_jansen_family (family : JansenFamily) : Prop :=
  family.father_age = 50 ∧
  family.grandfather_age = 70 ∧
  (family.mother_age + family.father_age + family.grandfather_age + family.children_total_age) / 
    (3 + family.num_children : ℝ) = 25 ∧
  (family.mother_age + family.grandfather_age + family.children_total_age) / 
    (2 + family.num_children : ℝ) = 20

/-- The number of children in a valid Jansen family is 3 -/
theorem jansen_family_has_three_children (family : JansenFamily) 
    (h : is_valid_jansen_family family) : family.num_children = 3 := by
  sorry

#check jansen_family_has_three_children

end jansen_family_has_three_children_l1751_175153


namespace salary_decrease_percentage_l1751_175115

/-- Proves that given an original salary of 5000, an initial increase of 10%,
    and a final salary of 5225, the percentage decrease after the initial increase is 5%. -/
theorem salary_decrease_percentage
  (original_salary : ℝ)
  (initial_increase_percentage : ℝ)
  (final_salary : ℝ)
  (h1 : original_salary = 5000)
  (h2 : initial_increase_percentage = 10)
  (h3 : final_salary = 5225)
  : ∃ (decrease_percentage : ℝ),
    decrease_percentage = 5 ∧
    final_salary = original_salary * (1 + initial_increase_percentage / 100) * (1 - decrease_percentage / 100) :=
by sorry

end salary_decrease_percentage_l1751_175115


namespace parabola_tangent_line_l1751_175108

theorem parabola_tangent_line (b c : ℝ) : 
  (∀ x, x^2 + b*x + c = 2*x → x = 2) ∧ 
  (2*2 = 2^2 + b*2 + c) ∧
  (∀ x, 2*x + b = 2) →
  b = -2 ∧ c = 4 := by
sorry

end parabola_tangent_line_l1751_175108


namespace average_age_of_nine_students_l1751_175123

theorem average_age_of_nine_students (total_students : ℕ) (total_average : ℝ) 
  (five_students : ℕ) (five_average : ℝ) (seventeenth_age : ℝ) 
  (nine_students : ℕ) (h1 : total_students = 17) 
  (h2 : total_average = 17) (h3 : five_students = 5) 
  (h4 : five_average = 14) (h5 : seventeenth_age = 75) 
  (h6 : nine_students = total_students - five_students - 1) :
  (total_students * total_average - five_students * five_average - seventeenth_age) / nine_students = 16 := by
  sorry

#check average_age_of_nine_students

end average_age_of_nine_students_l1751_175123


namespace marta_tips_l1751_175169

/-- Calculates the amount of tips Marta received given her total earnings, hourly rate, and hours worked -/
def tips_received (total_earnings hourly_rate hours_worked : ℕ) : ℕ :=
  total_earnings - hourly_rate * hours_worked

/-- Proves that Marta received $50 in tips -/
theorem marta_tips : tips_received 240 10 19 = 50 := by
  sorry

end marta_tips_l1751_175169


namespace backpack_traverse_time_l1751_175196

/-- Theorem: Time taken to carry backpack through obstacle course --/
theorem backpack_traverse_time (total_time door_time second_traverse_minutes second_traverse_seconds : ℕ) :
  let second_traverse_time := second_traverse_minutes * 60 + second_traverse_seconds
  let remaining_time := total_time - (door_time + second_traverse_time)
  total_time = 874 ∧ door_time = 73 ∧ second_traverse_minutes = 5 ∧ second_traverse_seconds = 58 →
  remaining_time = 443 := by
  sorry

end backpack_traverse_time_l1751_175196


namespace amusement_park_spending_l1751_175193

/-- Calculates the total amount spent by a group of children at an amusement park -/
def total_spent (num_children : ℕ) 
  (ferris_wheel_cost ferris_wheel_riders : ℕ)
  (roller_coaster_cost roller_coaster_riders : ℕ)
  (merry_go_round_cost : ℕ)
  (bumper_cars_cost bumper_cars_riders : ℕ)
  (ice_cream_cost ice_cream_eaters : ℕ)
  (hot_dog_cost hot_dog_eaters : ℕ)
  (pizza_cost pizza_eaters : ℕ) : ℕ :=
  ferris_wheel_cost * ferris_wheel_riders +
  roller_coaster_cost * roller_coaster_riders +
  merry_go_round_cost * num_children +
  bumper_cars_cost * bumper_cars_riders +
  ice_cream_cost * ice_cream_eaters +
  hot_dog_cost * hot_dog_eaters +
  pizza_cost * pizza_eaters

/-- Theorem stating that the total amount spent by the group is $170 -/
theorem amusement_park_spending :
  total_spent 8 5 5 7 3 3 4 6 8 5 6 4 4 3 = 170 := by
  sorry

end amusement_park_spending_l1751_175193


namespace condition1_implies_bijective_condition2_implies_bijective_condition3_implies_not_injective_not_surjective_condition4_not_necessarily_injective_or_surjective_l1751_175191

-- Define the function type
def RealFunction := ℝ → ℝ

-- Define the properties
def Injective (f : RealFunction) : Prop :=
  ∀ x y, f x = f y → x = y

def Surjective (f : RealFunction) : Prop :=
  ∀ y, ∃ x, f x = y

def Bijective (f : RealFunction) : Prop :=
  Injective f ∧ Surjective f

-- Theorem statements
theorem condition1_implies_bijective (f : RealFunction) 
  (h : ∀ x, f (f x - 1) = x + 1) : Bijective f := by sorry

theorem condition2_implies_bijective (f : RealFunction) 
  (h : ∀ x y, f (x + f y) = f x + y^5) : Bijective f := by sorry

theorem condition3_implies_not_injective_not_surjective (f : RealFunction) 
  (h : ∀ x, f (f x) = Real.sin x) : ¬(Injective f) ∧ ¬(Surjective f) := by sorry

theorem condition4_not_necessarily_injective_or_surjective : 
  ∃ f : RealFunction, (∀ x y, f (x + y^2) = f x * f y + x * f y - y^3 * f x) ∧ 
  ¬(Injective f) ∧ ¬(Surjective f) := by sorry

end condition1_implies_bijective_condition2_implies_bijective_condition3_implies_not_injective_not_surjective_condition4_not_necessarily_injective_or_surjective_l1751_175191


namespace expand_expression_l1751_175178

theorem expand_expression (y : ℝ) : (7 * y + 12) * (3 * y) = 21 * y^2 + 36 * y := by
  sorry

end expand_expression_l1751_175178


namespace ellipse_circle_intersection_l1751_175124

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive_a : 0 < a
  h_positive_b : 0 < b
  h_a_ge_b : a ≥ b

/-- Represents a circle with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ
  h_positive_r : 0 < r

/-- The statement of the problem -/
theorem ellipse_circle_intersection (e : Ellipse) (c : Circle) :
  e.a = 3 ∧ e.b = 2 ∧
  (∃ (x y : ℝ), x^2 / 9 + y^2 / 4 = 1 ∧ (x - c.h)^2 + (y - c.k)^2 = c.r^2) ∧
  (∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
    (x₁^2 / 9 + y₁^2 / 4 = 1 ∧ (x₁ - c.h)^2 + (y₁ - c.k)^2 = c.r^2) ∧
    (x₂^2 / 9 + y₂^2 / 4 = 1 ∧ (x₂ - c.h)^2 + (y₂ - c.k)^2 = c.r^2) ∧
    (x₃^2 / 9 + y₃^2 / 4 = 1 ∧ (x₃ - c.h)^2 + (y₃ - c.k)^2 = c.r^2) ∧
    (x₄^2 / 9 + y₄^2 / 4 = 1 ∧ (x₄ - c.h)^2 + (y₄ - c.k)^2 = c.r^2) ∧
    (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (x₃, y₃) ∧ (x₁, y₁) ≠ (x₄, y₄) ∧
    (x₂, y₂) ≠ (x₃, y₃) ∧ (x₂, y₂) ≠ (x₄, y₄) ∧ (x₃, y₃) ≠ (x₄, y₄)) →
  c.r ≥ Real.sqrt 5 ∧ c.r < 9 :=
sorry

end ellipse_circle_intersection_l1751_175124


namespace male_democrat_ratio_l1751_175189

/-- Proves the ratio of male democrats to total male participants in a meeting --/
theorem male_democrat_ratio (total_participants : ℕ) 
  (female_democrats : ℕ) (h1 : total_participants = 660) 
  (h2 : female_democrats = 110) 
  (h3 : female_democrats * 2 = total_participants / 3) : 
  (total_participants / 3 - female_democrats) * 4 = 
  (total_participants - female_democrats * 2) :=
sorry

end male_democrat_ratio_l1751_175189


namespace point_on_line_l1751_175185

/-- If (m, n) and (m + 2, n + k) are two points on the line with equation x = 2y + 3, then k = 1 -/
theorem point_on_line (m n k : ℝ) : 
  (m = 2*n + 3) → 
  (m + 2 = 2*(n + k) + 3) → 
  k = 1 := by
sorry

end point_on_line_l1751_175185


namespace power_of_two_equation_solution_l1751_175125

theorem power_of_two_equation_solution :
  ∀ (a n : ℕ), a ≥ n → n ≥ 2 → (∃ x : ℕ, (a + 1)^n + a - 1 = 2^x) →
  a = 4 ∧ n = 3 := by
  sorry

end power_of_two_equation_solution_l1751_175125


namespace cube_order_equivalence_l1751_175113

theorem cube_order_equivalence (a b : ℝ) : a > b ↔ a^3 > b^3 := by sorry

end cube_order_equivalence_l1751_175113


namespace common_roots_cubic_polynomials_l1751_175173

theorem common_roots_cubic_polynomials (a b : ℝ) :
  (∃ r s : ℝ, r ≠ s ∧
    r^3 + a*r^2 + 13*r + 12 = 0 ∧
    r^3 + b*r^2 + 17*r + 15 = 0 ∧
    s^3 + a*s^2 + 13*s + 12 = 0 ∧
    s^3 + b*s^2 + 17*s + 15 = 0) →
  a = 0 ∧ b = -1 := by
sorry

end common_roots_cubic_polynomials_l1751_175173


namespace sqrt_floor_equality_l1751_175157

theorem sqrt_floor_equality (n : ℕ+) :
  ⌊Real.sqrt n + Real.sqrt (n + 1)⌋ = ⌊Real.sqrt (4 * n + 1)⌋ ∧
  ⌊Real.sqrt (4 * n + 1)⌋ = ⌊Real.sqrt (4 * n + 2)⌋ ∧
  ⌊Real.sqrt (4 * n + 2)⌋ = ⌊Real.sqrt (4 * n + 3)⌋ := by
  sorry

end sqrt_floor_equality_l1751_175157


namespace det_B_squared_minus_3B_l1751_175127

def B : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 3, 2]

theorem det_B_squared_minus_3B : 
  Matrix.det ((B ^ 2) - 3 • B) = -704 := by sorry

end det_B_squared_minus_3B_l1751_175127


namespace terry_lunch_options_l1751_175174

/-- The number of lunch combination options for Terry's salad bar lunch. -/
def lunch_combinations (lettuce_types : ℕ) (tomato_types : ℕ) (olive_types : ℕ) 
                       (bread_types : ℕ) (fruit_types : ℕ) (soup_types : ℕ) : ℕ :=
  lettuce_types * tomato_types * olive_types * bread_types * fruit_types * soup_types

/-- Theorem stating that Terry's lunch combinations equal 4320. -/
theorem terry_lunch_options :
  lunch_combinations 4 5 6 3 4 3 = 4320 := by
  sorry

end terry_lunch_options_l1751_175174


namespace geometric_sequence_common_ratio_l1751_175156

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h1 : ∀ n, S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - (a 2 / a 1)))
  (h2 : ∀ n, a (n + 1) = a n * (a 2 / a 1))
  (h3 : S 3 = 15)
  (h4 : a 3 = 5) :
  (a 2 / a 1 = -1/2) ∨ (a 2 / a 1 = 1) :=
sorry

end geometric_sequence_common_ratio_l1751_175156


namespace min_red_chips_l1751_175139

theorem min_red_chips (r w b : ℕ) : 
  b ≥ w / 3 →
  b ≤ r / 4 →
  w + b ≥ 70 →
  r ≥ 72 ∧ ∀ r' : ℕ, (∃ w' b' : ℕ, b' ≥ w' / 3 ∧ b' ≤ r' / 4 ∧ w' + b' ≥ 70) → r' ≥ r :=
by sorry

end min_red_chips_l1751_175139


namespace intersection_point_satisfies_equations_l1751_175164

/-- Two lines in a 2D plane -/
structure TwoLines where
  line1 : ℝ → ℝ → ℝ
  line2 : ℝ → ℝ → ℝ

/-- The intersection point of two lines -/
def intersection_point : ℝ × ℝ := (-4, 3)

/-- The given two lines -/
def given_lines : TwoLines where
  line1 := fun x y => 3 * x + 2 * y + 6
  line2 := fun x y => 2 * x + 5 * y - 7

theorem intersection_point_satisfies_equations : 
  let (x, y) := intersection_point
  given_lines.line1 x y = 0 ∧ given_lines.line2 x y = 0 := by
  sorry

#check intersection_point_satisfies_equations

end intersection_point_satisfies_equations_l1751_175164


namespace x_plus_q_in_terms_of_q_l1751_175159

theorem x_plus_q_in_terms_of_q (x q : ℝ) (h1 : |x + 3| = q) (h2 : x > -3) :
  x + q = 2*q - 3 := by
sorry

end x_plus_q_in_terms_of_q_l1751_175159


namespace max_value_abc_l1751_175167

theorem max_value_abc (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 1) :
  a + a * b + a * b * c ≤ 1 := by
sorry

end max_value_abc_l1751_175167


namespace box_problem_l1751_175199

theorem box_problem (total_boxes : ℕ) (small_box_units : ℕ) (large_box_units : ℕ)
  (h_total : total_boxes = 62)
  (h_small : small_box_units = 5)
  (h_large : large_box_units = 3)
  (h_load_large_first : ∃ (x : ℕ), x * (1 / large_box_units) + 15 * (1 / small_box_units) = (total_boxes - x) * (1 / small_box_units) + 15 * (1 / large_box_units))
  : ∃ (large_boxes : ℕ), large_boxes = 27 ∧ total_boxes = large_boxes + (total_boxes - large_boxes) :=
by
  sorry

end box_problem_l1751_175199


namespace shortest_distance_on_cube_face_l1751_175148

/-- The shortest distance on the surface of a cube between midpoints of opposite edges on the same face -/
theorem shortest_distance_on_cube_face (edge_length : ℝ) (h : edge_length = 2) :
  let midpoint_distance := Real.sqrt 2
  ∃ (path : ℝ), path ≥ midpoint_distance ∧
    (∀ (other_path : ℝ), other_path ≥ midpoint_distance → path ≤ other_path) :=
by sorry

end shortest_distance_on_cube_face_l1751_175148


namespace linear_congruence_solution_l1751_175141

theorem linear_congruence_solution (x : ℤ) : 
  (9 * x + 2) % 15 = 7 → x % 5 = 0 := by
sorry

end linear_congruence_solution_l1751_175141


namespace cricket_players_count_l1751_175150

theorem cricket_players_count (total_players hockey_players football_players softball_players : ℕ) 
  (h1 : total_players = 77)
  (h2 : hockey_players = 15)
  (h3 : football_players = 21)
  (h4 : softball_players = 19) :
  total_players - (hockey_players + football_players + softball_players) = 22 := by
sorry

end cricket_players_count_l1751_175150


namespace necessary_but_not_sufficient_condition_l1751_175160

theorem necessary_but_not_sufficient_condition (a : ℝ) : 
  (∃ x ∈ Set.Icc 1 3, x^2 - a*x + 4 < 0) → 
  (a > 3 ∧ ∃ b > 3, ¬(∃ x ∈ Set.Icc 1 3, x^2 - b*x + 4 < 0)) := by
  sorry

end necessary_but_not_sufficient_condition_l1751_175160


namespace power_equation_solution_l1751_175118

theorem power_equation_solution : ∃ K : ℕ, 16^3 * 8^3 = 2^K ∧ K = 21 := by
  sorry

end power_equation_solution_l1751_175118


namespace race_total_time_l1751_175176

theorem race_total_time (total_runners : ℕ) (first_group : ℕ) (first_time : ℕ) (extra_time : ℕ) 
  (h1 : total_runners = 8)
  (h2 : first_group = 5)
  (h3 : first_time = 8)
  (h4 : extra_time = 2) :
  first_group * first_time + (total_runners - first_group) * (first_time + extra_time) = 70 := by
  sorry

end race_total_time_l1751_175176


namespace mentoring_program_fraction_l1751_175116

theorem mentoring_program_fraction (total : ℕ) (s : ℕ) (n : ℕ) : 
  total = s + n →
  s > 0 →
  n > 0 →
  (n : ℚ) / 4 = (s : ℚ) / 3 →
  ((n : ℚ) / 4 + (s : ℚ) / 3) / total = 2 / 7 :=
by sorry

end mentoring_program_fraction_l1751_175116


namespace no_solution_for_x_equals_one_l1751_175195

theorem no_solution_for_x_equals_one :
  ¬∃ (y : ℝ), (1 : ℝ) / (1 + 1) + y = (1 : ℝ) / (1 - 1) :=
by sorry

end no_solution_for_x_equals_one_l1751_175195


namespace equal_area_line_slope_l1751_175161

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Calculates the distance from a point to a line -/
def distancePointToLine (p : ℝ × ℝ) (l : Line) : ℝ :=
  sorry

/-- Determines if a line divides a circle into equal areas -/
def divideCircleEqually (c : Circle) (l : Line) : Prop :=
  sorry

theorem equal_area_line_slope :
  let c1 : Circle := ⟨(10, 40), 5⟩
  let c2 : Circle := ⟨(15, 30), 5⟩
  let p : ℝ × ℝ := (12, 20)
  ∃ (l : Line),
    l.slope = (5 - Real.sqrt 73) / 2 ∨ l.slope = (5 + Real.sqrt 73) / 2 ∧
    (p.1 * l.slope + l.yIntercept = p.2) ∧
    divideCircleEqually c1 l ∧
    divideCircleEqually c2 l :=
  sorry

end equal_area_line_slope_l1751_175161


namespace number_in_different_bases_l1751_175129

theorem number_in_different_bases : ∃ (n : ℕ), 
  (∃ (a : ℕ), a < 7 ∧ n = a * 7 + 0) ∧ 
  (∃ (a b : ℕ), a < 9 ∧ b < 9 ∧ a ≠ b ∧ n = a * 9 + b) ∧ 
  (n = 3 * 8 + 5) := by
  sorry

end number_in_different_bases_l1751_175129


namespace unique_solution_l1751_175198

def A (a b : ℝ) := {x : ℝ | x^2 + a*x + b = 0}
def B (c : ℝ) := {x : ℝ | x^2 + c*x + 15 = 0}

theorem unique_solution :
  ∃! (a b c : ℝ),
    (A a b ∪ B c = {3, 5}) ∧
    (A a b ∩ B c = {3}) ∧
    a = -6 ∧ b = 9 ∧ c = -8 := by sorry

end unique_solution_l1751_175198


namespace problem_statement_l1751_175143

theorem problem_statement (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 + b^2 = 1) : 
  (abs a + abs b ≤ Real.sqrt 2) ∧ (abs (a^3 / b) + abs (b^3 / a) ≥ 1) := by
  sorry

end problem_statement_l1751_175143


namespace line_intersection_l1751_175197

theorem line_intersection :
  ∃! (x y : ℚ), (8 * x - 5 * y = 40) ∧ (6 * x - y = -5) ∧ (x = 15/38) ∧ (y = 140/19) := by
  sorry

end line_intersection_l1751_175197


namespace y_expression_equivalence_l1751_175177

theorem y_expression_equivalence (x : ℝ) : 
  Real.sqrt ((x - 2)^2) + Real.sqrt (x^2 + 4*x + 5) = 
  |x - 2| + Real.sqrt ((x + 2)^2 + 1) := by
  sorry

end y_expression_equivalence_l1751_175177


namespace A_sufficient_not_necessary_for_D_l1751_175187

-- Define the propositions
variable (A B C D : Prop)

-- Define the relationships between the propositions
variable (h1 : A → B ∧ ¬(B → A))
variable (h2 : (B → C) ∧ (C → B))
variable (h3 : (C → D) ∧ ¬(D → C))

-- Theorem to prove
theorem A_sufficient_not_necessary_for_D : 
  (A → D) ∧ ¬(D → A) :=
sorry

end A_sufficient_not_necessary_for_D_l1751_175187


namespace hyperbola_equation_l1751_175134

/-- Given a hyperbola and a circle with specific properties, prove the equation of the hyperbola -/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 → 
    (∃ t : ℝ, (b*x + a*y = 0 ∨ b*x - a*y = 0) ∧ 
      ((x - 3)^2 + y^2 = 4 ↔ t = 0))) → 
  (∃ c : ℝ, c > 0 ∧ c^2 = a^2 + b^2 ∧ c = 3) →
  (a^2 = 5 ∧ b^2 = 4) := by
  sorry

end hyperbola_equation_l1751_175134


namespace car_speed_second_hour_l1751_175151

/-- Proves that given a car's speed in the first hour is 120 km/h and its average speed over two hours is 95 km/h, the speed of the car in the second hour is 70 km/h. -/
theorem car_speed_second_hour 
  (speed_first_hour : ℝ) 
  (average_speed : ℝ) 
  (h1 : speed_first_hour = 120) 
  (h2 : average_speed = 95) : 
  (2 * average_speed - speed_first_hour = 70) := by
  sorry

#check car_speed_second_hour

end car_speed_second_hour_l1751_175151


namespace infinite_grid_graph_chromatic_number_infinite_grid_graph_chromatic_number_lower_bound_infinite_grid_graph_chromatic_number_exact_l1751_175144

/-- An infinite grid graph -/
def InfiniteGridGraph : Type := ℤ × ℤ

/-- A coloring function for the infinite grid graph -/
def Coloring (G : Type) := G → Fin 2

/-- A valid coloring of the infinite grid graph -/
def IsValidColoring (c : Coloring InfiniteGridGraph) : Prop :=
  ∀ (x y : ℤ), (x + y) % 2 = c (x, y)

/-- The chromatic number of the infinite grid graph is at most 2 -/
theorem infinite_grid_graph_chromatic_number :
  ∃ (c : Coloring InfiniteGridGraph), IsValidColoring c :=
sorry

/-- The chromatic number of the infinite grid graph is at least 2 -/
theorem infinite_grid_graph_chromatic_number_lower_bound :
  ¬∃ (c : InfiniteGridGraph → Fin 1), 
    ∀ (x y : ℤ), c (x, y) ≠ c (x + 1, y) ∨ c (x, y) ≠ c (x, y + 1) :=
sorry

/-- The chromatic number of the infinite grid graph is exactly 2 -/
theorem infinite_grid_graph_chromatic_number_exact : 
  (∃ (c : Coloring InfiniteGridGraph), IsValidColoring c) ∧
  (¬∃ (c : InfiniteGridGraph → Fin 1), 
    ∀ (x y : ℤ), c (x, y) ≠ c (x + 1, y) ∨ c (x, y) ≠ c (x, y + 1)) :=
sorry

end infinite_grid_graph_chromatic_number_infinite_grid_graph_chromatic_number_lower_bound_infinite_grid_graph_chromatic_number_exact_l1751_175144


namespace sam_age_two_years_ago_l1751_175170

def john_age (sam_age : ℕ) : ℕ := 3 * sam_age

theorem sam_age_two_years_ago (sam_current_age : ℕ) : 
  john_age sam_current_age = 3 * sam_current_age ∧ 
  john_age sam_current_age + 9 = 2 * (sam_current_age + 9) →
  sam_current_age - 2 = 7 := by
sorry

end sam_age_two_years_ago_l1751_175170


namespace f_of_f_has_four_roots_l1751_175181

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 3*x + 2

-- State the theorem
theorem f_of_f_has_four_roots :
  ∃ (a b c d : ℝ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  (∀ x : ℝ, f (f x) = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = d) :=
sorry

end f_of_f_has_four_roots_l1751_175181


namespace system_solution_l1751_175180

theorem system_solution (x y z : ℝ) : 
  ((x + 1) * y * z = 12 ∧ 
   (y + 1) * z * x = 4 ∧ 
   (z + 1) * x * y = 4) ↔ 
  ((x = 2 ∧ y = -2 ∧ z = -2) ∨ 
   (x = 1/3 ∧ y = 3 ∧ z = 3)) := by
sorry

end system_solution_l1751_175180


namespace english_marks_calculation_l1751_175179

def average_marks : ℝ := 70
def num_subjects : ℕ := 5
def math_marks : ℕ := 65
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 67
def biology_marks : ℕ := 85

theorem english_marks_calculation :
  ∃ (english_marks : ℕ),
    (english_marks + math_marks + physics_marks + chemistry_marks + biology_marks : ℝ) / num_subjects = average_marks ∧
    english_marks = 51 := by
  sorry

end english_marks_calculation_l1751_175179


namespace fractional_equation_solution_l1751_175138

theorem fractional_equation_solution :
  ∃ x : ℝ, (x / (x + 2) + 4 / (x^2 - 4) = 1) ∧ (x = 4) := by
  sorry

end fractional_equation_solution_l1751_175138


namespace max_value_expression_l1751_175122

theorem max_value_expression (x y : ℝ) : 
  (2 * x + Real.sqrt 2 * y) / (2 * x^4 + 4 * y^4 + 9) ≤ 1/4 := by
  sorry

end max_value_expression_l1751_175122


namespace no_integer_solutions_l1751_175120

theorem no_integer_solutions : ¬∃ (a b : ℤ), (1 : ℚ) / a + (1 : ℚ) / b = -(1 : ℚ) / (a + b) :=
sorry

end no_integer_solutions_l1751_175120


namespace product_of_roots_quadratic_equation_l1751_175104

theorem product_of_roots_quadratic_equation :
  ∀ x₁ x₂ : ℝ, (x₁^2 + x₁ - 2 = 0) → (x₂^2 + x₂ - 2 = 0) → x₁ * x₂ = -2 := by
  sorry

end product_of_roots_quadratic_equation_l1751_175104


namespace inequality_system_integer_solutions_l1751_175146

theorem inequality_system_integer_solutions :
  {x : ℤ | 2 * x + 1 > 0 ∧ 2 * x ≤ 4} = {0, 1, 2} := by
  sorry

end inequality_system_integer_solutions_l1751_175146


namespace f_properties_l1751_175194

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x - Real.sqrt x + 1
  else if x = 0 then 0
  else x + Real.sqrt (-x) - 1

-- State the properties of f
theorem f_properties :
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (∀ x > 0, f x = x - Real.sqrt x + 1) ∧  -- given definition for x > 0
  (Set.range f = {y | y ≥ 3/4 ∨ y ≤ -3/4 ∨ y = 0}) := by
  sorry

end f_properties_l1751_175194


namespace triangular_grid_theorem_l1751_175102

/-- Represents an infinite triangular grid with black unit equilateral triangles -/
structure TriangularGrid where
  black_triangles : ℕ

/-- Represents an equilateral triangle whose sides align with grid lines -/
structure AlignedTriangle where
  -- Add necessary fields

/-- Checks if there's exactly one black triangle outside the given aligned triangle -/
def has_one_outside (grid : TriangularGrid) (triangle : AlignedTriangle) : Prop :=
  sorry

/-- The main theorem statement -/
theorem triangular_grid_theorem (N : ℕ) :
  N > 0 →
  (∃ (grid : TriangularGrid) (triangle : AlignedTriangle),
    grid.black_triangles = N ∧ has_one_outside grid triangle) ↔
  N = 1 ∨ N = 2 ∨ N = 3 :=
sorry

end triangular_grid_theorem_l1751_175102


namespace min_exposed_surface_area_l1751_175111

/- Define a rectangular solid -/
structure RectangularSolid where
  length : ℝ
  width : ℝ
  height : ℝ
  volume_eq : length * width * height = 128
  positive : length > 0 ∧ width > 0 ∧ height > 0

/- Define the three solids -/
def solid1 : RectangularSolid := {
  length := 4,
  width := 1,
  height := 32,
  volume_eq := by norm_num,
  positive := by simp
}

def solid2 : RectangularSolid := {
  length := 8,
  width := 8,
  height := 2,
  volume_eq := by norm_num,
  positive := by simp
}

def solid3 : RectangularSolid := {
  length := 4,
  width := 2,
  height := 16,
  volume_eq := by norm_num,
  positive := by simp
}

/- Calculate the exposed surface area of the tower -/
def exposedSurfaceArea (s1 s2 s3 : RectangularSolid) : ℝ :=
  2 * (s1.length * s1.width + s2.length * s2.width + s3.length * s3.width) +
  2 * (s1.length * s1.height + s2.length * s2.height + s3.length * s3.height) +
  2 * (s1.width * s1.height + s2.width * s2.height + s3.width * s3.height) -
  2 * (s1.length * s1.width + s2.length * s2.width)

/- Theorem statement -/
theorem min_exposed_surface_area :
  exposedSurfaceArea solid1 solid2 solid3 = 832 := by sorry

end min_exposed_surface_area_l1751_175111


namespace triangle_abc_properties_l1751_175101

/-- Theorem about a triangle ABC with specific side lengths and angle properties -/
theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  -- Given conditions
  a = Real.sqrt 2 →
  b = 2 →
  Real.sin B + Real.cos B = Real.sqrt 2 →
  -- Triangle inequality and positive side lengths
  a + b > c ∧ b + c > a ∧ c + a > b →
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- A, B, C form angles of a triangle
  A + B + C = π →
  -- Side lengths correspond to opposite angles
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  -- Conclusions
  A = π / 6 ∧
  (1 / 2) * a * b * Real.sin C = (1 + Real.sqrt 3) / 2 :=
by sorry

end triangle_abc_properties_l1751_175101


namespace distance_to_school_is_two_prove_distance_to_school_l1751_175136

/-- The distance to school in miles -/
def distance_to_school : ℝ := 2

/-- Jerry's one-way trip time in minutes -/
def jerry_one_way_time : ℝ := 15

/-- Carson's speed in miles per hour -/
def carson_speed : ℝ := 8

/-- Theorem stating that the distance to school is 2 miles -/
theorem distance_to_school_is_two :
  distance_to_school = 2 :=
by
  sorry

/-- Lemma: Jerry's round trip time equals Carson's one-way trip time -/
lemma jerry_round_trip_equals_carson_one_way :
  2 * jerry_one_way_time = distance_to_school / (carson_speed / 60) :=
by
  sorry

/-- Main theorem proving the distance to school -/
theorem prove_distance_to_school :
  distance_to_school = 2 :=
by
  sorry

end distance_to_school_is_two_prove_distance_to_school_l1751_175136


namespace infinitely_many_divisible_powers_l1751_175100

theorem infinitely_many_divisible_powers (a b c : ℤ) 
  (h : (a + b + c) ∣ (a^2 + b^2 + c^2)) :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, (a + b + c) ∣ (a^n + b^n + c^n) :=
by sorry

end infinitely_many_divisible_powers_l1751_175100


namespace problem_solution_l1751_175158

theorem problem_solution : 
  (5 * Real.sqrt 2 - (Real.sqrt 18 + Real.sqrt (1/2)) = (3/2) * Real.sqrt 2) ∧
  ((2 * Real.sqrt 3 - 1)^2 + Real.sqrt 24 / Real.sqrt 2 = 13 - 2 * Real.sqrt 3) := by
  sorry

end problem_solution_l1751_175158


namespace rectangle_ellipse_perimeter_l1751_175155

/-- Given a rectangle and an ellipse with specific properties, prove that the perimeter of the rectangle is 450. -/
theorem rectangle_ellipse_perimeter :
  ∀ (x y : ℝ) (a b : ℝ),
  -- Rectangle conditions
  x * y = 2500 ∧
  x / y = 5 / 4 ∧
  -- Ellipse conditions
  π * a * b = 2500 * π ∧
  x + y = 2 * a ∧
  (x^2 + y^2 : ℝ) = 4 * (a^2 - b^2) →
  -- Conclusion
  2 * (x + y) = 450 := by
sorry

end rectangle_ellipse_perimeter_l1751_175155


namespace set_operations_l1751_175162

theorem set_operations (A B : Set ℕ) (hA : A = {3, 5, 6, 8}) (hB : B = {4, 5, 7, 8}) :
  (A ∩ B = {5, 8}) ∧ (A ∪ B = {3, 4, 5, 6, 7, 8}) := by
  sorry

end set_operations_l1751_175162


namespace parabola_parameter_l1751_175137

/-- For a parabola with equation y^2 = 4ax and directrix x = -2, the value of a is 2. -/
theorem parabola_parameter (y x a : ℝ) : 
  (∀ y x, y^2 = 4*a*x) →  -- Equation of the parabola
  (∀ x, x = -2 → x = x) →  -- Equation of the directrix (x = -2 represented as a predicate)
  a = 2 := by
sorry

end parabola_parameter_l1751_175137


namespace gcd_equality_pairs_l1751_175105

theorem gcd_equality_pairs :
  ∀ a b : ℕ+, a ≤ b →
  (∀ x : ℕ+, Nat.gcd x a * Nat.gcd x b = Nat.gcd x 20 * Nat.gcd x 22) →
  ((a = 2 ∧ b = 220) ∨ (a = 4 ∧ b = 110) ∨ (a = 10 ∧ b = 44) ∨ (a = 20 ∧ b = 22)) :=
by sorry

end gcd_equality_pairs_l1751_175105


namespace min_divisions_is_48_l1751_175110

/-- Represents a cell division strategy -/
structure DivisionStrategy where
  div42 : ℕ  -- number of divisions resulting in 42 cells
  div44 : ℕ  -- number of divisions resulting in 44 cells

/-- The number of cells after applying a division strategy -/
def resultingCells (s : DivisionStrategy) : ℕ :=
  1 + 41 * s.div42 + 43 * s.div44

/-- A division strategy is valid if it results in exactly 1993 cells -/
def isValidStrategy (s : DivisionStrategy) : Prop :=
  resultingCells s = 1993

/-- The total number of divisions in a strategy -/
def totalDivisions (s : DivisionStrategy) : ℕ :=
  s.div42 + s.div44

/-- There exists a valid division strategy -/
axiom exists_valid_strategy : ∃ s : DivisionStrategy, isValidStrategy s

/-- The minimum number of divisions needed is 48 -/
theorem min_divisions_is_48 :
  ∃ s : DivisionStrategy, isValidStrategy s ∧
    totalDivisions s = 48 ∧
    ∀ t : DivisionStrategy, isValidStrategy t → totalDivisions s ≤ totalDivisions t :=
  sorry

end min_divisions_is_48_l1751_175110


namespace constant_value_theorem_l1751_175130

/-- Given constants a and b, if f(x) = x^2 + 4x + 3 and f(ax + b) = x^2 + 10x + 24, then 5a - b = 2 -/
theorem constant_value_theorem (a b : ℝ) : 
  (∀ x, (x^2 + 4*x + 3 : ℝ) = ((a*x + b)^2 + 4*(a*x + b) + 3 : ℝ)) → 
  (∀ x, (x^2 + 4*x + 3 : ℝ) = (x^2 + 10*x + 24 : ℝ)) → 
  (5*a - b : ℝ) = 2 := by
  sorry

end constant_value_theorem_l1751_175130


namespace orange_harvest_problem_l1751_175131

/-- Proves that the number of sacks harvested per day is 66, given the conditions of the orange harvest problem. -/
theorem orange_harvest_problem (oranges_per_sack : ℕ) (harvest_days : ℕ) (total_oranges : ℕ) 
  (h1 : oranges_per_sack = 25)
  (h2 : harvest_days = 87)
  (h3 : total_oranges = 143550) :
  total_oranges / (oranges_per_sack * harvest_days) = 66 := by
  sorry

#eval 143550 / (25 * 87)  -- Should output 66

end orange_harvest_problem_l1751_175131


namespace right_triangle_side_length_l1751_175121

theorem right_triangle_side_length 
  (area : ℝ) 
  (side1 : ℝ) 
  (is_right_triangle : Bool) 
  (h1 : area = 8) 
  (h2 : side1 = Real.sqrt 10) 
  (h3 : is_right_triangle = true) : 
  ∃ side2 : ℝ, side2 = 1.6 * Real.sqrt 10 ∧ (1/2) * side1 * side2 = area :=
sorry

end right_triangle_side_length_l1751_175121


namespace certain_number_exists_and_unique_l1751_175132

theorem certain_number_exists_and_unique : 
  ∃! x : ℝ, x / 5 + x + 5 = 65 := by sorry

end certain_number_exists_and_unique_l1751_175132


namespace chord_equation_of_ellipse_l1751_175106

/-- The equation of a line passing through a chord of an ellipse -/
theorem chord_equation_of_ellipse (x₁ y₁ x₂ y₂ : ℝ) :
  (x₁^2 / 36 + y₁^2 / 9 = 1) →  -- Point (x₁, y₁) is on the ellipse
  (x₂^2 / 36 + y₂^2 / 9 = 1) →  -- Point (x₂, y₂) is on the ellipse
  ((x₁ + x₂) / 2 = 2) →         -- Midpoint x-coordinate is 2
  ((y₁ + y₂) / 2 = 2) →         -- Midpoint y-coordinate is 2
  ∃ (x y : ℝ), x + 4*y - 10 = 0  -- Equation of the line
  := by sorry

end chord_equation_of_ellipse_l1751_175106


namespace vector_at_negative_one_l1751_175133

/-- A line parameterized by t in 3D space -/
structure ParametricLine where
  -- The vector on the line at t = 0
  origin : Fin 3 → ℝ
  -- The vector on the line at t = 1
  point_at_one : Fin 3 → ℝ

/-- The vector on the line at a given t -/
def vector_at_t (line : ParametricLine) (t : ℝ) : Fin 3 → ℝ :=
  λ i => line.origin i + t * (line.point_at_one i - line.origin i)

/-- The theorem stating the vector at t = -1 for the given line -/
theorem vector_at_negative_one (line : ParametricLine) 
  (h0 : line.origin = λ i => [2, 4, 9].get i)
  (h1 : line.point_at_one = λ i => [3, 1, 5].get i) :
  vector_at_t line (-1) = λ i => [1, 7, 13].get i := by
  sorry

end vector_at_negative_one_l1751_175133


namespace probability_x_plus_y_less_than_4_l1751_175171

/-- A square with vertices at (0, 0), (0, 3), (3, 3), and (3, 0) -/
def Square : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3}

/-- The region where x + y < 4 within the square -/
def RegionXPlusYLessThan4 : Set (ℝ × ℝ) :=
  {p ∈ Square | p.1 + p.2 < 4}

/-- The area of the square -/
def squareArea : ℝ := 9

/-- The area of the region where x + y < 4 within the square -/
def regionArea : ℝ := 7

theorem probability_x_plus_y_less_than_4 :
  (regionArea / squareArea : ℝ) = 7 / 9 := by sorry

end probability_x_plus_y_less_than_4_l1751_175171


namespace temperature_decrease_l1751_175112

theorem temperature_decrease (initial_temp final_temp decrease : ℤ) :
  initial_temp = -3 →
  decrease = 6 →
  final_temp = initial_temp - decrease →
  final_temp = -9 :=
by sorry

end temperature_decrease_l1751_175112


namespace first_discount_percentage_l1751_175166

theorem first_discount_percentage (original_price : ℝ) (final_price : ℝ) (second_discount : ℝ) :
  original_price = 340 →
  final_price = 231.2 →
  second_discount = 0.15 →
  ∃ (first_discount : ℝ),
    first_discount = 0.2 ∧
    final_price = original_price * (1 - first_discount) * (1 - second_discount) :=
by sorry

end first_discount_percentage_l1751_175166


namespace paige_pencils_l1751_175128

theorem paige_pencils (initial_pencils : ℕ) (used_pencils : ℕ) : 
  initial_pencils = 94 → used_pencils = 3 → initial_pencils - used_pencils = 91 := by
  sorry

end paige_pencils_l1751_175128


namespace find_a_value_l1751_175192

/-- Given sets A and B, prove that a is either -2/3 or -7/4 -/
theorem find_a_value (x : ℝ) (a : ℝ) : 
  let A : Set ℝ := {1, 2, x^2 - 5*x + 9}
  let B : Set ℝ := {3, x^2 + a*x + a}
  A = {1, 2, 3} → 2 ∈ B → (a = -2/3 ∨ a = -7/4) := by
sorry

end find_a_value_l1751_175192


namespace quadratic_equation_solution_l1751_175152

theorem quadratic_equation_solution : ∃ (x₁ x₂ : ℝ), 
  x₁^2 - 6*x₁ + 8 = 0 ∧ 
  x₂^2 - 6*x₂ + 8 = 0 ∧ 
  x₁ = 2 ∧ 
  x₂ = 4 := by
  sorry

end quadratic_equation_solution_l1751_175152


namespace apartment_number_theorem_l1751_175182

/-- The number of apartments on each floor (actual) -/
def apartments_per_floor : ℕ := 7

/-- The number of apartments Anna initially thought were on each floor -/
def assumed_apartments_per_floor : ℕ := 6

/-- The floor number where Anna's apartment is located -/
def target_floor : ℕ := 4

/-- The set of possible apartment numbers on the target floor when there are 6 apartments per floor -/
def apartment_numbers_6 : Set ℕ := Set.Icc ((target_floor - 1) * assumed_apartments_per_floor + 1) (target_floor * assumed_apartments_per_floor)

/-- The set of possible apartment numbers on the target floor when there are 7 apartments per floor -/
def apartment_numbers_7 : Set ℕ := Set.Icc ((target_floor - 1) * apartments_per_floor + 1) (target_floor * apartments_per_floor)

/-- The set of apartment numbers that exist in both scenarios -/
def possible_apartment_numbers : Set ℕ := apartment_numbers_6 ∩ apartment_numbers_7

theorem apartment_number_theorem : possible_apartment_numbers = {22, 23, 24} := by
  sorry

end apartment_number_theorem_l1751_175182


namespace x_squared_eq_one_is_quadratic_l1751_175184

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 = 1 -/
def f (x : ℝ) : ℝ := x^2 - 1

/-- Theorem: x^2 = 1 is a quadratic equation in one variable -/
theorem x_squared_eq_one_is_quadratic : is_quadratic_equation f := by
  sorry

end x_squared_eq_one_is_quadratic_l1751_175184


namespace triangle_inequality_l1751_175114

/-- Given a triangle with sides a, b, and c, and s = (a+b+c)/2, 
    if s^2 = 2ab, then s < 2a -/
theorem triangle_inequality (a b c s : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b)
  (h_s_def : s = (a + b + c) / 2)
  (h_s_sq : s^2 = 2*a*b) : 
  s < 2*a := by
  sorry

end triangle_inequality_l1751_175114


namespace negative_three_a_plus_two_a_equals_negative_a_l1751_175117

theorem negative_three_a_plus_two_a_equals_negative_a (a : ℝ) : -3*a + 2*a = -a := by
  sorry

end negative_three_a_plus_two_a_equals_negative_a_l1751_175117


namespace shorter_side_is_eight_l1751_175135

/-- A rectangle with given area and perimeter -/
structure Rectangle where
  length : ℝ
  width : ℝ
  area_eq : length * width = 104
  perimeter_eq : 2 * (length + width) = 42

/-- The shorter side of the rectangle is 8 feet -/
theorem shorter_side_is_eight (r : Rectangle) : min r.length r.width = 8 := by
  sorry

end shorter_side_is_eight_l1751_175135


namespace chess_tournament_square_players_l1751_175175

/-- Represents a chess tournament with men and women players -/
structure ChessTournament where
  k : ℕ  -- number of men
  m : ℕ  -- number of women

/-- The total number of players in the tournament -/
def ChessTournament.totalPlayers (t : ChessTournament) : ℕ := t.k + t.m

/-- The condition that total points scored by men against women equals total points scored by women against men -/
def ChessTournament.equalCrossScores (t : ChessTournament) : Prop :=
  (t.k * (t.k - 1)) / 2 + (t.m * (t.m - 1)) / 2 = t.k * t.m

theorem chess_tournament_square_players (t : ChessTournament) 
  (h : t.equalCrossScores) : 
  ∃ n : ℕ, t.totalPlayers = n^2 := by
  sorry


end chess_tournament_square_players_l1751_175175


namespace angle_c_90_sufficient_not_necessary_l1751_175183

/-- Triangle ABC with angles A, B, and C -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  angle_sum : A + B + C = π

/-- Theorem stating that in a triangle ABC, angle C = 90° is a sufficient 
    but not necessary condition for cos A + sin A = cos B + sin B -/
theorem angle_c_90_sufficient_not_necessary (t : Triangle) :
  (t.C = π / 2 → Real.cos t.A + Real.sin t.A = Real.cos t.B + Real.sin t.B) ∧
  ∃ t' : Triangle, Real.cos t'.A + Real.sin t'.A = Real.cos t'.B + Real.sin t'.B ∧ t'.C ≠ π / 2 :=
sorry

end angle_c_90_sufficient_not_necessary_l1751_175183


namespace unique_four_digit_square_with_property_l1751_175188

theorem unique_four_digit_square_with_property : ∃! n : ℕ,
  (1000 ≤ n ∧ n ≤ 9999) ∧  -- four-digit number
  (∃ m : ℕ, n = m^2) ∧     -- perfect square
  (n / 100 = 3 * (n % 100) + 1) ∧  -- satisfies the equation
  n = 2809 := by
sorry

end unique_four_digit_square_with_property_l1751_175188


namespace cookie_distribution_l1751_175186

theorem cookie_distribution (num_people : ℕ) (cookies_per_person : ℕ) 
  (h1 : num_people = 5)
  (h2 : cookies_per_person = 7) : 
  num_people * cookies_per_person = 35 := by
sorry

end cookie_distribution_l1751_175186


namespace johns_number_l1751_175165

theorem johns_number : ∃! n : ℕ, 1000 < n ∧ n < 3000 ∧ 64 ∣ n ∧ 45 ∣ n ∧ n = 2880 := by
  sorry

end johns_number_l1751_175165


namespace absolute_value_square_l1751_175142

theorem absolute_value_square (a b : ℝ) : a > |b| → a^2 > b^2 := by
  sorry

end absolute_value_square_l1751_175142


namespace day_crew_loading_fraction_l1751_175145

/-- The fraction of boxes loaded by the day crew given the relative capacities of night and day crews -/
theorem day_crew_loading_fraction 
  (D : ℝ) -- number of boxes loaded by each day crew worker
  (W : ℝ) -- number of workers in the day crew
  (h1 : D > 0) -- assume positive number of boxes
  (h2 : W > 0) -- assume positive number of workers
  : (D * W) / ((D * W) + ((3/4 * D) * (5/6 * W))) = 8/13 := by
  sorry

end day_crew_loading_fraction_l1751_175145


namespace repeating_decimal_three_three_six_l1751_175126

/-- Represents a repeating decimal where the decimal part repeats infinitely -/
def RepeatingDecimal (whole : ℤ) (repeating : ℕ) : ℚ :=
  whole + (repeating : ℚ) / (99 : ℚ)

/-- The statement that 3.363636... equals 37/11 -/
theorem repeating_decimal_three_three_six : RepeatingDecimal 3 36 = 37 / 11 := by
  sorry

end repeating_decimal_three_three_six_l1751_175126


namespace sum_always_four_digits_l1751_175147

theorem sum_always_four_digits :
  ∀ (A B : ℕ), 1 ≤ A ∧ A ≤ 9 → 1 ≤ B ∧ B ≤ 9 →
  ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ n = 7654 + (900 + 10 * A + 7) + (10 + B) :=
by sorry

end sum_always_four_digits_l1751_175147


namespace bus_profit_maximization_l1751_175172

/-- The profit function for a bus operating for x years -/
def profit (x : ℕ+) : ℚ := -x^2 + 18*x - 36

/-- The average profit function for a bus operating for x years -/
def avgProfit (x : ℕ+) : ℚ := (profit x) / x

theorem bus_profit_maximization :
  (∃ (x : ℕ+), ∀ (y : ℕ+), profit x ≥ profit y) ∧
  (∃ (x : ℕ+), profit x = 45) ∧
  (∃ (x : ℕ+), ∀ (y : ℕ+), avgProfit x ≥ avgProfit y) ∧
  (∃ (x : ℕ+), avgProfit x = 6) :=
sorry

end bus_profit_maximization_l1751_175172


namespace only_three_four_five_is_right_triangle_l1751_175119

def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem only_three_four_five_is_right_triangle :
  (¬ is_right_triangle 1 2 3) ∧
  (¬ is_right_triangle 2 3 4) ∧
  (is_right_triangle 3 4 5) ∧
  (¬ is_right_triangle 1 2 3) :=
sorry

end only_three_four_five_is_right_triangle_l1751_175119


namespace geometric_sequence_general_term_l1751_175140

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_general_term
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_first : a 1 = 2)
  (h_second : a 2 = 4)
  (h_ineq : ∀ x : ℝ, -x^2 + 6*x - 8 > 0 ↔ 2 < x ∧ x < 4) :
  ∀ n : ℕ, a n = 2^n :=
sorry

end geometric_sequence_general_term_l1751_175140


namespace flour_sugar_difference_l1751_175107

/-- Given a recipe and current ingredients, calculate the difference between additional flour needed and total sugar needed. -/
theorem flour_sugar_difference 
  (total_flour : ℕ) 
  (total_sugar : ℕ) 
  (added_flour : ℕ) 
  (h1 : total_flour = 10) 
  (h2 : total_sugar = 2) 
  (h3 : added_flour = 7) : 
  (total_flour - added_flour) - total_sugar = 1 := by
  sorry

#check flour_sugar_difference

end flour_sugar_difference_l1751_175107


namespace sunday_to_saturday_ratio_l1751_175149

/-- Tameka's cracker box sales over three days -/
structure CrackerSales where
  friday : ℕ
  saturday : ℕ
  sunday : ℕ
  total : ℕ
  h1 : friday = 40
  h2 : saturday = 2 * friday - 10
  h3 : total = friday + saturday + sunday
  h4 : total = 145

/-- The ratio of boxes sold on Sunday to boxes sold on Saturday is 1/2 -/
theorem sunday_to_saturday_ratio (sales : CrackerSales) :
  sales.sunday / sales.saturday = 1 / 2 := by
  sorry

end sunday_to_saturday_ratio_l1751_175149


namespace distribute_five_balls_four_boxes_l1751_175109

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: There are 4^5 ways to distribute 5 distinguishable balls into 4 distinguishable boxes -/
theorem distribute_five_balls_four_boxes :
  distribute_balls 5 4 = 4^5 := by
  sorry

end distribute_five_balls_four_boxes_l1751_175109


namespace abc_divides_sum_power_seven_l1751_175168

theorem abc_divides_sum_power_seven
  (a b c : ℕ+)
  (h1 : a ∣ b^2)
  (h2 : b ∣ c^2)
  (h3 : c ∣ a^2) :
  (a * b * c) ∣ (a + b + c)^7 :=
by sorry

end abc_divides_sum_power_seven_l1751_175168


namespace prism_intersection_area_l1751_175190

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents a rectangular prism -/
structure RectangularPrism where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Calculate the area of intersection between a rectangular prism and a plane -/
def intersectionArea (prism : RectangularPrism) (plane : Plane) : ℝ :=
  sorry

theorem prism_intersection_area :
  let prism : RectangularPrism := ⟨8, 12, 0⟩  -- height is arbitrary, set to 0
  let plane : Plane := ⟨3, -5, 6, 30⟩
  intersectionArea prism plane = 64.92 := by sorry

end prism_intersection_area_l1751_175190


namespace simplify_expression_l1751_175163

theorem simplify_expression : (-5) + (-6) - (-5) + 4 = -5 - 6 + 5 + 4 := by
  sorry

end simplify_expression_l1751_175163


namespace eagle_types_total_l1751_175103

theorem eagle_types_total (types_per_section : ℕ) (num_sections : ℕ) (h1 : types_per_section = 6) (h2 : num_sections = 3) :
  types_per_section * num_sections = 18 := by
  sorry

end eagle_types_total_l1751_175103


namespace midpoint_coordinate_sum_l1751_175154

/-- The sum of the coordinates of the midpoint of a segment with endpoints (10, 7) and (4, -3) is 9 -/
theorem midpoint_coordinate_sum : 
  let p1 : ℝ × ℝ := (10, 7)
  let p2 : ℝ × ℝ := (4, -3)
  let midpoint : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  midpoint.1 + midpoint.2 = 9 := by sorry

end midpoint_coordinate_sum_l1751_175154
