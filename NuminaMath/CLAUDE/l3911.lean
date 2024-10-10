import Mathlib

namespace team_average_score_l3911_391149

theorem team_average_score (lefty_score : ℕ) (righty_score : ℕ) (other_score : ℕ) :
  lefty_score = 20 →
  righty_score = lefty_score / 2 →
  other_score = 6 * righty_score →
  (lefty_score + righty_score + other_score) / 3 = 30 := by
  sorry

end team_average_score_l3911_391149


namespace dog_weight_fraction_is_one_fourth_l3911_391123

/-- Represents the capacity and weight scenario of Penny's canoe --/
structure CanoeScenario where
  capacity : ℕ              -- Capacity without dog
  capacityWithDog : ℚ       -- Fraction of capacity with dog
  personWeight : ℕ          -- Weight of each person in pounds
  totalWeightWithDog : ℕ    -- Total weight in canoe with dog and people

/-- Calculates the dog's weight as a fraction of a person's weight --/
def dogWeightFraction (scenario : CanoeScenario) : ℚ :=
  let peopleWithDog := ⌊scenario.capacity * scenario.capacityWithDog⌋
  let peopleWeight := peopleWithDog * scenario.personWeight
  let dogWeight := scenario.totalWeightWithDog - peopleWeight
  dogWeight / scenario.personWeight

/-- Theorem stating that the dog's weight is 1/4 of a person's weight --/
theorem dog_weight_fraction_is_one_fourth (scenario : CanoeScenario) 
  (h1 : scenario.capacity = 6)
  (h2 : scenario.capacityWithDog = 2/3)
  (h3 : scenario.personWeight = 140)
  (h4 : scenario.totalWeightWithDog = 595) :
  dogWeightFraction scenario = 1/4 := by
  sorry


end dog_weight_fraction_is_one_fourth_l3911_391123


namespace job_completion_time_l3911_391194

/-- Given that 5/8 of a job is completed in 10 days at a constant pace, 
    prove that the entire job will be completed in 16 days. -/
theorem job_completion_time (days_for_part : ℚ) (part_completed : ℚ) (total_days : ℕ) : 
  days_for_part = 10 → part_completed = 5/8 → total_days = 16 := by
  sorry

end job_completion_time_l3911_391194


namespace luna_budget_sum_l3911_391167

def luna_budget (house_rental food phone_bill : ℝ) : Prop :=
  food = 0.6 * house_rental ∧
  phone_bill = 0.1 * food ∧
  house_rental + food + phone_bill = 249

theorem luna_budget_sum (house_rental food phone_bill : ℝ) :
  luna_budget house_rental food phone_bill →
  house_rental + food = 240 :=
by
  sorry

end luna_budget_sum_l3911_391167


namespace squared_inequality_l3911_391185

theorem squared_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > a*b ∧ a*b > b^2 := by
  sorry

end squared_inequality_l3911_391185


namespace dataset_growth_percentage_l3911_391128

theorem dataset_growth_percentage (initial_size : ℕ) (final_size : ℕ) : 
  initial_size = 200 →
  final_size = 180 →
  ∃ (growth_percentage : ℚ),
    growth_percentage = 20 ∧
    (3/4 : ℚ) * (initial_size + initial_size * (growth_percentage / 100)) = final_size :=
by sorry

end dataset_growth_percentage_l3911_391128


namespace quadratic_equation_solutions_l3911_391100

theorem quadratic_equation_solutions : ∃ (k : ℝ),
  (∀ (x : ℝ), k * x^2 - 7 * x - 6 = 0 ↔ (x = 2 ∨ x = -3/2)) ∧ k = 14 := by
  sorry

end quadratic_equation_solutions_l3911_391100


namespace det_B_is_one_l3911_391108

open Matrix

def B (x y : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![x, 2; -3, y]

theorem det_B_is_one (x y : ℝ) (h : B x y + (B x y)⁻¹ = 0) : 
  det (B x y) = 1 := by
  sorry

end det_B_is_one_l3911_391108


namespace all_diagonal_triangles_multiplicative_l3911_391141

/-- A triangle is multiplicative if the product of the lengths of two of its sides
    is equal to the length of the third side. -/
def IsMultiplicative (a b c : ℝ) : Prop :=
  a * b = c ∨ b * c = a ∨ c * a = b

/-- A regular polygon with n sides, all of length 1. -/
structure RegularPolygon (n : ℕ) where
  (n_ge_3 : n ≥ 3)
  (side_length : ℝ := 1)

/-- A triangle formed by two adjacent vertices and a diagonal in a regular polygon. -/
structure DiagonalTriangle (n : ℕ) where
  (polygon : RegularPolygon n)
  (k : ℕ)
  (k_lt_n : k < n)

/-- The lengths of the sides of a diagonal triangle. -/
def DiagonalTriangleSides (n : ℕ) (t : DiagonalTriangle n) : ℝ × ℝ × ℝ :=
  sorry

theorem all_diagonal_triangles_multiplicative (n : ℕ) (h : n ≥ 3) :
  ∀ (t : DiagonalTriangle n), 
    let (a, b, c) := DiagonalTriangleSides n t
    IsMultiplicative a b c :=
  sorry

end all_diagonal_triangles_multiplicative_l3911_391141


namespace logical_equivalences_l3911_391181

theorem logical_equivalences (x y : ℝ) : True := by
  have original : x + y = 5 → x = 3 ∧ y = 2 := sorry
  
  -- Converse
  have converse : x = 3 ∧ y = 2 → x + y = 5 := sorry
  
  -- Inverse
  have inverse : x + y ≠ 5 → x ≠ 3 ∨ y ≠ 2 := sorry
  
  -- Contrapositive
  have contrapositive : x ≠ 3 ∨ y ≠ 2 → x + y ≠ 5 := sorry
  
  -- Truth values
  have converse_true : ∀ x y, (x = 3 ∧ y = 2 → x + y = 5) := sorry
  have inverse_true : ∀ x y, (x + y ≠ 5 → x ≠ 3 ∨ y ≠ 2) := sorry
  have contrapositive_false : ¬(∀ x y, (x ≠ 3 ∨ y ≠ 2 → x + y ≠ 5)) := sorry
  
  sorry

#check logical_equivalences

end logical_equivalences_l3911_391181


namespace river_current_speed_l3911_391116

/-- The speed of the river's current in miles per hour -/
def river_speed : ℝ := 9.8

/-- The distance traveled downstream and upstream in miles -/
def distance : ℝ := 21

/-- The woman's initial canoeing speed in still water (miles per hour) -/
noncomputable def initial_speed : ℝ := 
  Real.sqrt (river_speed^2 + 7 * river_speed)

/-- Time difference between upstream and downstream journeys in hours -/
def time_difference : ℝ := 3

/-- Time difference after increasing paddling speed by 50% in hours -/
def reduced_time_difference : ℝ := 0.75

theorem river_current_speed :
  (distance / (initial_speed + river_speed) + time_difference 
    = distance / (initial_speed - river_speed)) ∧
  (distance / (1.5 * initial_speed + river_speed) + reduced_time_difference 
    = distance / (1.5 * initial_speed - river_speed)) :=
sorry

end river_current_speed_l3911_391116


namespace sphere_radius_in_truncated_cone_l3911_391144

/-- A truncated cone with a tangent sphere -/
structure TruncatedConeWithSphere where
  bottom_radius : ℝ
  top_radius : ℝ
  sphere_radius : ℝ
  is_tangent : Bool

/-- The theorem stating the radius of the sphere tangent to a truncated cone -/
theorem sphere_radius_in_truncated_cone 
  (cone : TruncatedConeWithSphere) 
  (h1 : cone.bottom_radius = 24) 
  (h2 : cone.top_radius = 6) 
  (h3 : cone.is_tangent = true) : 
  cone.sphere_radius = 12 := by
  sorry

#check sphere_radius_in_truncated_cone

end sphere_radius_in_truncated_cone_l3911_391144


namespace inequality_proof_l3911_391184

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_one : a + b + c = 1) :
  Real.sqrt (a * b / (a * b + c)) + Real.sqrt (b * c / (b * c + a)) + Real.sqrt (c * a / (c * a + b)) ≤ 3 / 2 := by
  sorry

end inequality_proof_l3911_391184


namespace greatest_integer_of_fraction_l3911_391140

theorem greatest_integer_of_fraction (x : ℝ) : 
  x = (5^150 + 3^150) / (5^147 + 3^147) → 
  ⌊x⌋ = 124 := by sorry

end greatest_integer_of_fraction_l3911_391140


namespace seven_not_spheric_spheric_power_is_spheric_l3911_391177

/-- A rational number is spheric if it is the sum of three squares of rational numbers. -/
def is_spheric (r : ℚ) : Prop :=
  ∃ x y z : ℚ, r = x^2 + y^2 + z^2

theorem seven_not_spheric : ¬ is_spheric 7 := by
  sorry

theorem spheric_power_is_spheric (r : ℚ) (n : ℕ) (hn : n > 1) :
  is_spheric r → is_spheric (r^n) := by
  sorry

end seven_not_spheric_spheric_power_is_spheric_l3911_391177


namespace minimum_value_inequality_l3911_391118

theorem minimum_value_inequality (x y z w : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hw : w ≠ 0) :
  (6 * x * y + 5 * y * z + 6 * z * w) / (x^2 + y^2 + z^2 + w^2) ≤ 9/2 := by
  sorry

end minimum_value_inequality_l3911_391118


namespace sqrt_sum_problem_l3911_391125

theorem sqrt_sum_problem (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) : 
  Real.sqrt x + 1 / Real.sqrt x = 2 * Real.sqrt 13 := by
  sorry

end sqrt_sum_problem_l3911_391125


namespace fourth_equation_in_sequence_l3911_391187

/-- Given a sequence of equations, prove that the fourth equation follows the pattern. -/
theorem fourth_equation_in_sequence : 
  (3^2 + 4^2 = 5^2) → 
  (10^2 + 11^2 + 12^2 = 13^2 + 14^2) → 
  (21^2 + 22^2 + 23^2 + 24^2 = 25^2 + 26^2 + 27^2) → 
  (36^2 + 37^2 + 38^2 + 39^2 + 40^2 = 41^2 + 42^2 + 43^2 + 44^2) := by
  sorry

end fourth_equation_in_sequence_l3911_391187


namespace min_sum_of_valid_set_l3911_391162

def is_valid_set (s : Finset ℕ) : Prop :=
  s.card = 10 ∧ 
  (∀ t ⊆ s, t.card = 5 → Even (t.prod id)) ∧
  Odd (s.sum id)

theorem min_sum_of_valid_set :
  ∃ (s : Finset ℕ), is_valid_set s ∧ 
  (∀ t : Finset ℕ, is_valid_set t → s.sum id ≤ t.sum id) ∧
  s.sum id = 65 := by
sorry

end min_sum_of_valid_set_l3911_391162


namespace digit_at_573_l3911_391107

/-- The decimal representation of 11/37 -/
def decimal_rep : ℚ := 11 / 37

/-- The length of the repeating sequence in the decimal representation of 11/37 -/
def period : ℕ := 3

/-- The repeating sequence in the decimal representation of 11/37 -/
def repeating_sequence : Fin 3 → ℕ
| 0 => 2
| 1 => 9
| 2 => 7

/-- The position we're interested in -/
def target_position : ℕ := 573

theorem digit_at_573 : 
  (repeating_sequence (target_position % period : Fin 3)) = 7 := by
  sorry

end digit_at_573_l3911_391107


namespace prism_18_edges_8_faces_l3911_391115

/-- A prism is a three-dimensional shape with two identical ends (bases) and flat sides. -/
structure Prism where
  edges : ℕ

/-- The number of faces in a prism. -/
def num_faces (p : Prism) : ℕ :=
  let lateral_faces := p.edges / 3
  lateral_faces + 2

/-- Theorem: A prism with 18 edges has 8 faces. -/
theorem prism_18_edges_8_faces :
  ∀ (p : Prism), p.edges = 18 → num_faces p = 8 := by
  sorry

end prism_18_edges_8_faces_l3911_391115


namespace fraction_inequality_l3911_391137

theorem fraction_inequality (a b : ℝ) (ha : a ≠ 0) (ha1 : a + 1 ≠ 0) :
  ¬(∀ a b, b / a = (b + 1) / (a + 1)) :=
sorry

end fraction_inequality_l3911_391137


namespace subtract_and_add_l3911_391189

theorem subtract_and_add : (3005 - 3000) + 10 = 15 := by
  sorry

end subtract_and_add_l3911_391189


namespace sams_remaining_dimes_l3911_391151

/-- Given Sam's initial number of dimes and the number borrowed by his sister and friend,
    prove that the remaining number of dimes is correct. -/
theorem sams_remaining_dimes (initial_dimes sister_borrowed friend_borrowed : ℕ) :
  initial_dimes = 8 ∧ sister_borrowed = 4 ∧ friend_borrowed = 2 →
  initial_dimes - (sister_borrowed + friend_borrowed) = 2 :=
by sorry

end sams_remaining_dimes_l3911_391151


namespace fraction_equality_l3911_391150

theorem fraction_equality (P Q : ℝ) : 
  (∀ x : ℝ, x ≠ -6 ∧ x ≠ 0 ∧ x ≠ 5 → 
    P / (x + 6) + Q / (x^2 - 5*x) = (x^2 - 3*x + 15) / (x^3 + x^2 - 30*x)) ↔ 
  (P = 1 ∧ Q = 5/2) :=
sorry

end fraction_equality_l3911_391150


namespace number_of_students_l3911_391170

theorem number_of_students (n : ℕ) : 
  n % 5 = 0 ∧ 
  ∃ t : ℕ, (n + 1) * t = 527 ∧
  (n + 1) ∣ 527 ∧
  (n + 1) % 5 = 1 →
  n = 30 := by
sorry

end number_of_students_l3911_391170


namespace problem_statement_l3911_391193

theorem problem_statement (x y : ℝ) (h : 3 * x^2 + 3 * y^2 - 2 * x * y = 5) :
  (x + y ≥ -Real.sqrt 5) ∧
  (x^2 + y^2 ≥ 5/4) ∧
  (x - y/3 ≥ -Real.sqrt 15 / 3) :=
by sorry

end problem_statement_l3911_391193


namespace intersection_parallel_to_l_l3911_391130

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_line : Line → Line → Prop)

-- Define the parallel relation between lines and planes
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the relation for a line being contained in a plane
variable (line_in_plane : Line → Plane → Prop)

-- Define the relation for planes intersecting
variable (planes_intersect : Plane → Plane → Prop)

-- Define the relation for the line of intersection between two planes
variable (intersection_line : Plane → Plane → Line)

-- Define the parallel relation between lines
variable (parallel_line : Line → Line → Prop)

-- Define the skew relation between lines
variable (skew_lines : Line → Line → Prop)

theorem intersection_parallel_to_l 
  (m n l : Line) (α β : Plane) 
  (h1 : skew_lines m n)
  (h2 : perp_line_plane m α)
  (h3 : perp_line_plane n β)
  (h4 : perp_line l m)
  (h5 : perp_line l n)
  (h6 : ¬ line_in_plane l α)
  (h7 : ¬ line_in_plane l β) :
  planes_intersect α β ∧ parallel_line (intersection_line α β) l :=
sorry

end intersection_parallel_to_l_l3911_391130


namespace prism_15_edges_has_7_faces_l3911_391105

/-- A prism is a three-dimensional geometric shape with two identical ends (bases) and flat sides. -/
structure Prism where
  edges : ℕ

/-- The number of faces in a prism given its number of edges. -/
def num_faces (p : Prism) : ℕ :=
  (p.edges / 3) + 2

/-- Theorem: A prism with 15 edges has 7 faces. -/
theorem prism_15_edges_has_7_faces :
  ∀ (p : Prism), p.edges = 15 → num_faces p = 7 := by
  sorry

#check prism_15_edges_has_7_faces

end prism_15_edges_has_7_faces_l3911_391105


namespace smallest_sum_is_3257_l3911_391145

def Digits : Finset ℕ := {3, 7, 2, 9, 5}

def is_valid_pair (a b : ℕ) : Prop :=
  (a ≥ 1000 ∧ a < 10000) ∧ 
  (b ≥ 100 ∧ b < 1000) ∧
  (Finset.card (Finset.filter (λ d => d ∈ Digits) (Finset.range 10)) = 5)

def sum_of_pair (a b : ℕ) : ℕ := a + b

theorem smallest_sum_is_3257 :
  ∀ a b : ℕ, is_valid_pair a b → sum_of_pair a b ≥ 3257 :=
sorry

end smallest_sum_is_3257_l3911_391145


namespace david_chemistry_marks_l3911_391120

def marks_problem (english math physics biology : ℕ) (average : ℚ) : Prop :=
  let total_known := english + math + physics + biology
  let total_all := average * 5
  let chemistry := total_all - total_known
  chemistry = 63

theorem david_chemistry_marks :
  marks_problem 70 63 80 65 (68.2 : ℚ) := by
  sorry

end david_chemistry_marks_l3911_391120


namespace slower_pipe_filling_time_l3911_391188

/-- Proves that given two pipes with specified relative speeds and combined filling time,
    the slower pipe takes 180 minutes to fill the tank alone. -/
theorem slower_pipe_filling_time 
  (fast_pipe_speed : ℝ) 
  (slow_pipe_speed : ℝ) 
  (combined_time : ℝ) 
  (h1 : fast_pipe_speed = 4 * slow_pipe_speed) 
  (h2 : combined_time = 36) 
  (h3 : (fast_pipe_speed + slow_pipe_speed) * combined_time = 1) : 
  1 / slow_pipe_speed = 180 := by
  sorry

end slower_pipe_filling_time_l3911_391188


namespace min_speed_for_race_l3911_391198

/-- Proves that the minimum speed required to travel 5 kilometers in 20 minutes is 15 km/h -/
theorem min_speed_for_race (distance : ℝ) (time_minutes : ℝ) (speed : ℝ) : 
  distance = 5 → 
  time_minutes = 20 → 
  speed = distance / (time_minutes / 60) → 
  speed = 15 := by sorry

end min_speed_for_race_l3911_391198


namespace convex_polygon_properties_l3911_391122

/-- A convex n-gon -/
structure ConvexPolygon (n : ℕ) where
  -- Add necessary fields here
  n_ge_3 : n ≥ 3

/-- The sum of interior angles of a convex n-gon -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- The number of triangles formed by non-intersecting diagonals in a convex n-gon -/
def num_triangles (n : ℕ) : ℕ := n - 2

theorem convex_polygon_properties {n : ℕ} (p : ConvexPolygon n) :
  (sum_interior_angles n = (n - 2) * 180) ∧
  (num_triangles n = n - 2) :=
by sorry

end convex_polygon_properties_l3911_391122


namespace percentage_square_root_l3911_391153

theorem percentage_square_root (x : ℝ) : 
  Real.sqrt (x / 100) = 20 → x = 20 := by sorry

end percentage_square_root_l3911_391153


namespace x_range_l3911_391106

/-- An acute triangle with side lengths 2, 3, and x, where x is a positive real number -/
structure AcuteTriangle where
  x : ℝ
  x_pos : x > 0
  acute : x^2 < 2^2 + 3^2  -- Condition for the triangle to be acute

/-- The range of x in an acute triangle with side lengths 2, 3, and x -/
theorem x_range (t : AcuteTriangle) : Real.sqrt 5 < t.x ∧ t.x < Real.sqrt 13 := by
  sorry

end x_range_l3911_391106


namespace cosine_equality_l3911_391165

theorem cosine_equality (x y : ℝ) : 
  x = 2 * Real.cos (2 * Real.pi / 5) →
  y = 2 * Real.cos (4 * Real.pi / 5) →
  x + y + 1 = 0 →
  x = (-1 + Real.sqrt 5) / 2 ∧ y = (-1 - Real.sqrt 5) / 2 := by
  sorry

end cosine_equality_l3911_391165


namespace distribute_a_over_sum_l3911_391131

theorem distribute_a_over_sum (a b c : ℝ) : a * (a + b - c) = a^2 + a*b - a*c := by sorry

end distribute_a_over_sum_l3911_391131


namespace no_x_satisfies_arccos_lt_arcsin_l3911_391160

theorem no_x_satisfies_arccos_lt_arcsin : ¬∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ Real.arccos x < Real.arcsin x := by
  sorry

end no_x_satisfies_arccos_lt_arcsin_l3911_391160


namespace boxes_per_carton_l3911_391132

/-- Proves that the number of boxes in each carton is 1 -/
theorem boxes_per_carton (c : ℕ) : c > 0 → ∃ b : ℕ, b > 0 ∧ b * c = 1 := by
  sorry

#check boxes_per_carton

end boxes_per_carton_l3911_391132


namespace prime_pairs_dividing_powers_of_five_plus_one_l3911_391121

theorem prime_pairs_dividing_powers_of_five_plus_one :
  ∀ p q : ℕ, 
    Nat.Prime p → Nat.Prime q → 
    p ∣ (5^q + 1) → q ∣ (5^p + 1) → 
    ((p = 2 ∧ q = 2) ∨ 
     (p = 2 ∧ q = 13) ∨ 
     (p = 3 ∧ q = 3) ∨ 
     (p = 3 ∧ q = 7) ∨ 
     (p = 13 ∧ q = 2) ∨ 
     (p = 7 ∧ q = 3)) :=
by sorry

end prime_pairs_dividing_powers_of_five_plus_one_l3911_391121


namespace sum_proper_divisors_81_l3911_391146

def proper_divisors (n : ℕ) : Finset ℕ :=
  (Finset.range n).filter (λ d => d ∣ n)

theorem sum_proper_divisors_81 :
  (proper_divisors 81).sum id = 40 :=
sorry

end sum_proper_divisors_81_l3911_391146


namespace parallelogram_area_l3911_391127

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the incenter
def incenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the conditions of the triangle
def triangle_conditions (t : Triangle) : Prop :=
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d t.A t.C = 6 ∧ 
  d t.B t.C = 7 ∧
  ((t.B.1 - t.A.1) * (t.C.1 - t.A.1) + (t.B.2 - t.A.2) * (t.C.2 - t.A.2)) / 
    (d t.A t.B * d t.A t.C) = 1/5

-- Theorem statement
theorem parallelogram_area (t : Triangle) (h : triangle_conditions t) :
  let O := incenter t
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d t.A t.B * d O t.A * Real.sin (Real.arccos (((t.B.1 - O.1) * (t.A.1 - O.1) + 
    (t.B.2 - O.2) * (t.A.2 - O.2)) / (d O t.B * d O t.A))) = 10 * Real.sqrt 6 / 3 :=
sorry

end parallelogram_area_l3911_391127


namespace tenth_root_of_unity_l3911_391169

theorem tenth_root_of_unity : 
  ∃ n : ℕ, 0 ≤ n ∧ n < 10 ∧ 
  (Complex.tan (π / 6) + Complex.I) / (Complex.tan (π / 6) - Complex.I) = 
  Complex.exp (2 * π * I * (n : ℂ) / 10) := by sorry

end tenth_root_of_unity_l3911_391169


namespace unique_solution_system_l3911_391103

theorem unique_solution_system (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : Real.cos (π * x)^2 + 2 * Real.sin (π * y) = 1)
  (h2 : Real.sin (π * x) + Real.sin (π * y) = 0)
  (h3 : x^2 - y^2 = 12) :
  x = 4 ∧ y = 2 := by
sorry

end unique_solution_system_l3911_391103


namespace toll_booth_proof_l3911_391109

/-- Represents the number of vehicles passing through a toll booth on each day of the week -/
structure VehicleCount where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ
  sunday : ℕ

/-- Represents the toll rates for different vehicle types -/
structure TollRates where
  car : ℕ
  bus : ℕ
  truck : ℕ

def total_vehicles : ℕ := 450

def toll_booth_theorem (vc : VehicleCount) (tr : TollRates) : Prop :=
  vc.monday = 50 ∧
  vc.tuesday = vc.monday + (vc.monday / 10) ∧
  vc.wednesday = 2 * (vc.monday + vc.tuesday) ∧
  vc.thursday = vc.wednesday / 2 ∧
  vc.friday = vc.saturday ∧ vc.saturday = vc.sunday ∧
  vc.monday + vc.tuesday + vc.wednesday + vc.thursday + vc.friday + vc.saturday + vc.sunday = total_vehicles ∧
  tr.car = 2 ∧ tr.bus = 5 ∧ tr.truck = 10 →
  (vc.tuesday + vc.wednesday + vc.thursday = 370) ∧
  (vc.friday = 10) ∧
  ((total_vehicles / 3) * (tr.car + tr.bus + tr.truck) = 2550)

theorem toll_booth_proof (vc : VehicleCount) (tr : TollRates) : 
  toll_booth_theorem vc tr := by sorry

end toll_booth_proof_l3911_391109


namespace digit_sum_double_digit_sum_half_l3911_391138

/-- Represents a natural number as a list of its digits in base 10 -/
def digits (n : ℕ) : List ℕ := sorry

/-- Calculates the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := (digits n).sum

/-- Checks if two natural numbers are digit rearrangements of each other -/
def is_digit_rearrangement (a b : ℕ) : Prop := sorry

theorem digit_sum_double (a b : ℕ) (h : is_digit_rearrangement a b) : 
  sum_of_digits (2 * a) = sum_of_digits (2 * b) := by sorry

theorem digit_sum_half (a b : ℕ) (h1 : is_digit_rearrangement a b) 
  (h2 : Even a) (h3 : Even b) : 
  sum_of_digits (a / 2) = sum_of_digits (b / 2) := by sorry

end digit_sum_double_digit_sum_half_l3911_391138


namespace waiter_tables_l3911_391114

theorem waiter_tables (total_customers : ℕ) (left_customers : ℕ) (people_per_table : ℕ)
  (h1 : total_customers = 21)
  (h2 : left_customers = 12)
  (h3 : people_per_table = 3) :
  (total_customers - left_customers) / people_per_table = 3 :=
by sorry

end waiter_tables_l3911_391114


namespace book_selection_ways_l3911_391166

def number_of_books : ℕ := 3

def ways_to_choose (n : ℕ) : ℕ := 2^n - 1

theorem book_selection_ways :
  ways_to_choose number_of_books = 7 := by
  sorry

end book_selection_ways_l3911_391166


namespace smallest_angle_in_triangle_l3911_391152

theorem smallest_angle_in_triangle (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- angles are positive
  b = (5/4) * a →          -- ratio of second to first angle is 5:4
  c = (9/4) * a →          -- ratio of third to first angle is 9:4
  a + b + c = 180 →        -- sum of angles in a triangle is 180°
  a = 40 :=                -- smallest angle is 40°
by sorry

end smallest_angle_in_triangle_l3911_391152


namespace box_height_proof_l3911_391197

theorem box_height_proof (h w l : ℝ) (volume : ℝ) : 
  l = 3 * h →
  l = 4 * w →
  volume = l * w * h →
  volume = 3888 →
  h = 12 := by
sorry

end box_height_proof_l3911_391197


namespace root_sum_gt_one_l3911_391171

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x * Real.log x) / (x - 1) - a

noncomputable def h (a : ℝ) (x : ℝ) : ℝ := (x^2 - x) * f a x

theorem root_sum_gt_one (a m : ℝ) (x₁ x₂ : ℝ) :
  a < 0 →
  x₁ ≠ x₂ →
  h a x₁ = m →
  h a x₂ = m →
  x₁ + x₂ > 1 := by
sorry

end root_sum_gt_one_l3911_391171


namespace different_color_probability_l3911_391157

def red : ℕ := 4
def green : ℕ := 5
def white : ℕ := 12
def blue : ℕ := 3

def total : ℕ := red + green + white + blue

theorem different_color_probability :
  let prob_diff_color := (red * green + red * white + red * blue +
                          green * white + green * blue + white * blue) /
                         (total * (total - 1))
  prob_diff_color = 191 / 552 := by sorry

end different_color_probability_l3911_391157


namespace product_of_squares_is_one_l3911_391133

theorem product_of_squares_is_one 
  (x y z k : ℝ) 
  (distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z) 
  (eq1 : x + 1/y = k) 
  (eq2 : y + 1/z = k) 
  (eq3 : z + 1/x = k) : 
  x^2 * y^2 * z^2 = 1 := by
sorry

end product_of_squares_is_one_l3911_391133


namespace find_b_value_l3911_391168

theorem find_b_value (a b c : ℝ) 
  (sum_eq : a + b + c = 120)
  (equal_after_changes : a + 5 = b - 5 ∧ b - 5 = c^2) : 
  b = 61.25 := by
  sorry

end find_b_value_l3911_391168


namespace sin_45_degrees_l3911_391135

theorem sin_45_degrees : Real.sin (π / 4) = Real.sqrt 2 / 2 := by sorry

end sin_45_degrees_l3911_391135


namespace oarsmen_count_l3911_391112

/-- The number of oarsmen in the boat -/
def n : ℕ := sorry

/-- The total weight of the oarsmen before replacement -/
def W : ℝ := sorry

/-- The average weight increase after replacement -/
def weight_increase : ℝ := 2

/-- The weight of the replaced crew member -/
def old_weight : ℝ := 40

/-- The weight of the new crew member -/
def new_weight : ℝ := 80

/-- Theorem stating that the number of oarsmen is 20 -/
theorem oarsmen_count : n = 20 := by
  have h1 : (W + new_weight - old_weight) / n = W / n + weight_increase := by sorry
  sorry

end oarsmen_count_l3911_391112


namespace house_wall_nails_l3911_391159

/-- The number of nails needed per plank -/
def nails_per_plank : ℕ := 2

/-- The number of planks used for the house wall -/
def planks_used : ℕ := 16

/-- The total number of nails needed for the house wall -/
def total_nails : ℕ := nails_per_plank * planks_used

theorem house_wall_nails : total_nails = 32 := by
  sorry

end house_wall_nails_l3911_391159


namespace pies_sold_per_day_l3911_391104

theorem pies_sold_per_day (total_pies : ℕ) (days_in_week : ℕ) 
  (h1 : total_pies = 56) 
  (h2 : days_in_week = 7) : 
  total_pies / days_in_week = 8 := by
  sorry

end pies_sold_per_day_l3911_391104


namespace max_y_over_x_l3911_391142

theorem max_y_over_x (x y : ℝ) (h : (x - 2)^2 + y^2 = 3) : 
  ∃ (max : ℝ), (∀ (x' y' : ℝ), (x' - 2)^2 + y'^2 = 3 → y' / x' ≤ max) ∧ max = Real.sqrt 3 := by
sorry

end max_y_over_x_l3911_391142


namespace sixth_term_of_geometric_sequence_l3911_391192

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem sixth_term_of_geometric_sequence 
  (a : ℕ → ℝ) 
  (h_geometric : geometric_sequence a) 
  (h_fourth : a 4 = 512) 
  (h_ninth : a 9 = 8) : 
  a 6 = 128 := by
sorry

end sixth_term_of_geometric_sequence_l3911_391192


namespace distance_after_two_hours_l3911_391148

/-- The distance between two people walking in opposite directions after a given time -/
def distanceApart (jaySpeed : Real) (paulSpeed : Real) (time : Real) : Real :=
  (jaySpeed + paulSpeed) * time

theorem distance_after_two_hours :
  let jaySpeed : Real := 1 / 20 -- miles per minute
  let paulSpeed : Real := 3 / 40 -- miles per minute
  let time : Real := 2 * 60 -- 2 hours in minutes
  distanceApart jaySpeed paulSpeed time = 15 := by
  sorry

end distance_after_two_hours_l3911_391148


namespace sin_sixty_degrees_l3911_391183

theorem sin_sixty_degrees : Real.sin (π / 3) = Real.sqrt 3 / 2 := by
  sorry

end sin_sixty_degrees_l3911_391183


namespace rice_profit_calculation_l3911_391154

/-- Calculates the profit from selling a sack of rice -/
theorem rice_profit_calculation (weight : ℝ) (cost : ℝ) (price_per_kg : ℝ) : 
  weight = 50 ∧ cost = 50 ∧ price_per_kg = 1.2 → price_per_kg * weight - cost = 10 := by
  sorry

end rice_profit_calculation_l3911_391154


namespace square_side_length_average_l3911_391161

theorem square_side_length_average (a b c : Real) (ha : a = 25) (hb : b = 64) (hc : c = 121) :
  (Real.sqrt a + Real.sqrt b + Real.sqrt c) / 3 = 8 := by
  sorry

end square_side_length_average_l3911_391161


namespace tetrahedron_reciprocal_squares_equality_l3911_391180

/-- A tetrahedron with heights and distances between opposite edges. -/
structure Tetrahedron where
  h₁ : ℝ
  h₂ : ℝ
  h₃ : ℝ
  h₄ : ℝ
  d₁ : ℝ
  d₂ : ℝ
  d₃ : ℝ
  h₁_pos : 0 < h₁
  h₂_pos : 0 < h₂
  h₃_pos : 0 < h₃
  h₄_pos : 0 < h₄
  d₁_pos : 0 < d₁
  d₂_pos : 0 < d₂
  d₃_pos : 0 < d₃

/-- The sum of reciprocal squares of heights equals the sum of reciprocal squares of distances. -/
theorem tetrahedron_reciprocal_squares_equality (t : Tetrahedron) :
    1 / t.h₁^2 + 1 / t.h₂^2 + 1 / t.h₃^2 + 1 / t.h₄^2 = 1 / t.d₁^2 + 1 / t.d₂^2 + 1 / t.d₃^2 := by
  sorry

end tetrahedron_reciprocal_squares_equality_l3911_391180


namespace red_candies_count_l3911_391178

/-- Represents the number of candies of each color -/
structure CandyCounts where
  red : ℕ
  yellow : ℕ
  blue : ℕ

/-- The conditions of the candy problem -/
def candy_conditions (c : CandyCounts) : Prop :=
  c.yellow = 3 * c.red - 20 ∧
  c.blue = c.yellow / 2 ∧
  c.red + c.blue = 90

/-- The theorem stating that there are 40 red candies -/
theorem red_candies_count :
  ∃ c : CandyCounts, candy_conditions c ∧ c.red = 40 := by
  sorry

end red_candies_count_l3911_391178


namespace free_throws_count_l3911_391126

/-- Represents a basketball team's scoring -/
structure BasketballScore where
  two_pointers : ℕ
  three_pointers : ℕ
  free_throws : ℕ

/-- Calculates the total score -/
def total_score (s : BasketballScore) : ℕ :=
  2 * s.two_pointers + 3 * s.three_pointers + s.free_throws

/-- Theorem: Given the conditions, the number of free throws is 13 -/
theorem free_throws_count (s : BasketballScore) :
  (2 * s.two_pointers = 3 * s.three_pointers) →
  (s.free_throws = s.two_pointers + 1) →
  (total_score s = 61) →
  s.free_throws = 13 := by
  sorry

#check free_throws_count

end free_throws_count_l3911_391126


namespace symmetrical_circles_sin_cos_theta_l3911_391195

/-- Given two circles C₁ and C₂ defined by their equations and a line of symmetry,
    prove that sin θ cos θ = -2/5 --/
theorem symmetrical_circles_sin_cos_theta (a θ : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + a*x = 0 → 2*x - y - 1 = 0) →  -- C₁ is symmetrical about the line
  (∀ x y : ℝ, x^2 + y^2 + 2*a*x + y*(Real.tan θ) = 0 → 2*x - y - 1 = 0) →  -- C₂ is symmetrical about the line
  Real.sin θ * Real.cos θ = -2/5 := by
  sorry

end symmetrical_circles_sin_cos_theta_l3911_391195


namespace billy_ate_nine_apples_wednesday_l3911_391179

/-- The number of apples Billy ate each day of the week -/
structure AppleConsumption where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- The conditions of Billy's apple consumption throughout the week -/
def billy_apple_conditions (ac : AppleConsumption) : Prop :=
  ac.monday = 2 ∧
  ac.tuesday = 2 * ac.monday ∧
  ac.thursday = 4 * ac.friday ∧
  ac.friday = ac.monday / 2 ∧
  ac.monday + ac.tuesday + ac.wednesday + ac.thursday + ac.friday = 20

/-- Theorem stating that given the conditions, Billy ate 9 apples on Wednesday -/
theorem billy_ate_nine_apples_wednesday (ac : AppleConsumption) 
  (h : billy_apple_conditions ac) : ac.wednesday = 9 := by
  sorry


end billy_ate_nine_apples_wednesday_l3911_391179


namespace rate_of_profit_l3911_391155

theorem rate_of_profit (cost_price selling_price : ℝ) : 
  cost_price = 50 → selling_price = 100 → 
  (selling_price - cost_price) / cost_price * 100 = 100 := by
  sorry

end rate_of_profit_l3911_391155


namespace brand_A_soap_users_l3911_391182

theorem brand_A_soap_users (total : ℕ) (neither : ℕ) (both : ℕ) (ratio : ℕ) : 
  total = 300 →
  neither = 80 →
  both = 40 →
  ratio = 3 →
  total - neither - (ratio * both) - both = 60 :=
by sorry

end brand_A_soap_users_l3911_391182


namespace max_a_for_increasing_cubic_l3911_391113

/-- Given that f(x) = x^3 - ax is increasing on [1, +∞), 
    the maximum value of the real number a is 3. -/
theorem max_a_for_increasing_cubic (f : ℝ → ℝ) (a : ℝ) : 
  (∀ x ≥ 1, ∀ y ≥ 1, x < y → f x < f y) →
  (∀ x : ℝ, f x = x^3 - a*x) →
  a ≤ 3 ∧ ∀ b > 3, ∃ x ≥ 1, ∃ y > x, f x ≥ f y := by
  sorry

end max_a_for_increasing_cubic_l3911_391113


namespace factorial_sum_perfect_power_l3911_391102

def factorial_sum (n : ℕ+) : ℕ := (Finset.range n).sum (λ i => Nat.factorial (i + 1))

def is_perfect_power (m : ℕ) : Prop := ∃ (a b : ℕ), b > 1 ∧ a^b = m

theorem factorial_sum_perfect_power (n : ℕ+) :
  is_perfect_power (factorial_sum n) ↔ n = 1 ∨ n = 3 :=
sorry

end factorial_sum_perfect_power_l3911_391102


namespace square_sum_given_sum_and_product_l3911_391117

theorem square_sum_given_sum_and_product (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 2) :
  x^2 + y^2 = 21 := by
sorry

end square_sum_given_sum_and_product_l3911_391117


namespace complex_equation_solution_l3911_391143

theorem complex_equation_solution (z : ℂ) : z * (1 - Complex.I) = 3 + Complex.I → z = 1 + 2 * Complex.I := by
  sorry

end complex_equation_solution_l3911_391143


namespace contrapositive_equivalence_l3911_391119

-- Define the type for points
def Point : Type := ℝ × ℝ

-- Define the type for hyperbolas
def Hyperbola : Type := Point → Prop

-- Define what it means for a point to satisfy the equation of a hyperbola
def SatisfiesEquation (M : Point) (C : Hyperbola) : Prop := C M

-- Define what it means for a point to be on the graph of a hyperbola
def OnGraph (M : Point) (C : Hyperbola) : Prop := C M

-- State the theorem
theorem contrapositive_equivalence (C : Hyperbola) :
  (∀ M : Point, SatisfiesEquation M C → OnGraph M C) ↔
  (∀ M : Point, ¬OnGraph M C → ¬SatisfiesEquation M C) :=
sorry

end contrapositive_equivalence_l3911_391119


namespace expand_product_l3911_391196

theorem expand_product (x : ℝ) : (x + 3) * (x^2 + 4*x + 6) = x^3 + 7*x^2 + 18*x + 18 := by
  sorry

end expand_product_l3911_391196


namespace valid_call_start_l3911_391186

/-- Represents a time in 24-hour format -/
structure Time where
  hour : Nat
  minute : Nat
  h_valid : hour < 24
  m_valid : minute < 60

/-- The time difference between Beijing and Moscow in hours -/
def time_difference : Nat := 5

/-- Checks if a given time is within the allowed call window -/
def is_valid_call_time (t : Time) : Prop :=
  9 ≤ t.hour ∧ t.hour < 17

/-- Converts Beijing time to Moscow time -/
def beijing_to_moscow (t : Time) : Time :=
  let new_hour := (t.hour - time_difference + 24) % 24
  { hour := new_hour, minute := t.minute, 
    h_valid := by sorry,
    m_valid := t.m_valid }

/-- The proposed call start time in Beijing -/
def call_start_beijing : Time :=
  { hour := 15, minute := 0, h_valid := by sorry, m_valid := by sorry }

/-- Theorem stating that the proposed call start time is valid -/
theorem valid_call_start : 
  is_valid_call_time call_start_beijing ∧ 
  is_valid_call_time (beijing_to_moscow call_start_beijing) :=
by sorry


end valid_call_start_l3911_391186


namespace complex_expression_equals_negative_three_fourths_l3911_391139

theorem complex_expression_equals_negative_three_fourths :
  |Real.sqrt 3 - 3| - Real.tan (60 * π / 180) ^ 2 + 2 ^ (-2 : ℤ) + 2 / (Real.sqrt 3 + 1) = -3/4 := by
  sorry

end complex_expression_equals_negative_three_fourths_l3911_391139


namespace right_triangle_sets_l3911_391156

/-- Checks if three side lengths can form a right triangle -/
def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a)

theorem right_triangle_sets :
  ¬(is_right_triangle 5 7 10) ∧
  (is_right_triangle 3 4 5) ∧
  ¬(is_right_triangle 1 3 2) ∧
  (is_right_triangle 7 24 25) :=
by sorry

end right_triangle_sets_l3911_391156


namespace dividend_calculation_l3911_391190

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 36)
  (h2 : quotient = 19)
  (h3 : remainder = 5) :
  divisor * quotient + remainder = 689 :=
by sorry

end dividend_calculation_l3911_391190


namespace savings_calculation_l3911_391164

/-- Given an income and an income-to-expenditure ratio, calculate the savings -/
def calculate_savings (income : ℚ) (income_ratio : ℚ) (expenditure_ratio : ℚ) : ℚ :=
  income - (income * expenditure_ratio / income_ratio)

/-- Theorem: For an income of 21000 and an income-to-expenditure ratio of 7:6, the savings is 3000 -/
theorem savings_calculation :
  calculate_savings 21000 7 6 = 3000 := by
  sorry

end savings_calculation_l3911_391164


namespace horse_pig_compensation_difference_l3911_391111

-- Define the consumption of each animal type relative to a pig
def pig_consumption : ℚ := 1
def sheep_consumption : ℚ := 2 * pig_consumption
def horse_consumption : ℚ := 2 * sheep_consumption
def cow_consumption : ℚ := 2 * horse_consumption

-- Define the total compensation
def total_compensation : ℚ := 9

-- Theorem statement
theorem horse_pig_compensation_difference :
  let total_consumption := pig_consumption + sheep_consumption + horse_consumption + cow_consumption
  let unit_compensation := total_compensation / total_consumption
  let horse_compensation := horse_consumption * unit_compensation
  let pig_compensation := pig_consumption * unit_compensation
  horse_compensation - pig_compensation = 9/5 := by
  sorry

end horse_pig_compensation_difference_l3911_391111


namespace circle_constant_properties_l3911_391134

noncomputable def π : ℝ := Real.pi

theorem circle_constant_properties (a b c d : ℚ) : 
  -- Original proposition
  ((a * π + b = c * π + d) → (a = c ∧ b = d)) ∧
  -- Negation is false
  ¬((a * π + b = c * π + d) → (a ≠ c ∨ b ≠ d)) ∧
  -- Converse is true
  ((a = c ∧ b = d) → (a * π + b = c * π + d)) ∧
  -- Inverse is true
  ((a * π + b ≠ c * π + d) → (a ≠ c ∨ b ≠ d)) ∧
  -- Contrapositive is true
  ((a ≠ c ∨ b ≠ d) → (a * π + b ≠ c * π + d)) :=
by sorry


end circle_constant_properties_l3911_391134


namespace unique_solution_values_non_monotonic_range_l3911_391158

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 + (a + 2) * x + b

-- Part 1
theorem unique_solution_values (a b : ℝ) :
  f a b (-1) = -2 →
  (∃! x, f a b x = 2 * x) →
  a = 2 ∧ b = 1 := by sorry

-- Part 2
theorem non_monotonic_range (a b : ℝ) :
  (∃ x y, x ∈ Set.Icc (-2 : ℝ) 2 ∧ 
          y ∈ Set.Icc (-2 : ℝ) 2 ∧ 
          x < y ∧ 
          f a b x > f a b y) →
  -6 < a ∧ a < 2 := by sorry

end unique_solution_values_non_monotonic_range_l3911_391158


namespace divide_by_three_l3911_391101

theorem divide_by_three (x : ℚ) (h : x / 4 = 12) : x / 3 = 16 := by
  sorry

end divide_by_three_l3911_391101


namespace difference_of_squares_l3911_391172

theorem difference_of_squares (m n : ℝ) : m^2 - n^2 = (m + n) * (m - n) := by sorry

end difference_of_squares_l3911_391172


namespace selenas_book_pages_selenas_book_pages_proof_l3911_391173

theorem selenas_book_pages : ℕ → Prop :=
  fun s => (s / 2 - 20 = 180) → s = 400

-- The proof
theorem selenas_book_pages_proof : selenas_book_pages 400 := by
  sorry

end selenas_book_pages_selenas_book_pages_proof_l3911_391173


namespace cubic_root_equation_solution_l3911_391110

theorem cubic_root_equation_solution (x : ℝ) :
  (∃ y : ℝ, x^(1/3) + y^(1/3) = 3 ∧ x + y = 26) →
  (∃ p q : ℤ, x = p - Real.sqrt q ∧ p + q = 26) :=
by sorry

end cubic_root_equation_solution_l3911_391110


namespace investment_problem_l3911_391129

/-- Proves that given the conditions of the investment problem, the initial investment in the 2.5% account was $290 -/
theorem investment_problem (total_investment : ℝ) (interest_rate1 : ℝ) (interest_rate2 : ℝ) 
  (final_amount : ℝ) (investment1 : ℝ) :
  total_investment = 1500 →
  interest_rate1 = 0.025 →
  interest_rate2 = 0.045 →
  final_amount = 1650 →
  investment1 * (1 + interest_rate1)^2 + (total_investment - investment1) * (1 + interest_rate2)^2 = final_amount →
  investment1 = 290 :=
by sorry

end investment_problem_l3911_391129


namespace lexie_family_age_difference_l3911_391124

/-- Given Lexie's age, calculate the age difference between her brother and sister -/
def age_difference (lexie_age : ℕ) : ℕ :=
  let brother_age := lexie_age - 6
  let sister_age := lexie_age * 2
  sister_age - brother_age

/-- Theorem stating the age difference between Lexie's brother and sister -/
theorem lexie_family_age_difference :
  age_difference 8 = 14 := by
  sorry

end lexie_family_age_difference_l3911_391124


namespace total_inflation_time_inflation_time_proof_l3911_391191

/-- Calculates the total time taken to inflate soccer balls -/
theorem total_inflation_time (alexia_time ermias_time leila_time : ℕ)
  (alexia_balls : ℕ) (ermias_extra leila_fewer : ℕ) : ℕ :=
  let alexia_total := alexia_time * alexia_balls
  let ermias_balls := alexia_balls + ermias_extra
  let ermias_total := ermias_time * ermias_balls
  let leila_balls := ermias_balls - leila_fewer
  let leila_total := leila_time * leila_balls
  alexia_total + ermias_total + leila_total

/-- Proves that the total time taken to inflate all soccer balls is 4160 minutes -/
theorem inflation_time_proof :
  total_inflation_time 18 25 30 50 12 5 = 4160 := by
  sorry


end total_inflation_time_inflation_time_proof_l3911_391191


namespace tom_calorie_consumption_l3911_391176

/-- Calculates the total calories consumed by Tom given the weight and calorie content of carrots and broccoli. -/
def total_calories (carrot_weight : ℝ) (broccoli_weight : ℝ) (carrot_calories : ℝ) (broccoli_calories : ℝ) : ℝ :=
  carrot_weight * carrot_calories + broccoli_weight * broccoli_calories

/-- Theorem stating that Tom's total calorie consumption is 85 given the problem conditions. -/
theorem tom_calorie_consumption :
  let carrot_weight : ℝ := 1
  let broccoli_weight : ℝ := 2 * carrot_weight
  let carrot_calories : ℝ := 51
  let broccoli_calories : ℝ := (1/3) * carrot_calories
  total_calories carrot_weight broccoli_weight carrot_calories broccoli_calories = 85 := by
  sorry

end tom_calorie_consumption_l3911_391176


namespace exists_negative_f_iff_a_less_than_three_halves_max_value_one_implies_a_values_l3911_391147

-- Define the function f(x)
def f (a x : ℝ) : ℝ := x^2 + (2*a - 1)*x - 3

-- Part 1
theorem exists_negative_f_iff_a_less_than_three_halves (a : ℝ) :
  (∃ x : ℝ, x > 1 ∧ f a x < 0) ↔ a < 3/2 :=
sorry

-- Part 2
theorem max_value_one_implies_a_values (a : ℝ) :
  (∀ x ∈ Set.Icc (-1) 3, f a x ≤ 1) ∧ (∃ x ∈ Set.Icc (-1) 3, f a x = 1) →
  a = -1/3 ∨ a = -1 :=
sorry

end exists_negative_f_iff_a_less_than_three_halves_max_value_one_implies_a_values_l3911_391147


namespace ice_floe_mass_l3911_391175

/-- The mass of the ice floe given the polar bear's mass and trajectory diameters -/
theorem ice_floe_mass (m : ℝ) (D d : ℝ) (hm : m = 600) (hD : D = 10) (hd : d = 9.5) :
  (m * d) / (D - d) = 11400 := by
  sorry

end ice_floe_mass_l3911_391175


namespace football_sample_size_l3911_391163

/-- Calculates the number of people to be sampled from a group in stratified sampling -/
def stratified_sample_size (total_population : ℕ) (group_size : ℕ) (total_sample : ℕ) : ℕ :=
  (group_size * total_sample) / total_population

/-- Proves that the stratified sample size for the football group is 8 -/
theorem football_sample_size :
  let total_population : ℕ := 120
  let football_size : ℕ := 40
  let basketball_size : ℕ := 60
  let volleyball_size : ℕ := 20
  let total_sample : ℕ := 24
  stratified_sample_size total_population football_size total_sample = 8 := by
  sorry

#eval stratified_sample_size 120 40 24

end football_sample_size_l3911_391163


namespace sons_age_l3911_391199

theorem sons_age (son_age father_age : ℕ) : 
  father_age = son_age + 20 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 18 := by
sorry

end sons_age_l3911_391199


namespace ceiling_negative_fraction_cube_l3911_391174

theorem ceiling_negative_fraction_cube : ⌈(-7/4)^3⌉ = -5 := by
  sorry

end ceiling_negative_fraction_cube_l3911_391174


namespace cal_anthony_ratio_l3911_391136

/-- Represents the number of transactions handled by each person --/
structure Transactions where
  mabel : ℕ
  anthony : ℕ
  cal : ℕ
  jade : ℕ

/-- The given conditions of the problem --/
def problem_conditions (t : Transactions) : Prop :=
  t.mabel = 90 ∧
  t.anthony = t.mabel + t.mabel / 10 ∧
  t.jade = 80 ∧
  t.jade = t.cal + 14

/-- The theorem to be proved --/
theorem cal_anthony_ratio (t : Transactions) 
  (h : problem_conditions t) : 
  (t.cal : ℚ) / t.anthony = 2 / 3 := by
  sorry


end cal_anthony_ratio_l3911_391136
