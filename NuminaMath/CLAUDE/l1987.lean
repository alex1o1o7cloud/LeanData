import Mathlib

namespace inequality_proof_l1987_198746

theorem inequality_proof (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1/9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
  sorry

end inequality_proof_l1987_198746


namespace hippopotamus_crayons_l1987_198743

theorem hippopotamus_crayons (initial_crayons final_crayons : ℕ) 
  (h1 : initial_crayons = 87) 
  (h2 : final_crayons = 80) : 
  initial_crayons - final_crayons = 7 := by
  sorry

end hippopotamus_crayons_l1987_198743


namespace problem_solution_l1987_198706

/-- The number of ways to distribute n distinct objects to k recipients -/
def distribute (n k : ℕ) : ℕ := k^n

/-- The number of ways to distribute n distinct objects to k recipients,
    where 2 specific objects must be given to the same recipient -/
def distributeWithPair (n k : ℕ) : ℕ := k * (k^(n - 2))

theorem problem_solution :
  distributeWithPair 8 10 = 10^7 := by sorry

end problem_solution_l1987_198706


namespace polygon_arrangement_sides_l1987_198735

/-- Represents a regular polygon with a given number of sides. -/
structure RegularPolygon where
  sides : ℕ
  sides_positive : sides > 0

/-- Represents the arrangement of polygons as described in the problem. -/
structure PolygonArrangement where
  pentagon : RegularPolygon
  triangle : RegularPolygon
  heptagon : RegularPolygon
  nonagon : RegularPolygon
  dodecagon : RegularPolygon
  pentagon_sides : pentagon.sides = 5
  triangle_sides : triangle.sides = 3
  heptagon_sides : heptagon.sides = 7
  nonagon_sides : nonagon.sides = 9
  dodecagon_sides : dodecagon.sides = 12

/-- The number of exposed sides in the polygon arrangement. -/
def exposed_sides (arrangement : PolygonArrangement) : ℕ :=
  arrangement.pentagon.sides + arrangement.triangle.sides + arrangement.heptagon.sides +
  arrangement.nonagon.sides + arrangement.dodecagon.sides - 7

theorem polygon_arrangement_sides (arrangement : PolygonArrangement) :
  exposed_sides arrangement = 28 := by
  sorry

end polygon_arrangement_sides_l1987_198735


namespace absolute_value_equation_solution_l1987_198707

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 3| = |x + 1| :=
by
  sorry

end absolute_value_equation_solution_l1987_198707


namespace smallest_multiple_of_6_and_15_l1987_198722

theorem smallest_multiple_of_6_and_15 :
  ∃ b : ℕ, b > 0 ∧ 6 ∣ b ∧ 15 ∣ b ∧ ∀ c : ℕ, c > 0 → 6 ∣ c → 15 ∣ c → b ≤ c :=
by
  sorry

end smallest_multiple_of_6_and_15_l1987_198722


namespace jeff_sunday_morning_laps_l1987_198725

/-- The number of laps Jeff swam on Sunday morning before the break -/
def sunday_morning_laps (total_laps required_laps saturday_laps remaining_laps : ℕ) : ℕ :=
  total_laps - saturday_laps - remaining_laps

theorem jeff_sunday_morning_laps :
  sunday_morning_laps 98 27 56 = 15 := by
  sorry

end jeff_sunday_morning_laps_l1987_198725


namespace topology_check_l1987_198775

-- Define the set X
def X : Set Char := {'{', 'a', 'b', 'c', '}'}

-- Define the four sets v
def v1 : Set (Set Char) := {∅, {'a'}, {'c'}, {'a', 'b', 'c'}}
def v2 : Set (Set Char) := {∅, {'b'}, {'c'}, {'b', 'c'}, {'a', 'b', 'c'}}
def v3 : Set (Set Char) := {∅, {'a'}, {'a', 'b'}, {'a', 'c'}}
def v4 : Set (Set Char) := {∅, {'a', 'c'}, {'b', 'c'}, {'c'}, {'a', 'b', 'c'}}

-- Define the topology property
def is_topology (v : Set (Set Char)) : Prop :=
  X ∈ v ∧ ∅ ∈ v ∧
  (∀ (S : Set (Set Char)), S ⊆ v → ⋃₀ S ∈ v) ∧
  (∀ (S : Set (Set Char)), S ⊆ v → ⋂₀ S ∈ v)

-- Theorem statement
theorem topology_check :
  is_topology v2 ∧ is_topology v4 ∧ ¬is_topology v1 ∧ ¬is_topology v3 :=
sorry

end topology_check_l1987_198775


namespace overlapped_area_of_reflected_triangle_l1987_198770

/-- Given three points O, A, and B on a coordinate plane, and the reflection of triangle OAB
    along the line y = 6 creating triangle PQR, prove that the overlapped area is 8 square units. -/
theorem overlapped_area_of_reflected_triangle (O A B : ℝ × ℝ) : 
  O = (0, 0) →
  A = (12, 2) →
  B = (0, 8) →
  let P : ℝ × ℝ := (O.1, 12 - O.2)
  let Q : ℝ × ℝ := (A.1, 12 - A.2)
  let R : ℝ × ℝ := (B.1, 12 - B.2)
  let M : ℝ × ℝ := (4, 6)
  8 = (1/2) * |M.1 - B.1| * |B.2 - R.2| := by sorry

end overlapped_area_of_reflected_triangle_l1987_198770


namespace number_increased_by_45_percent_l1987_198779

theorem number_increased_by_45_percent (x : ℝ) : x * 1.45 = 870 ↔ x = 600 := by
  sorry

end number_increased_by_45_percent_l1987_198779


namespace no_ten_consecutive_power_of_two_values_l1987_198773

theorem no_ten_consecutive_power_of_two_values (a b : ℝ) : 
  ¬ ∃ (k : ℤ → ℕ) (x₀ : ℤ), ∀ x : ℤ, x₀ ≤ x ∧ x < x₀ + 10 → 
    x^2 + a*x + b = 2^(k x) :=
sorry

end no_ten_consecutive_power_of_two_values_l1987_198773


namespace system_equivalent_to_line_l1987_198782

/-- The system of equations -/
def system (x y : ℝ) : Prop :=
  x - 2*y = 1 ∧ x^3 - 6*x*y - 8*y^3 = 1

/-- The line representing the solution -/
def solution_line (x y : ℝ) : Prop :=
  y = (x - 1) / 2

/-- Theorem stating that the system is equivalent to the solution line -/
theorem system_equivalent_to_line :
  ∀ x y : ℝ, system x y ↔ solution_line x y :=
sorry

end system_equivalent_to_line_l1987_198782


namespace painting_time_equation_l1987_198748

/-- The time it takes Doug to paint the room alone, in hours -/
def doug_time : ℝ := 5

/-- The time it takes Dave to paint the room alone, in hours -/
def dave_time : ℝ := 7

/-- The number of one-hour breaks taken -/
def breaks : ℝ := 2

/-- The total time it takes Doug and Dave to paint the room together, including breaks -/
noncomputable def total_time : ℝ := sorry

/-- Theorem stating that the equation (1/5 + 1/7)(t - 2) = 1 is satisfied by the total time -/
theorem painting_time_equation : 
  (1 / doug_time + 1 / dave_time) * (total_time - breaks) = 1 := by sorry

end painting_time_equation_l1987_198748


namespace intersection_M_N_l1987_198713

-- Define the sets M and N
def M : Set ℝ := {x | x^2 + x - 2 < 0}
def N : Set ℝ := {x | Real.log x / Real.log (1/2) > -1}

-- State the theorem
theorem intersection_M_N :
  M ∩ N = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end intersection_M_N_l1987_198713


namespace equation_solution_l1987_198711

theorem equation_solution (b c : ℝ) (θ : ℝ) :
  let x := (b^2 - c^2 * Real.sin θ^2) / (2 * b)
  x^2 + c^2 * Real.sin θ^2 = (b - x)^2 := by
  sorry

end equation_solution_l1987_198711


namespace trig_simplification_l1987_198767

theorem trig_simplification :
  (Real.sin (30 * π / 180) + Real.sin (60 * π / 180)) /
  (Real.cos (30 * π / 180) + Real.cos (60 * π / 180)) =
  Real.tan (45 * π / 180) := by sorry

end trig_simplification_l1987_198767


namespace translated_points_exponent_l1987_198768

/-- Given two points A and B, and their translations A₁ and B₁, prove that a^b = 32 -/
theorem translated_points_exponent (A B A₁ B₁ : ℝ × ℝ) (a b : ℝ) : 
  A = (-1, 3) → 
  B = (2, -3) → 
  A₁ = (a, 1) → 
  B₁ = (5, -b) → 
  A₁.1 - A.1 = 3 → 
  A.2 - A₁.2 = 2 → 
  B₁.1 - B.1 = 3 → 
  B.2 - B₁.2 = 2 → 
  a^b = 32 := by
sorry

end translated_points_exponent_l1987_198768


namespace outfit_combinations_l1987_198765

/-- The number of available shirts, pants, and hats -/
def num_items : ℕ := 7

/-- The number of available colors -/
def num_colors : ℕ := 7

/-- The total number of possible outfit combinations -/
def total_combinations : ℕ := num_items * num_items * num_items

/-- The number of outfits where all items are the same color -/
def same_color_outfits : ℕ := num_colors

/-- The number of valid outfit combinations -/
def valid_outfits : ℕ := total_combinations - same_color_outfits

theorem outfit_combinations : valid_outfits = 336 := by
  sorry

end outfit_combinations_l1987_198765


namespace appetizers_per_guest_is_six_l1987_198701

def number_of_guests : ℕ := 30

def prepared_appetizers : ℕ := 3 * 12 + 2 * 12 + 2 * 12

def additional_appetizers : ℕ := 8 * 12

def total_appetizers : ℕ := prepared_appetizers + additional_appetizers

def appetizers_per_guest : ℚ := total_appetizers / number_of_guests

theorem appetizers_per_guest_is_six :
  appetizers_per_guest = 6 := by
  sorry

end appetizers_per_guest_is_six_l1987_198701


namespace hedgehog_strawberries_l1987_198776

/-- The number of strawberries in each basket, given the conditions of the hedgehog problem -/
theorem hedgehog_strawberries (num_hedgehogs : ℕ) (num_baskets : ℕ) 
  (strawberries_per_hedgehog : ℕ) (remaining_fraction : ℚ) :
  num_hedgehogs = 2 →
  num_baskets = 3 →
  strawberries_per_hedgehog = 1050 →
  remaining_fraction = 2/9 →
  ∃ (total_strawberries : ℕ),
    total_strawberries = num_hedgehogs * strawberries_per_hedgehog / (1 - remaining_fraction) ∧
    total_strawberries / num_baskets = 900 :=
by sorry

end hedgehog_strawberries_l1987_198776


namespace spinner_final_direction_l1987_198759

-- Define the directions
inductive Direction
  | North
  | East
  | South
  | West

-- Define the rotation function
def rotate (initial : Direction) (clockwise : Rat) (counterclockwise : Rat) : Direction :=
  sorry

-- Theorem statement
theorem spinner_final_direction :
  rotate Direction.South (7/2 : Rat) (7/4 : Rat) = Direction.East :=
sorry

end spinner_final_direction_l1987_198759


namespace abs_neg_one_ninth_l1987_198799

theorem abs_neg_one_ninth : |(-1 : ℚ) / 9| = 1 / 9 := by
  sorry

end abs_neg_one_ninth_l1987_198799


namespace storm_average_rainfall_l1987_198745

theorem storm_average_rainfall 
  (duration : ℝ) 
  (first_30min : ℝ) 
  (next_30min : ℝ) 
  (last_hour : ℝ) :
  duration = 2 →
  first_30min = 5 →
  next_30min = first_30min / 2 →
  last_hour = 1 / 2 →
  (first_30min + next_30min + last_hour) / duration = 4 := by
sorry

end storm_average_rainfall_l1987_198745


namespace two_lines_in_cube_l1987_198778

/-- Represents a cube in 3D space -/
structure Cube where
  vertices : Fin 8 → ℝ × ℝ × ℝ

/-- Represents a point in 3D space -/
def Point := ℝ × ℝ × ℝ

/-- Represents a line in 3D space -/
structure Line where
  point : Point
  direction : ℝ × ℝ × ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  point : Point
  normal : ℝ × ℝ × ℝ

/-- Calculates the angle between a line and a plane -/
def angle_line_plane (l : Line) (p : Plane) : ℝ := sorry

/-- Checks if a point is on an edge of the cube -/
def point_on_edge (c : Cube) (p : Point) : Prop := sorry

/-- Counts the number of lines passing through a point and making a specific angle with two planes -/
def count_lines (c : Cube) (p : Point) (angle : ℝ) (plane1 plane2 : Plane) : ℕ := sorry

/-- The main theorem statement -/
theorem two_lines_in_cube (c : Cube) (p : Point) :
  point_on_edge c p →
  let plane_abcd := Plane.mk sorry sorry
  let plane_abc1d1 := Plane.mk sorry sorry
  count_lines c p (30 * π / 180) plane_abcd plane_abc1d1 = 2 := by sorry

end two_lines_in_cube_l1987_198778


namespace simplify_expression_l1987_198781

theorem simplify_expression (a : ℝ) : (3 * a^2)^2 = 9 * a^4 := by
  sorry

end simplify_expression_l1987_198781


namespace parabola_line_intersection_length_l1987_198798

/-- Given a parabola x² = 2py (p > 0) and a line y = 2x + p/2 that intersects
    the parabola at points A and B, prove that the length of AB is 10p. -/
theorem parabola_line_intersection_length (p : ℝ) (h : p > 0) : 
  let parabola := fun x y => x^2 = 2*p*y
  let line := fun x y => y = 2*x + p/2
  ∃ A B : ℝ × ℝ, 
    parabola A.1 A.2 ∧ 
    parabola B.1 B.2 ∧ 
    line A.1 A.2 ∧ 
    line B.1 B.2 ∧ 
    A ≠ B ∧
    ‖A - B‖ = 10*p :=
by sorry

end parabola_line_intersection_length_l1987_198798


namespace point_line_range_l1987_198755

theorem point_line_range (a : ℝ) : 
  (∀ x y : ℝ, (x = -3 ∧ y = -1) ∨ (x = 4 ∧ y = -6) → 
    (3*(-3) - 2*(-1) - a) * (3*4 - 2*(-6) - a) < 0) ↔ 
  -7 < a ∧ a < 24 :=
sorry

end point_line_range_l1987_198755


namespace decimal_56_to_binary_binary_to_decimal_56_decimal_56_binary_equivalence_l1987_198705

/-- Converts a natural number to its binary representation as a list of bits -/
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Converts a list of bits to its decimal representation -/
def from_binary (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem decimal_56_to_binary :
  to_binary 56 = [false, false, false, true, true, true] :=
by sorry

theorem binary_to_decimal_56 :
  from_binary [false, false, false, true, true, true] = 56 :=
by sorry

theorem decimal_56_binary_equivalence :
  to_binary 56 = [false, false, false, true, true, true] ∧
  from_binary [false, false, false, true, true, true] = 56 :=
by sorry

end decimal_56_to_binary_binary_to_decimal_56_decimal_56_binary_equivalence_l1987_198705


namespace community_center_ticket_sales_l1987_198737

/-- Calculates the total amount collected from ticket sales given the ticket prices and quantities sold. -/
def total_amount_collected (adult_price child_price : ℕ) (total_tickets adult_tickets : ℕ) : ℕ :=
  adult_price * adult_tickets + child_price * (total_tickets - adult_tickets)

/-- Theorem stating that given the specific conditions of the problem, the total amount collected is $275. -/
theorem community_center_ticket_sales :
  let adult_price : ℕ := 5
  let child_price : ℕ := 2
  let total_tickets : ℕ := 85
  let adult_tickets : ℕ := 35
  total_amount_collected adult_price child_price total_tickets adult_tickets = 275 := by
sorry

end community_center_ticket_sales_l1987_198737


namespace trigonometric_identity_l1987_198771

theorem trigonometric_identity (α : Real) :
  (2 * (Real.cos (2 * α))^2 - 1) / 
  (2 * Real.tan (π/4 - 2*α) * (Real.sin (3*π/4 - 2*α))^2) - 
  Real.tan (2*α) + Real.cos (2*α) - Real.sin (2*α) = 
  (2 * Real.sqrt 2 * Real.sin (π/4 - 2*α) * (Real.cos α)^2) / 
  Real.cos (2*α) :=
by sorry

end trigonometric_identity_l1987_198771


namespace add_preserves_inequality_l1987_198758

theorem add_preserves_inequality (a b : ℝ) (h : a > b) : 2 + a > 2 + b := by
  sorry

end add_preserves_inequality_l1987_198758


namespace triangle_abc_properties_l1987_198720

theorem triangle_abc_properties (a b c A B C : ℝ) (h1 : a = b * Real.sin A + Real.sqrt 3 * a * Real.cos B)
  (h2 : b = 4) (h3 : (1/2) * a * c = 4) :
  B = Real.pi / 2 ∧ a + b + c = 4 + 4 * Real.sqrt 2 := by
  sorry

end triangle_abc_properties_l1987_198720


namespace banana_count_l1987_198769

theorem banana_count (apples oranges total : ℕ) (h1 : apples = 9) (h2 : oranges = 15) (h3 : total = 146) :
  ∃ bananas : ℕ, 
    3 * (apples + oranges + bananas) + (apples - 2 + oranges - 2 + bananas - 2) = total ∧ 
    bananas = 52 :=
by sorry

end banana_count_l1987_198769


namespace inverse_multiplication_l1987_198718

theorem inverse_multiplication (a : ℝ) (h : a ≠ 0) : a * a⁻¹ = 1 := by
  sorry

end inverse_multiplication_l1987_198718


namespace vertical_asymptote_at_three_l1987_198741

/-- The function f(x) = (x^3 + x^2 + 1) / (x - 3) has a vertical asymptote at x = 3 -/
theorem vertical_asymptote_at_three (x : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ (x^3 + x^2 + 1) / (x - 3)
  ∃ (ε : ℝ), ε > 0 ∧ ∀ (δ : ℝ), 0 < δ → δ < ε → |f (3 + δ)| > (1 / δ) ∧ |f (3 - δ)| > (1 / δ) :=
by
  sorry

end vertical_asymptote_at_three_l1987_198741


namespace problem_statement_l1987_198702

theorem problem_statement (x : ℝ) (h : x = 4) : 5 * x + 3 - x^2 = 7 := by
  sorry

end problem_statement_l1987_198702


namespace alicia_local_taxes_l1987_198700

theorem alicia_local_taxes (hourly_wage : ℝ) (tax_rate : ℝ) : 
  hourly_wage = 25 → tax_rate = 0.02 → hourly_wage * tax_rate * 100 = 50 := by
  sorry

end alicia_local_taxes_l1987_198700


namespace eleven_play_both_l1987_198727

/-- Represents a sports club with members playing badminton and tennis -/
structure SportsClub where
  total : ℕ
  badminton : ℕ
  tennis : ℕ
  neither : ℕ

/-- Calculates the number of members playing both badminton and tennis -/
def playsBoth (club : SportsClub) : ℕ :=
  club.badminton + club.tennis - (club.total - club.neither)

/-- Theorem: In the given sports club, 11 members play both badminton and tennis -/
theorem eleven_play_both (club : SportsClub)
  (h_total : club.total = 27)
  (h_badminton : club.badminton = 17)
  (h_tennis : club.tennis = 19)
  (h_neither : club.neither = 2) :
  playsBoth club = 11 := by
  sorry

#eval playsBoth { total := 27, badminton := 17, tennis := 19, neither := 2 }

end eleven_play_both_l1987_198727


namespace same_prime_factors_imply_power_of_two_l1987_198789

theorem same_prime_factors_imply_power_of_two (b m n : ℕ) 
  (hb : b > 1) 
  (hm : m > 0) 
  (hn : n > 0) 
  (hmn : m ≠ n) 
  (h_same_factors : ∀ p : ℕ, Prime p → (p ∣ (b^m - 1) ↔ p ∣ (b^n - 1))) : 
  ∃ k : ℕ, b + 1 = 2^k :=
sorry

end same_prime_factors_imply_power_of_two_l1987_198789


namespace quadratic_expression_l1987_198787

theorem quadratic_expression (m : ℤ) : 
  (∃ (a b c : ℤ), a * m^2 + b * m + c = (m - 8) * (m + 3)) → 
  (∃ (a b c : ℤ), a * m^2 + b * m + c = m^2 - 5*m - 24) :=
by sorry

end quadratic_expression_l1987_198787


namespace fraction_sum_and_reciprocal_sum_integer_fraction_sum_and_reciprocal_sum_integer_distinct_numerators_l1987_198730

theorem fraction_sum_and_reciprocal_sum_integer :
  ∃ (a b c d e f : ℕ), 
    (0 < a ∧ a < b) ∧ 
    (0 < c ∧ c < d) ∧ 
    (0 < e ∧ e < f) ∧
    (Nat.gcd a b = 1) ∧
    (Nat.gcd c d = 1) ∧
    (Nat.gcd e f = 1) ∧
    (a : ℚ) / b + (c : ℚ) / d + (e : ℚ) / f = 1 ∧
    ∃ (n : ℕ), (b : ℚ) / a + (d : ℚ) / c + (f : ℚ) / e = n :=
by sorry

theorem fraction_sum_and_reciprocal_sum_integer_distinct_numerators :
  ∃ (a b c d e f : ℕ), 
    (0 < a ∧ a < b) ∧ 
    (0 < c ∧ c < d) ∧ 
    (0 < e ∧ e < f) ∧
    (Nat.gcd a b = 1) ∧
    (Nat.gcd c d = 1) ∧
    (Nat.gcd e f = 1) ∧
    a ≠ c ∧ a ≠ e ∧ c ≠ e ∧
    (a : ℚ) / b + (c : ℚ) / d + (e : ℚ) / f = 1 ∧
    ∃ (n : ℕ), (b : ℚ) / a + (d : ℚ) / c + (f : ℚ) / e = n :=
by sorry

end fraction_sum_and_reciprocal_sum_integer_fraction_sum_and_reciprocal_sum_integer_distinct_numerators_l1987_198730


namespace max_balls_in_specific_cylinder_l1987_198780

/-- The maximum number of unit balls that can be placed in a cylinder -/
def max_balls_in_cylinder (cylinder_diameter : ℝ) (cylinder_height : ℝ) (ball_diameter : ℝ) : ℕ :=
  sorry

/-- Theorem: In a cylinder with diameter √2 + 1 and height 8, the maximum number of balls with diameter 1 that can be placed is 36 -/
theorem max_balls_in_specific_cylinder :
  max_balls_in_cylinder (Real.sqrt 2 + 1) 8 1 = 36 := by
  sorry

end max_balls_in_specific_cylinder_l1987_198780


namespace conditional_probability_balls_l1987_198797

/-- Represents the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The probability of event A (drawing two balls of different colors) -/
def probA : ℚ := (choose 5 1 * choose 3 1 + choose 5 1 * choose 4 1 + choose 3 1 * choose 4 1) / choose 12 2

/-- The probability of event B (drawing one yellow and one blue ball) -/
def probB : ℚ := (choose 5 1 * choose 4 1) / choose 12 2

/-- The probability of both events A and B occurring -/
def probAB : ℚ := probB

theorem conditional_probability_balls :
  probAB / probA = 20 / 47 := by sorry

end conditional_probability_balls_l1987_198797


namespace set_no_duplicate_elements_l1987_198792

theorem set_no_duplicate_elements {α : Type*} (S : Set α) :
  ∀ x ∈ S, ∀ y ∈ S, x = y → x = y :=
by sorry

end set_no_duplicate_elements_l1987_198792


namespace four_integers_sum_l1987_198750

theorem four_integers_sum (a b c d : ℤ) :
  a + b + c = 6 ∧
  a + b + d = 7 ∧
  a + c + d = 8 ∧
  b + c + d = 9 →
  a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 4 := by
sorry

end four_integers_sum_l1987_198750


namespace no_common_complex_root_l1987_198786

theorem no_common_complex_root :
  ¬ ∃ (α : ℂ) (a b : ℚ), α^5 - α - 1 = 0 ∧ α^2 + a*α + b = 0 := by
  sorry

end no_common_complex_root_l1987_198786


namespace binary_1011_equals_11_l1987_198729

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_1011_equals_11 :
  binary_to_decimal [true, true, false, true] = 11 := by
  sorry

end binary_1011_equals_11_l1987_198729


namespace walk_in_closet_doorway_width_l1987_198794

/-- Represents the dimensions of a rectangular room -/
structure RoomDimensions where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Represents a rectangular opening (door or window) -/
structure Opening where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangular surface -/
def rectangleArea (width height : ℝ) : ℝ := width * height

/-- Calculates the total wall area of a room -/
def totalWallArea (room : RoomDimensions) : ℝ :=
  2 * (room.width + room.length) * room.height

/-- Calculates the area of an opening -/
def openingArea (opening : Opening) : ℝ := rectangleArea opening.width opening.height

theorem walk_in_closet_doorway_width 
  (room : RoomDimensions)
  (doorway1 : Opening)
  (window : Opening)
  (closetDoorwayHeight : ℝ)
  (areaToPaint : ℝ)
  (h1 : room.width = 20)
  (h2 : room.length = 20)
  (h3 : room.height = 8)
  (h4 : doorway1.width = 3)
  (h5 : doorway1.height = 7)
  (h6 : window.width = 6)
  (h7 : window.height = 4)
  (h8 : closetDoorwayHeight = 7)
  (h9 : areaToPaint = 560) :
  ∃ (closetDoorwayWidth : ℝ), 
    closetDoorwayWidth = 5 ∧
    areaToPaint = totalWallArea room - openingArea doorway1 - openingArea window - rectangleArea closetDoorwayWidth closetDoorwayHeight :=
by sorry

end walk_in_closet_doorway_width_l1987_198794


namespace equal_probability_for_all_l1987_198742

/-- Represents the sampling method used in the TV show -/
structure SamplingMethod where
  total_population : ℕ
  sample_size : ℕ
  removed_first : ℕ
  
/-- The probability of being selected for each individual in the population -/
def selection_probability (sm : SamplingMethod) : ℚ :=
  sm.sample_size / sm.total_population

/-- The specific sampling method used in the TV show -/
def tv_show_sampling : SamplingMethod := {
  total_population := 2014
  sample_size := 50
  removed_first := 14
}

theorem equal_probability_for_all (sm : SamplingMethod) :
  selection_probability sm = 25 / 1007 :=
sorry

#check equal_probability_for_all tv_show_sampling

end equal_probability_for_all_l1987_198742


namespace michael_needs_additional_money_l1987_198749

def michael_money : ℝ := 50
def cake_cost : ℝ := 20
def bouquet_cost : ℝ := 36
def balloons_cost : ℝ := 5
def perfume_cost_gbp : ℝ := 30
def photo_album_cost_eur : ℝ := 25
def gbp_to_usd : ℝ := 1.4
def eur_to_usd : ℝ := 1.2

theorem michael_needs_additional_money :
  let perfume_cost_usd := perfume_cost_gbp * gbp_to_usd
  let photo_album_cost_usd := photo_album_cost_eur * eur_to_usd
  let total_cost := cake_cost + bouquet_cost + balloons_cost + perfume_cost_usd + photo_album_cost_usd
  total_cost - michael_money = 83 := by sorry

end michael_needs_additional_money_l1987_198749


namespace soldier_height_arrangement_l1987_198714

theorem soldier_height_arrangement (n : ℕ) (a b : Fin n → ℝ) :
  (∀ i : Fin n, a i ≤ b i) →
  (∀ i j : Fin n, i < j → a i ≥ a j) →
  (∀ i j : Fin n, i < j → b i ≥ b j) →
  ∀ i : Fin n, a i ≤ b i :=
by sorry

end soldier_height_arrangement_l1987_198714


namespace sin_45_degrees_l1987_198740

theorem sin_45_degrees : Real.sin (π / 4) = Real.sqrt 2 / 2 := by
  sorry

end sin_45_degrees_l1987_198740


namespace sushi_cost_l1987_198783

theorem sushi_cost (j e : ℝ) (h1 : j + e = 200) (h2 : e = 9 * j) : e = 180 := by
  sorry

end sushi_cost_l1987_198783


namespace algebraic_identities_l1987_198724

theorem algebraic_identities (x y : ℝ) : 
  ((2*x - 3*y)^2 = 4*x^2 - 12*x*y + 9*y^2) ∧ 
  ((x + y)*(x + y)*(x^2 + y^2) = x^4 + 2*x^2*y^2 + y^4 + 2*x^3*y + 2*x*y^3) := by
sorry

end algebraic_identities_l1987_198724


namespace z_in_terms_of_a_b_s_l1987_198703

theorem z_in_terms_of_a_b_s 
  (z a b s : ℝ) 
  (hz : z ≠ 0) 
  (heq : z = a^3 * b^2 + 6*z*s - 9*s^2) :
  z = (a^3 * b^2 - 9*s^2) / (1 - 6*s) :=
by sorry

end z_in_terms_of_a_b_s_l1987_198703


namespace quadratic_equation_from_means_l1987_198757

theorem quadratic_equation_from_means (a b : ℝ) : 
  (a + b) / 2 = 8 → 
  Real.sqrt (a * b) = 10 → 
  ∀ x, x^2 - 16*x + 100 = 0 ↔ (x = a ∨ x = b) := by
sorry

end quadratic_equation_from_means_l1987_198757


namespace sin_10_over_1_minus_sqrt3_tan_10_l1987_198760

theorem sin_10_over_1_minus_sqrt3_tan_10 :
  (Real.sin (10 * π / 180)) / (1 - Real.sqrt 3 * Real.tan (10 * π / 180)) = 1 / 2 := by
  sorry

end sin_10_over_1_minus_sqrt3_tan_10_l1987_198760


namespace fraction_power_product_l1987_198728

theorem fraction_power_product : (2 / 3 : ℚ)^4 * (1 / 5 : ℚ)^2 = 16 / 2025 := by
  sorry

end fraction_power_product_l1987_198728


namespace price_increase_to_equality_l1987_198708

theorem price_increase_to_equality (price_B : ℝ) (price_A : ℝ) 
    (h1 : price_A = price_B * 0.8) : 
  (price_B - price_A) / price_A * 100 = 25 := by
  sorry

end price_increase_to_equality_l1987_198708


namespace steves_cookies_l1987_198790

theorem steves_cookies (total_spent milk_cost cereal_cost banana_cost apple_cost : ℚ)
  (cereal_boxes banana_count apple_count : ℕ)
  (h_total : total_spent = 25)
  (h_milk : milk_cost = 3)
  (h_cereal : cereal_cost = 7/2)
  (h_cereal_boxes : cereal_boxes = 2)
  (h_banana : banana_cost = 1/4)
  (h_banana_count : banana_count = 4)
  (h_apple : apple_cost = 1/2)
  (h_apple_count : apple_count = 4)
  (h_cookie_cost : ∀ x, x = 2 * milk_cost) :
  ∃ (cookie_boxes : ℕ), cookie_boxes = 2 ∧
    total_spent = milk_cost + cereal_cost * cereal_boxes + 
      banana_cost * banana_count + apple_cost * apple_count + 
      (2 * milk_cost) * cookie_boxes :=
by sorry

end steves_cookies_l1987_198790


namespace arithmetic_calculations_l1987_198754

theorem arithmetic_calculations : 
  ((-23) + 13 - 12 = -22) ∧ 
  ((-2)^3 / 4 + 3 * (-5) = -17) ∧ 
  ((-24) * (1/2 - 3/4 - 1/8) = 9) ∧ 
  ((2-7) / 5^2 + (-1)^2023 * (1/10) = -3/10) := by
sorry

end arithmetic_calculations_l1987_198754


namespace triangle_side_values_l1987_198731

def triangle_exists (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_side_values :
  ∀ x : ℕ+, 
    (triangle_exists 8 11 (x.val ^ 2)) ↔ (x = 2 ∨ x = 3 ∨ x = 4) :=
by sorry

end triangle_side_values_l1987_198731


namespace apples_in_basket_l1987_198764

/-- Calculates the number of apples remaining in a basket after removals. -/
def remaining_apples (initial : ℕ) (ricki_removal : ℕ) : ℕ :=
  initial - (ricki_removal + 2 * ricki_removal)

/-- Theorem stating that given the initial conditions, 32 apples remain. -/
theorem apples_in_basket : remaining_apples 74 14 = 32 := by
  sorry

end apples_in_basket_l1987_198764


namespace project_completion_theorem_l1987_198766

/-- The number of days to complete a project given two workers with different rates -/
def project_completion_days (a_rate b_rate : ℚ) (a_quit_before : ℕ) : ℕ :=
  let total_days := 20
  total_days

theorem project_completion_theorem (a_rate b_rate : ℚ) (a_quit_before : ℕ) :
  a_rate = 1/20 ∧ b_rate = 1/40 ∧ a_quit_before = 10 →
  (project_completion_days a_rate b_rate a_quit_before - a_quit_before) * a_rate +
  project_completion_days a_rate b_rate a_quit_before * b_rate = 1 :=
by
  sorry

#eval project_completion_days (1/20) (1/40) 10

end project_completion_theorem_l1987_198766


namespace simplify_sqrt_difference_l1987_198795

theorem simplify_sqrt_difference : 
  Real.sqrt 300 / Real.sqrt 75 - Real.sqrt 220 / Real.sqrt 55 = 0 := by
  sorry

end simplify_sqrt_difference_l1987_198795


namespace matrix_inverse_and_transformation_l1987_198744

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 1, 2]

theorem matrix_inverse_and_transformation :
  let A_inv : Matrix (Fin 2) (Fin 2) ℝ := !![2, -3; -1, 2]
  let P : Fin 2 → ℝ := ![3, -1]
  (A⁻¹ = A_inv) ∧ (A.mulVec P = ![3, 1]) := by sorry

end matrix_inverse_and_transformation_l1987_198744


namespace oil_production_per_capita_correct_l1987_198723

/-- Oil production per capita for a region -/
structure OilProductionPerCapita where
  region : String
  value : Float

/-- Given oil production per capita data -/
def given_data : List OilProductionPerCapita := [
  ⟨"West", 55.084⟩,
  ⟨"Non-West", 214.59⟩,
  ⟨"Russia", 1038.33⟩
]

/-- Theorem: The oil production per capita for West, Non-West, and Russia are as given -/
theorem oil_production_per_capita_correct :
  ∀ region value, OilProductionPerCapita.mk region value ∈ given_data →
  (region = "West" → value = 55.084) ∧
  (region = "Non-West" → value = 214.59) ∧
  (region = "Russia" → value = 1038.33) :=
by sorry

end oil_production_per_capita_correct_l1987_198723


namespace exam_maximum_marks_l1987_198772

/-- The maximum marks for an exam -/
def maximum_marks : ℕ := 467

/-- The passing percentage as a rational number -/
def passing_percentage : ℚ := 60 / 100

/-- The marks scored by the student -/
def student_marks : ℕ := 200

/-- The marks by which the student failed -/
def failing_margin : ℕ := 80

theorem exam_maximum_marks :
  (↑maximum_marks * passing_percentage : ℚ).ceil = student_marks + failing_margin ∧
  (↑maximum_marks * passing_percentage : ℚ).ceil < ↑maximum_marks ∧
  (↑(maximum_marks - 1) * passing_percentage : ℚ).ceil < student_marks + failing_margin :=
sorry

end exam_maximum_marks_l1987_198772


namespace termite_ridden_homes_l1987_198712

theorem termite_ridden_homes (total_homes : ℝ) (termite_ridden_homes : ℝ) 
  (h1 : termite_ridden_homes > 0)
  (h2 : (4 : ℝ) / 7 * termite_ridden_homes = termite_ridden_homes - (1 : ℝ) / 7 * total_homes) :
  termite_ridden_homes = (1 : ℝ) / 3 * total_homes := by
sorry

end termite_ridden_homes_l1987_198712


namespace quadratic_equation_real_roots_l1987_198785

/-- Given a quadratic equation with complex coefficients that has real roots, 
    prove that the coefficient 'a' must be equal to -1. -/
theorem quadratic_equation_real_roots (a : ℝ) : 
  (∃ x : ℝ, (a * (1 + Complex.I)) * x^2 + (1 + a^2 * Complex.I) * x + (a^2 + Complex.I) = 0) → 
  a = -1 := by
sorry

end quadratic_equation_real_roots_l1987_198785


namespace xy_value_l1987_198736

theorem xy_value (x y : ℝ) 
  (h1 : (8:ℝ)^x / (4:ℝ)^(x+y) = 32)
  (h2 : (27:ℝ)^(x+y) / (9:ℝ)^(2*y) = 729) :
  x * y = -63/25 := by
sorry

end xy_value_l1987_198736


namespace parking_savings_yearly_parking_savings_l1987_198738

theorem parking_savings : ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
  fun weekly_rate monthly_rate weeks_per_year months_per_year savings =>
    weekly_rate * weeks_per_year - monthly_rate * months_per_year = savings

/-- Proof of yearly savings when renting monthly instead of weekly --/
theorem yearly_parking_savings : parking_savings 10 40 52 12 40 := by
  sorry

end parking_savings_yearly_parking_savings_l1987_198738


namespace task_pages_l1987_198733

/-- Represents the number of pages in the printing task -/
def P : ℕ := 480

/-- Represents the rate of Printer A in pages per minute -/
def rate_A : ℚ := P / 60

/-- Represents the rate of Printer B in pages per minute -/
def rate_B : ℚ := rate_A + 4

/-- Theorem stating that the number of pages in the task is 480 -/
theorem task_pages : P = 480 := by
  have h1 : rate_A + rate_B = P / 40 := by sorry
  have h2 : rate_A = P / 60 := by sorry
  have h3 : rate_B = rate_A + 4 := by sorry
  sorry

#check task_pages

end task_pages_l1987_198733


namespace polar_to_cartesian_circle_l1987_198732

theorem polar_to_cartesian_circle (x y ρ : ℝ) :
  ρ = 2 ↔ x^2 + y^2 = 4 :=
sorry

end polar_to_cartesian_circle_l1987_198732


namespace square_sum_implies_abs_sum_l1987_198756

theorem square_sum_implies_abs_sum (a b : ℝ) :
  a^2 + b^2 > 1 → |a| + |b| > 1 := by sorry

end square_sum_implies_abs_sum_l1987_198756


namespace complex_norm_problem_l1987_198715

theorem complex_norm_problem (z w : ℂ) 
  (h1 : Complex.abs (3 * z - w) = 15)
  (h2 : Complex.abs (z + 3 * w) = 3)
  (h3 : Complex.abs (z - w) = 1) :
  Complex.abs z = Real.sqrt (225 / 7) :=
sorry

end complex_norm_problem_l1987_198715


namespace university_students_count_l1987_198793

/-- Proves that given the initial ratio of male to female students and the ratio after increasing female students, the total number of students after the increase is 4800 -/
theorem university_students_count (x y : ℕ) : 
  x = 3 * y ∧ 
  9 * (y + 400) = 4 * x → 
  x + y + 400 = 4800 :=
by sorry

end university_students_count_l1987_198793


namespace benjamin_walks_158_miles_l1987_198710

/-- Calculates the total miles Benjamin walks in a week -/
def total_miles_walked : ℕ :=
  let work_distance := 8
  let work_days := 5
  let dog_walk_distance := 3
  let dog_walks_per_day := 2
  let days_in_week := 7
  let friend_distance := 5
  let friend_visits := 1
  let store_distance := 4
  let store_visits := 2
  let hike_distance := 10

  let work_miles := work_distance * 2 * work_days
  let dog_miles := dog_walk_distance * dog_walks_per_day * days_in_week
  let friend_miles := friend_distance * 2 * friend_visits
  let store_miles := store_distance * 2 * store_visits
  let hike_miles := hike_distance

  work_miles + dog_miles + friend_miles + store_miles + hike_miles

theorem benjamin_walks_158_miles : total_miles_walked = 158 := by
  sorry

end benjamin_walks_158_miles_l1987_198710


namespace non_intersecting_paths_l1987_198751

/-- The number of non-intersecting pairs of paths on a grid -/
theorem non_intersecting_paths 
  (m n p q : ℕ+) 
  (h1 : p < m) 
  (h2 : q < n) : 
  ∃ S : ℕ, S = Nat.choose (m + n) m * Nat.choose (m + q - p) q - 
              Nat.choose (m + q) m * Nat.choose (m + n - p) n :=
by sorry

end non_intersecting_paths_l1987_198751


namespace toms_age_ratio_l1987_198734

/-- Proves that the ratio of Tom's current age to the number of years ago when his age was three times the sum of his children's ages is 5.5 -/
theorem toms_age_ratio :
  ∀ (T N : ℝ),
  (∃ (a b c d : ℝ), T = a + b + c + d) →  -- T is the sum of four children's ages
  (T - N = 3 * (T - 4 * N)) →              -- N years ago condition
  T / N = 5.5 := by
sorry

end toms_age_ratio_l1987_198734


namespace lcm_of_ratio_3_4_l1987_198784

/-- Given two natural numbers with a ratio of 3:4, where one number is 45 and the other is 60, 
    their least common multiple (LCM) is 180. -/
theorem lcm_of_ratio_3_4 (a b : ℕ) (h_ratio : 3 * b = 4 * a) (h_a : a = 45) (h_b : b = 60) : 
  Nat.lcm a b = 180 := by
  sorry

end lcm_of_ratio_3_4_l1987_198784


namespace at_least_one_solution_l1987_198704

open Complex

-- Define the equation
def satisfies_equation (z : ℂ) : Prop := exp z = z^2 + 1

-- Define the constraint
def within_bound (z : ℂ) : Prop := abs z < 20

-- Theorem statement
theorem at_least_one_solution :
  ∃ z : ℂ, satisfies_equation z ∧ within_bound z :=
sorry

end at_least_one_solution_l1987_198704


namespace sufficient_condition_range_l1987_198762

theorem sufficient_condition_range (a : ℝ) : 
  (∀ x : ℝ, |x - a| ≤ 1 → x^2 - 5*x + 4 ≤ 0) → 
  2 ≤ a ∧ a ≤ 3 := by
  sorry

end sufficient_condition_range_l1987_198762


namespace freshman_groups_l1987_198774

theorem freshman_groups (total_freshmen : Nat) (group_decrease : Nat) :
  total_freshmen = 2376 →
  group_decrease = 9 →
  ∃ (initial_groups final_groups : Nat),
    initial_groups = final_groups + group_decrease ∧
    total_freshmen % initial_groups = 0 ∧
    total_freshmen % final_groups = 0 ∧
    total_freshmen / final_groups < 30 ∧
    final_groups = 99 := by
  sorry

end freshman_groups_l1987_198774


namespace zeros_after_one_in_8000_to_50_l1987_198726

theorem zeros_after_one_in_8000_to_50 :
  let n : ℕ := 8000
  let k : ℕ := 50
  let base_ten_factor : ℕ := 3
  n = 8 * (10 ^ base_ten_factor) →
  (∃ m : ℕ, n^k = m * 10^(base_ten_factor * k) ∧ m % 10 ≠ 0) :=
by sorry

end zeros_after_one_in_8000_to_50_l1987_198726


namespace negative_cube_squared_l1987_198719

theorem negative_cube_squared (a b : ℝ) : (-a^3 * b)^2 = a^6 * b^2 := by
  sorry

end negative_cube_squared_l1987_198719


namespace kims_test_probability_l1987_198716

theorem kims_test_probability (p_english : ℝ) (p_history : ℝ) 
  (h_english : p_english = 5/9)
  (h_history : p_history = 1/3)
  (h_independent : True) -- We don't need to explicitly define independence in this statement
  : (1 - p_english) * p_history = 4/27 := by
  sorry

end kims_test_probability_l1987_198716


namespace no_solution_exists_l1987_198753

-- Function to reverse a number
def reverseNumber (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem no_solution_exists :
  ¬ ∃ (x : ℕ), x + 276 = 435 ∧ reverseNumber x = 731 := by
  sorry

end no_solution_exists_l1987_198753


namespace condition1_condition2_max_type_A_dictionaries_l1987_198747

/-- The price of dictionary A -/
def price_A : ℝ := 70

/-- The price of dictionary B -/
def price_B : ℝ := 50

/-- The total number of dictionaries to be purchased -/
def total_dictionaries : ℕ := 300

/-- The maximum total cost -/
def max_cost : ℝ := 16000

/-- Verification of the first condition -/
theorem condition1 : price_A + 2 * price_B = 170 := by sorry

/-- Verification of the second condition -/
theorem condition2 : 2 * price_A + 3 * price_B = 290 := by sorry

/-- The main theorem proving the maximum number of type A dictionaries -/
theorem max_type_A_dictionaries : 
  ∀ m : ℕ, m ≤ total_dictionaries ∧ 
    m * price_A + (total_dictionaries - m) * price_B ≤ max_cost → 
    m ≤ 50 := by sorry

end condition1_condition2_max_type_A_dictionaries_l1987_198747


namespace minimum_driving_age_l1987_198709

/-- The minimum driving age problem -/
theorem minimum_driving_age 
  (kayla_age : ℕ) 
  (kimiko_age : ℕ) 
  (min_driving_age : ℕ) 
  (h1 : kayla_age * 2 = kimiko_age) 
  (h2 : kimiko_age = 26) 
  (h3 : min_driving_age = kayla_age + 5) : 
  min_driving_age = 18 := by
sorry

end minimum_driving_age_l1987_198709


namespace number_of_coverings_number_of_coverings_eq_coverings_order_invariant_l1987_198717

/-- The number of coverings of a finite set -/
theorem number_of_coverings (n : ℕ) : ℕ := 
  2^(2^n - 1)

/-- The number of coverings of a finite set X with n elements is 2^(2^n - 1) -/
theorem number_of_coverings_eq (X : Finset ℕ) (h : X.card = n) :
  (Finset.powerset X).card = number_of_coverings n := by
  sorry

/-- The order of covering sets does not affect the total number of coverings -/
theorem coverings_order_invariant (X : Finset ℕ) (C₁ C₂ : Finset (Finset ℕ)) 
  (h₁ : ∀ x ∈ X, ∃ S ∈ C₁, x ∈ S) (h₂ : ∀ x ∈ X, ∃ S ∈ C₂, x ∈ S) :
  C₁.card = C₂.card := by
  sorry

end number_of_coverings_number_of_coverings_eq_coverings_order_invariant_l1987_198717


namespace building_floors_upper_bound_l1987_198763

theorem building_floors_upper_bound 
  (num_elevators : ℕ) 
  (floors_per_elevator : ℕ) 
  (h1 : num_elevators = 7)
  (h2 : floors_per_elevator = 6)
  (h3 : ∀ (f1 f2 : ℕ), f1 ≠ f2 → ∃ (e : ℕ), e ≤ num_elevators ∧ 
    (∃ (s : Finset ℕ), s.card = floors_per_elevator ∧ f1 ∈ s ∧ f2 ∈ s)) :
  ∃ (max_floors : ℕ), max_floors ≤ 14 ∧ 
    ∀ (n : ℕ), (∀ (f1 f2 : ℕ), f1 ≤ n ∧ f2 ≤ n ∧ f1 ≠ f2 → 
      ∃ (e : ℕ), e ≤ num_elevators ∧ 
        (∃ (s : Finset ℕ), s.card = floors_per_elevator ∧ f1 ∈ s ∧ f2 ∈ s)) → 
    n ≤ max_floors := by
  sorry

end building_floors_upper_bound_l1987_198763


namespace bowling_ball_weight_proof_l1987_198739

/-- The weight of one bowling ball in pounds -/
def bowling_ball_weight : ℚ := 56 / 5

/-- The weight of one canoe in pounds -/
def canoe_weight : ℚ := 28

theorem bowling_ball_weight_proof :
  (5 : ℚ) * bowling_ball_weight = 2 * canoe_weight ∧
  (3 : ℚ) * canoe_weight = 84 →
  bowling_ball_weight = 56 / 5 := by
sorry

end bowling_ball_weight_proof_l1987_198739


namespace rotation_equivalence_l1987_198788

/-- 
Given:
- A point P is rotated 750 degrees clockwise about point Q, resulting in point R.
- The same point P is rotated y degrees counterclockwise about point Q, also resulting in point R.
- y < 360

Prove that y = 330.
-/
theorem rotation_equivalence (y : ℝ) (h1 : y < 360) : 
  (750 % 360 : ℝ) + y = 360 → y = 330 := by
  sorry

end rotation_equivalence_l1987_198788


namespace M_properties_l1987_198752

def M (n : ℕ) : ℤ := (-2) ^ n

theorem M_properties :
  (M 5 + M 6 = 32) ∧
  (2 * M 2015 + M 2016 = 0) ∧
  (∀ n : ℕ, 2 * M n + M (n + 1) = 0) := by
  sorry

end M_properties_l1987_198752


namespace credit_card_more_profitable_min_days_for_credit_card_profitability_l1987_198721

/-- Represents the purchase amount in rubles -/
def purchase_amount : ℝ := 20000

/-- Represents the credit card cashback rate -/
def credit_cashback_rate : ℝ := 0.005

/-- Represents the debit card cashback rate -/
def debit_cashback_rate : ℝ := 0.01

/-- Represents the annual interest rate on the debit card -/
def annual_interest_rate : ℝ := 0.06

/-- Represents the number of days in a month (assumed) -/
def days_in_month : ℕ := 30

/-- Represents the number of days in a year -/
def days_in_year : ℕ := 360

/-- Theorem stating the minimum number of days for credit card to be more profitable -/
theorem credit_card_more_profitable (N : ℕ) : 
  (N : ℝ) * annual_interest_rate * purchase_amount / days_in_year + 
  credit_cashback_rate * purchase_amount > 
  debit_cashback_rate * purchase_amount → N ≥ 31 := by
  sorry

/-- Theorem stating that 31 days is the minimum for credit card to be more profitable -/
theorem min_days_for_credit_card_profitability : 
  ∃ (N : ℕ), N = 31 ∧ 
  (∀ (M : ℕ), M < N → 
    (M : ℝ) * annual_interest_rate * purchase_amount / days_in_year + 
    credit_cashback_rate * purchase_amount ≤ 
    debit_cashback_rate * purchase_amount) ∧
  ((N : ℝ) * annual_interest_rate * purchase_amount / days_in_year + 
   credit_cashback_rate * purchase_amount > 
   debit_cashback_rate * purchase_amount) := by
  sorry

end credit_card_more_profitable_min_days_for_credit_card_profitability_l1987_198721


namespace solution_of_exponential_equation_l1987_198791

theorem solution_of_exponential_equation :
  ∃ x : ℝ, (2 : ℝ)^(x - 3) = 8^(x + 1) ↔ x = -3 := by sorry

end solution_of_exponential_equation_l1987_198791


namespace perpendicular_vectors_l1987_198796

/-- Given two vectors a and b in ℝ², prove that when a = (1,3) and b = (x,1) are perpendicular, x = -3 -/
theorem perpendicular_vectors (x : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 3]
  let b : Fin 2 → ℝ := ![x, 1]
  (∀ i, i < 2 → a i * b i = 0) → x = -3 := by
  sorry

end perpendicular_vectors_l1987_198796


namespace catch_up_distance_l1987_198777

/-- Prove that B catches up with A 200 km from the start -/
theorem catch_up_distance (speed_A speed_B : ℝ) (time_diff : ℝ) : 
  speed_A = 10 → 
  speed_B = 20 → 
  time_diff = 10 → 
  speed_B * (time_diff + (speed_B * time_diff - speed_A * time_diff) / (speed_B - speed_A)) = 200 := by
  sorry

#check catch_up_distance

end catch_up_distance_l1987_198777


namespace line_contains_point_l1987_198761

/-- Given a line with equation -2/3 - 3kx = 7y that contains the point (1/3, -5), 
    prove that the value of k is 103/3. -/
theorem line_contains_point (k : ℚ) : 
  (-2/3 : ℚ) - 3 * k * (1/3 : ℚ) = 7 * (-5 : ℚ) → k = 103/3 := by
  sorry

end line_contains_point_l1987_198761
