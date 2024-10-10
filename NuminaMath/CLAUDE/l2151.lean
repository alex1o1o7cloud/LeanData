import Mathlib

namespace slope_angle_of_parametric_line_l2151_215126

/-- The slope angle of a line with parametric equations x = 2 + t and y = 1 + (√3/3)t is π/6 -/
theorem slope_angle_of_parametric_line : 
  ∀ (t : ℝ), 
  let x := 2 + t
  let y := 1 + (Real.sqrt 3 / 3) * t
  let slope := (Real.sqrt 3 / 3)
  let slope_angle := Real.arctan slope
  slope_angle = π / 6 := by sorry

end slope_angle_of_parametric_line_l2151_215126


namespace top_card_after_74_shuffles_l2151_215114

/-- Represents the order of cards -/
inductive Card
| A
| B
| C
| D
| E

/-- Represents the stack of cards -/
def Stack := List Card

/-- The initial configuration of cards -/
def initial_stack : Stack := [Card.A, Card.B, Card.C, Card.D, Card.E]

/-- Performs one shuffle operation on the stack -/
def shuffle (s : Stack) : Stack :=
  match s with
  | x :: y :: rest => rest ++ [y, x]
  | _ => s

/-- Performs n shuffle operations on the stack -/
def n_shuffles (n : Nat) (s : Stack) : Stack :=
  match n with
  | 0 => s
  | n + 1 => shuffle (n_shuffles n s)

theorem top_card_after_74_shuffles :
  (n_shuffles 74 initial_stack).head? = some Card.E := by
  sorry

end top_card_after_74_shuffles_l2151_215114


namespace parabola_properties_l2151_215165

-- Define the parabolas
def parabola_G (x y : ℝ) : Prop := x^2 = y
def parabola_M (x y : ℝ) : Prop := y^2 = 4*x

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the focus of parabola M
def focus_M : ℝ × ℝ := (1, 0)

-- Define the property of being inscribed in a parabola
def inscribed_in_parabola (t : Triangle) (p : ℝ → ℝ → Prop) : Prop :=
  p t.A.1 t.A.2 ∧ p t.B.1 t.B.2 ∧ p t.C.1 t.C.2

-- Define the property of a line being tangent to a parabola
def line_tangent_to_parabola (p q : ℝ × ℝ) (parabola : ℝ → ℝ → Prop) : Prop :=
  ∃ (t : ℝ), parabola ((1-t)*p.1 + t*q.1) ((1-t)*p.2 + t*q.2)

-- Define the property of points being concyclic
def concyclic (p q r s : ℝ × ℝ) : Prop :=
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2 ∧
    (q.1 - center.1)^2 + (q.2 - center.2)^2 = radius^2 ∧
    (r.1 - center.1)^2 + (r.2 - center.2)^2 = radius^2 ∧
    (s.1 - center.1)^2 + (s.2 - center.2)^2 = radius^2

theorem parabola_properties (t : Triangle) 
  (h1 : inscribed_in_parabola t parabola_G)
  (h2 : line_tangent_to_parabola t.A t.B parabola_M)
  (h3 : line_tangent_to_parabola t.A t.C parabola_M) :
  line_tangent_to_parabola t.B t.C parabola_M ∧
  concyclic t.A t.C t.B focus_M := by
  sorry

end parabola_properties_l2151_215165


namespace probability_two_of_each_color_l2151_215140

theorem probability_two_of_each_color (total_balls : ℕ) (black_balls : ℕ) (white_balls : ℕ) 
  (drawn_balls : ℕ) (h1 : total_balls = black_balls + white_balls) (h2 : total_balls = 17) 
  (h3 : black_balls = 9) (h4 : white_balls = 8) (h5 : drawn_balls = 4) : 
  (Nat.choose black_balls 2 * Nat.choose white_balls 2) / Nat.choose total_balls drawn_balls = 168 / 397 :=
sorry

end probability_two_of_each_color_l2151_215140


namespace equation_two_solutions_l2151_215139

theorem equation_two_solutions :
  ∃ (s : Finset ℝ), (∀ x ∈ s, Real.sqrt (9 - x) = x * Real.sqrt (9 - x)) ∧ s.card = 2 := by
  sorry

end equation_two_solutions_l2151_215139


namespace smallest_composite_no_small_factors_l2151_215106

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def has_no_prime_factors_less_than_20 (n : ℕ) : Prop :=
  ∀ p, p < 20 → ¬(Nat.Prime p ∧ p ∣ n)

theorem smallest_composite_no_small_factors :
  (is_composite 667 ∧ has_no_prime_factors_less_than_20 667) ∧
  (∀ m : ℕ, m < 667 → ¬(is_composite m ∧ has_no_prime_factors_less_than_20 m)) :=
sorry

end smallest_composite_no_small_factors_l2151_215106


namespace cube_less_than_triple_l2151_215176

theorem cube_less_than_triple : ∃! (x : ℤ), x^3 < 3*x :=
sorry

end cube_less_than_triple_l2151_215176


namespace sum_of_reciprocal_squares_l2151_215198

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - 9*x^2 + 8*x + 2

-- Define the roots
variable (p q r : ℝ)

-- State that p, q, and r are roots of f
axiom root_p : f p = 0
axiom root_q : f q = 0
axiom root_r : f r = 0

-- State that p, q, and r are distinct
axiom distinct_roots : p ≠ q ∧ q ≠ r ∧ p ≠ r

-- Theorem to prove
theorem sum_of_reciprocal_squares : 1/p^2 + 1/q^2 + 1/r^2 = 25 := by
  sorry

end sum_of_reciprocal_squares_l2151_215198


namespace frog_arrangement_count_l2151_215190

/-- Represents the color of a frog -/
inductive FrogColor
  | Green
  | Red
  | Blue

/-- Represents a row of frogs -/
def FrogRow := List FrogColor

/-- Checks if a frog arrangement is valid -/
def is_valid_arrangement (row : FrogRow) : Bool :=
  sorry

/-- Counts the number of frogs of each color in a row -/
def count_frogs (row : FrogRow) : Nat × Nat × Nat :=
  sorry

/-- Generates all possible arrangements of frogs -/
def all_arrangements : List FrogRow :=
  sorry

/-- Counts the number of valid arrangements -/
def count_valid_arrangements : Nat :=
  sorry

theorem frog_arrangement_count :
  count_valid_arrangements = 96 :=
sorry

end frog_arrangement_count_l2151_215190


namespace fruit_seller_problem_l2151_215112

/-- Represents the number of apples whose selling price equals the total gain -/
def reference_apples : ℕ := 50

/-- Represents the gain percent as a rational number -/
def gain_percent : ℚ := 100 / 3

/-- Calculates the number of apples sold given the reference apples and gain percent -/
def apples_sold (reference : ℕ) (gain : ℚ) : ℕ := sorry

theorem fruit_seller_problem :
  apples_sold reference_apples gain_percent = 200 := by sorry

end fruit_seller_problem_l2151_215112


namespace cos_x_plus_2y_equals_one_l2151_215168

theorem cos_x_plus_2y_equals_one 
  (x y a : ℝ) 
  (h1 : x ∈ Set.Icc (-π/4) (π/4))
  (h2 : y ∈ Set.Icc (-π/4) (π/4))
  (h3 : x^3 + Real.sin x - 2*a = 0)
  (h4 : 4*y^3 + Real.sin y * Real.cos y + a = 0) :
  Real.cos (x + 2*y) = 1 := by
sorry

end cos_x_plus_2y_equals_one_l2151_215168


namespace quadrilateral_exists_for_four_lines_l2151_215162

/-- A line in a plane --/
structure Line :=
  (a b c : ℝ)

/-- A point in a plane --/
structure Point :=
  (x y : ℝ)

/-- A region in a plane --/
structure Region :=
  (vertices : List Point)

/-- Checks if a region is a quadrilateral --/
def isQuadrilateral (r : Region) : Prop :=
  r.vertices.length = 4

/-- The set of all regions formed by the intersections of the given lines --/
def regionsFormedByLines (lines : List Line) : Set Region :=
  sorry

/-- The theorem stating that among the regions formed by 4 intersecting lines, 
    there exists at least one quadrilateral --/
theorem quadrilateral_exists_for_four_lines 
  (lines : List Line) 
  (h : lines.length = 4) : 
  ∃ r ∈ regionsFormedByLines lines, isQuadrilateral r :=
sorry

end quadrilateral_exists_for_four_lines_l2151_215162


namespace conference_handshakes_l2151_215171

/-- The number of people at the conference -/
def n : ℕ := 27

/-- The number of people who don't shake hands with each other -/
def k : ℕ := 3

/-- The maximum number of handshakes possible under the given conditions -/
def max_handshakes : ℕ := n.choose 2 - k.choose 2

/-- Theorem stating the maximum number of handshakes at the conference -/
theorem conference_handshakes :
  max_handshakes = 348 :=
by sorry

end conference_handshakes_l2151_215171


namespace hybrid_car_trip_length_l2151_215183

theorem hybrid_car_trip_length 
  (battery_distance : ℝ) 
  (gasoline_consumption_rate : ℝ) 
  (average_efficiency : ℝ) :
  battery_distance = 75 →
  gasoline_consumption_rate = 0.05 →
  average_efficiency = 50 →
  ∃ (total_distance : ℝ),
    total_distance = 125 ∧
    average_efficiency = total_distance / (gasoline_consumption_rate * (total_distance - battery_distance)) :=
by
  sorry

end hybrid_car_trip_length_l2151_215183


namespace john_mary_probability_l2151_215108

-- Define the set of people
inductive Person : Type
| John : Person
| Mary : Person
| Alice : Person
| Bob : Person
| Clara : Person

-- Define the seating arrangement
structure Seating :=
(long_side1 : Person × Person)
(long_side2 : Person × Person)
(short_side1 : Person)
(short_side2 : Person)

-- Define a function to check if John and Mary are seated together on a longer side
def john_and_mary_together (s : Seating) : Prop :=
  (s.long_side1 = (Person.John, Person.Mary) ∨ s.long_side1 = (Person.Mary, Person.John)) ∨
  (s.long_side2 = (Person.John, Person.Mary) ∨ s.long_side2 = (Person.Mary, Person.John))

-- Define the set of all possible seating arrangements
def all_seatings : Set Seating := sorry

-- Define the probability measure on the set of all seating arrangements
def prob : Set Seating → ℝ := sorry

-- The main theorem
theorem john_mary_probability :
  prob {s ∈ all_seatings | john_and_mary_together s} = 1/4 := by sorry

end john_mary_probability_l2151_215108


namespace police_hat_multiple_l2151_215142

/-- Proves that the multiple of Fire Chief Simpson's hats that Policeman O'Brien had before he lost one is 2 -/
theorem police_hat_multiple :
  let simpson_hats : ℕ := 15
  let obrien_current_hats : ℕ := 34
  let obrien_previous_hats : ℕ := obrien_current_hats + 1
  ∃ x : ℕ, x * simpson_hats + 5 = obrien_previous_hats ∧ x = 2 :=
by
  sorry

end police_hat_multiple_l2151_215142


namespace parade_formation_l2151_215175

theorem parade_formation (total : Nat) (red_flower : Nat) (red_balloon : Nat) (yellow_green : Nat)
  (h1 : total = 100)
  (h2 : red_flower = 42)
  (h3 : red_balloon = 63)
  (h4 : yellow_green = 28) :
  total - red_balloon - yellow_green + red_flower = 33 := by
  sorry

end parade_formation_l2151_215175


namespace spinster_cat_ratio_l2151_215177

theorem spinster_cat_ratio : 
  ∀ (spinsters cats : ℕ),
  spinsters = 18 →
  cats = spinsters + 63 →
  ∃ (n : ℕ), spinsters * n = 2 * cats →
  (spinsters : ℚ) / (cats : ℚ) = 2 / 9 := by
sorry

end spinster_cat_ratio_l2151_215177


namespace lowest_score_problem_l2151_215135

theorem lowest_score_problem (scores : List ℝ) (highest_score lowest_score : ℝ) : 
  scores.length = 15 →
  scores.sum / scores.length = 75 →
  highest_score ∈ scores →
  lowest_score ∈ scores →
  highest_score = 95 →
  (scores.sum - highest_score - lowest_score) / (scores.length - 2) = 78 →
  lowest_score = 16 :=
by sorry

end lowest_score_problem_l2151_215135


namespace modulus_of_z_l2151_215154

def z : ℂ := (2 + Complex.I) * (1 - Complex.I)

theorem modulus_of_z : Complex.abs z = Real.sqrt 10 := by
  sorry

end modulus_of_z_l2151_215154


namespace isosceles_triangle_angle_measure_l2151_215133

/-- In an isosceles triangle XYZ, where angle X is congruent to angle Z,
    and angle Z is five times angle Y, the measure of angle X is 900/11 degrees. -/
theorem isosceles_triangle_angle_measure (X Y Z : ℝ) : 
  X = Z →                   -- Angle X is congruent to angle Z
  Z = 5 * Y →               -- Angle Z is five times angle Y
  X + Y + Z = 180 →         -- Sum of angles in a triangle is 180 degrees
  X = 900 / 11 :=           -- Measure of angle X is 900/11 degrees
by sorry

end isosceles_triangle_angle_measure_l2151_215133


namespace r_exceeds_s_by_two_l2151_215103

theorem r_exceeds_s_by_two (x y r s : ℝ) : 
  3 * x + 2 * y = 16 →
  5 * x + 3 * y = 26 →
  r = x →
  s = y →
  r - s = 2 := by
sorry

end r_exceeds_s_by_two_l2151_215103


namespace quadratic_product_is_square_l2151_215111

/-- Given quadratic trinomials f and g satisfying the inequality condition,
    prove that their product is the square of some quadratic trinomial. -/
theorem quadratic_product_is_square
  (f g : ℝ → ℝ)
  (hf : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c)
  (hg : ∃ d e h : ℝ, ∀ x, g x = d * x^2 + e * x + h)
  (h_ineq : ∀ x, (deriv f x) * (deriv g x) ≥ |f x| + |g x|) :
  ∃ (k : ℝ) (p : ℝ → ℝ),
    (∃ a b c : ℝ, ∀ x, p x = a * x^2 + b * x + c) ∧
    (∀ x, f x * g x = k * (p x)^2) :=
sorry

end quadratic_product_is_square_l2151_215111


namespace no_valid_arrangement_l2151_215132

/-- Represents a card with a digit -/
structure Card where
  digit : Fin 10

/-- Represents an arrangement of cards -/
def Arrangement := List Card

/-- Checks if an arrangement satisfies the problem conditions -/
def satisfiesConditions (arr : Arrangement) : Prop :=
  ∀ i : Fin 10, ∃ pos1 pos2 : Nat,
    pos1 < pos2 ∧
    pos2 < arr.length ∧
    (arr.get ⟨pos1, by sorry⟩).digit = i ∧
    (arr.get ⟨pos2, by sorry⟩).digit = i ∧
    pos2 - pos1 - 1 = i.val

theorem no_valid_arrangement :
  ¬∃ (arr : Arrangement),
    arr.length = 20 ∧
    (∀ i : Fin 10, (arr.filter (λ c => c.digit = i)).length = 2) ∧
    satisfiesConditions arr := by
  sorry


end no_valid_arrangement_l2151_215132


namespace jerry_age_l2151_215185

/-- Given that Mickey's age is 5 years less than 200% of Jerry's age and Mickey is 19 years old,
    prove that Jerry is 12 years old. -/
theorem jerry_age (mickey_age jerry_age : ℕ) 
  (h1 : mickey_age = 2 * jerry_age - 5)
  (h2 : mickey_age = 19) : 
  jerry_age = 12 := by
  sorry

end jerry_age_l2151_215185


namespace half_power_inequality_l2151_215116

theorem half_power_inequality (a b : ℝ) (h : a > b) : (1/2 : ℝ)^a < (1/2 : ℝ)^b := by
  sorry

end half_power_inequality_l2151_215116


namespace pardee_road_length_is_12000_l2151_215179

/-- The length of Pardee Road in meters, given the conditions of the problem -/
def pardee_road_length : ℕ :=
  let telegraph_road_km : ℕ := 162
  let difference_km : ℕ := 150
  let meters_per_km : ℕ := 1000
  (telegraph_road_km - difference_km) * meters_per_km

theorem pardee_road_length_is_12000 : pardee_road_length = 12000 := by
  sorry

end pardee_road_length_is_12000_l2151_215179


namespace orange_boxes_l2151_215105

theorem orange_boxes (total_oranges : ℕ) (oranges_per_box : ℕ) (h1 : total_oranges = 2650) (h2 : oranges_per_box = 10) :
  total_oranges / oranges_per_box = 265 := by
  sorry

end orange_boxes_l2151_215105


namespace pizza_consumption_order_l2151_215197

def Alex : ℚ := 1/6
def Beth : ℚ := 2/5
def Cyril : ℚ := 1/3
def Dan : ℚ := 3/10
def Ella : ℚ := 1 - (Alex + Beth + Cyril + Dan)

def siblings : List ℚ := [Beth, Cyril, Dan, Alex, Ella]

theorem pizza_consumption_order : 
  List.Sorted (fun a b => a ≥ b) siblings ∧ 
  siblings = [Beth, Cyril, Dan, Alex, Ella] :=
sorry

end pizza_consumption_order_l2151_215197


namespace c_death_year_l2151_215131

structure Mathematician where
  name : String
  birth_year : ℕ
  death_year : ℕ

def arithmetic_sequence (a b c : ℕ) : Prop :=
  b - a = c - b

theorem c_death_year (a b c : Mathematician) (d : String) :
  a.name = "A" →
  b.name = "B" →
  c.name = "C" →
  d = "D" →
  a.death_year = 1980 →
  a.death_year - a.birth_year = 50 →
  b.death_year - b.birth_year < 50 →
  c.death_year - c.birth_year = 60 →
  a.death_year - b.death_year < 10 →
  a.death_year - b.death_year > 0 →
  b.death_year - b.birth_year = c.death_year - b.death_year →
  arithmetic_sequence a.birth_year b.birth_year c.birth_year →
  c.death_year = 1986 := by
  sorry

#check c_death_year

end c_death_year_l2151_215131


namespace min_tangent_length_l2151_215155

/-- Circle C with equation x^2 + y^2 - 2x - 4y + 1 = 0 -/
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 2*p.1 - 4*p.2 + 1 = 0}

/-- Line l -/
def line_l : Set (ℝ × ℝ) := sorry

/-- Maximum distance from any point on C to line l is 6 -/
axiom max_distance_to_l (p : ℝ × ℝ) :
  p ∈ circle_C → ∃ (q : ℝ × ℝ), q ∈ line_l ∧ dist p q ≤ 6

/-- Tangent line from a point on l to C -/
def tangent_length (a : ℝ × ℝ) : ℝ := sorry

theorem min_tangent_length :
  ∃ (a : ℝ × ℝ), a ∈ line_l ∧
  (∀ (b : ℝ × ℝ), b ∈ line_l → tangent_length a ≤ tangent_length b) ∧
  tangent_length a = 2 * Real.sqrt 3 := by
  sorry

end min_tangent_length_l2151_215155


namespace range_of_z_l2151_215144

theorem range_of_z (x y : ℝ) (h : x^2 + 2*x*y + 4*y^2 = 6) : 
  ∃ z, z = x + y ∧ -Real.sqrt 6 ≤ z ∧ z ≤ Real.sqrt 6 :=
by sorry

end range_of_z_l2151_215144


namespace new_gross_profit_percentage_l2151_215159

theorem new_gross_profit_percentage
  (old_selling_price : ℝ)
  (old_gross_profit_percentage : ℝ)
  (new_selling_price : ℝ)
  (h1 : old_selling_price = 88)
  (h2 : old_gross_profit_percentage = 10)
  (h3 : new_selling_price = 92) :
  let cost := old_selling_price / (1 + old_gross_profit_percentage / 100)
  let new_gross_profit := new_selling_price - cost
  new_gross_profit / cost * 100 = 15 := by
sorry

end new_gross_profit_percentage_l2151_215159


namespace unique_a_for_perpendicular_chords_l2151_215192

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

-- Define a point on the x-axis
def point_on_x_axis (a : ℝ) : ℝ × ℝ := (a, 0)

-- Define the property of perpendicular lines intersecting the hyperbola
def perpendicular_lines_property (a : ℝ) : Prop :=
  ∀ (l₁ l₂ : ℝ → ℝ → Prop),
    (∀ x y, l₁ x y ↔ l₂ y (-x)) →  -- l₁ and l₂ are perpendicular
    (l₁ a 0 ∧ l₂ a 0) →  -- both lines pass through (a, 0)
    ∀ (px py qx qy rx ry sx sy : ℝ),
      (hyperbola px py ∧ hyperbola qx qy ∧ l₁ px py ∧ l₁ qx qy) →  -- P and Q on l₁ and hyperbola
      (hyperbola rx ry ∧ hyperbola sx sy ∧ l₂ rx ry ∧ l₂ sx sy) →  -- R and S on l₂ and hyperbola
      (px - qx)^2 + (py - qy)^2 = (rx - sx)^2 + (ry - sy)^2  -- |PQ| = |RS|

-- The main theorem
theorem unique_a_for_perpendicular_chords :
  ∃! (a : ℝ), a > 1 ∧ perpendicular_lines_property a ∧ a = Real.sqrt 2 :=
sorry

end unique_a_for_perpendicular_chords_l2151_215192


namespace parabola_focus_coordinates_l2151_215101

/-- The focus of the parabola y = 2x^2 has coordinates (0, 1/8) -/
theorem parabola_focus_coordinates :
  let f : ℝ → ℝ := λ x => 2 * x^2
  ∃ (focus : ℝ × ℝ), focus = (0, 1/8) ∧
    ∀ (x y : ℝ), y = f x → 
      (x - focus.1)^2 + (y - focus.2)^2 = (y - focus.2 + 1/4)^2 :=
by sorry

end parabola_focus_coordinates_l2151_215101


namespace complex_number_quadrant_l2151_215100

theorem complex_number_quadrant : ∃ (z : ℂ), z = (I : ℂ) / (Real.sqrt 3 - 3 * I) ∧ z.re < 0 ∧ z.im > 0 := by
  sorry

end complex_number_quadrant_l2151_215100


namespace quadratic_sum_of_d_and_e_l2151_215134

/-- Given a quadratic polynomial x^2 - 16x + 15, when written in the form (x+d)^2 + e,
    the sum of d and e is -57. -/
theorem quadratic_sum_of_d_and_e : ∃ d e : ℝ, 
  (∀ x, x^2 - 16*x + 15 = (x+d)^2 + e) ∧ d + e = -57 := by
  sorry

end quadratic_sum_of_d_and_e_l2151_215134


namespace complement_intersect_theorem_l2151_215189

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2, 3}
def B : Set Nat := {2, 5}

theorem complement_intersect_theorem :
  (U \ B) ∩ A = {1, 3} := by sorry

end complement_intersect_theorem_l2151_215189


namespace flu_infection_rate_l2151_215164

theorem flu_infection_rate : ∃ (x : ℝ), 
  (x > 0) ∧ 
  (1 + x + x * (1 + x) = 196) ∧ 
  (x = 13) :=
sorry

end flu_infection_rate_l2151_215164


namespace integer_divisible_by_15_with_sqrt_between_30_and_30_5_l2151_215178

theorem integer_divisible_by_15_with_sqrt_between_30_and_30_5 :
  ∃ (n : ℕ), n > 0 ∧ 15 ∣ n ∧ 30 < Real.sqrt n ∧ Real.sqrt n < 30.5 ∧
  (n = 900 ∨ n = 915 ∨ n = 930) := by
sorry

end integer_divisible_by_15_with_sqrt_between_30_and_30_5_l2151_215178


namespace sculpture_base_height_l2151_215138

/-- Converts feet and inches to total inches -/
def feet_inches_to_inches (feet : ℕ) (inches : ℕ) : ℕ :=
  12 * feet + inches

/-- Calculates the total height of a sculpture and its base -/
theorem sculpture_base_height 
  (sculpture_feet : ℕ) 
  (sculpture_inches : ℕ) 
  (base_inches : ℕ) : 
  feet_inches_to_inches sculpture_feet sculpture_inches + base_inches = 38 :=
by
  sorry

#check sculpture_base_height 2 10 4

end sculpture_base_height_l2151_215138


namespace power_sum_zero_l2151_215188

theorem power_sum_zero : (-2 : ℤ) ^ (3^2) + 2 ^ (3^2) = 0 := by
  sorry

end power_sum_zero_l2151_215188


namespace calculation_proof_l2151_215113

theorem calculation_proof : (-1)^2024 - 1/2 * (8 - (-2)^2) = -1 := by
  sorry

end calculation_proof_l2151_215113


namespace probability_black_white_balls_l2151_215120

theorem probability_black_white_balls (total_balls : ℕ) (black_balls : ℕ) (white_balls : ℕ) (green_balls : ℕ) :
  total_balls = black_balls + white_balls + green_balls →
  black_balls = 3 →
  white_balls = 3 →
  green_balls = 1 →
  (black_balls * white_balls : ℚ) / ((total_balls * (total_balls - 1)) / 2) = 3 / 7 := by
  sorry

end probability_black_white_balls_l2151_215120


namespace half_angle_quadrant_l2151_215174

theorem half_angle_quadrant (α : Real) : 
  (∃ k : ℤ, 2 * k * π + π / 2 < α ∧ α < 2 * k * π + π) →
  (∃ m : ℤ, m * π < α / 2 ∧ α / 2 < m * π + π / 2) :=
sorry

end half_angle_quadrant_l2151_215174


namespace fruit_box_composition_l2151_215163

/-- Represents the contents of the fruit box -/
structure FruitBox where
  apples : ℕ
  pears : ℕ

/-- The total number of fruits in the box -/
def FruitBox.total (box : FruitBox) : ℕ := box.apples + box.pears

/-- Predicate to check if selecting n fruits always includes at least one apple -/
def always_includes_apple (box : FruitBox) (n : ℕ) : Prop :=
  box.pears < n

/-- Predicate to check if selecting n fruits always includes at least one pear -/
def always_includes_pear (box : FruitBox) (n : ℕ) : Prop :=
  box.apples < n

/-- The main theorem stating the unique composition of the fruit box -/
theorem fruit_box_composition :
  ∃! (box : FruitBox),
    box.total ≥ 5 ∧
    always_includes_apple box 3 ∧
    always_includes_pear box 4 :=
  sorry

end fruit_box_composition_l2151_215163


namespace friday_temperature_l2151_215123

/-- Temperatures for Tuesday, Wednesday, Thursday, and Friday --/
structure WeekTemperatures where
  tuesday : ℝ
  wednesday : ℝ
  thursday : ℝ
  friday : ℝ

/-- Theorem stating that Friday's temperature is 53°C given the conditions --/
theorem friday_temperature (t : WeekTemperatures) : t.friday = 53 :=
  by
  have h1 : (t.tuesday + t.wednesday + t.thursday) / 3 = 45 := by sorry
  have h2 : (t.wednesday + t.thursday + t.friday) / 3 = 50 := by sorry
  have h3 : t.tuesday = 38 := by sorry
  have h4 : t.tuesday = 38 ∨ t.wednesday = 53 ∨ t.thursday = 53 ∨ t.friday = 53 := by sorry
  sorry

end friday_temperature_l2151_215123


namespace puzzle_solution_l2151_215182

theorem puzzle_solution (a b : ℕ) 
  (sum_eq : a + b = 24581)
  (b_div_12 : ∃ k : ℕ, b = 12 * k)
  (a_times_10 : a * 10 = b) :
  b - a = 20801 := by
sorry

end puzzle_solution_l2151_215182


namespace arithmetic_sequence_l2151_215196

/-- Given real numbers x, y, and z satisfying the equation (z-x)^2 - 4(x-y)(y-z) = 0,
    prove that 2y = x + z, which implies that x, y, and z form an arithmetic sequence. -/
theorem arithmetic_sequence (x y z : ℝ) (h : (z - x)^2 - 4*(x - y)*(y - z) = 0) :
  2*y = x + z := by
  sorry

end arithmetic_sequence_l2151_215196


namespace lyndees_chicken_pieces_l2151_215130

/-- Given the total number of chicken pieces, the number of friends, and the number of pieces each friend ate,
    calculate the number of pieces Lyndee ate. -/
theorem lyndees_chicken_pieces (total_pieces friends_pieces friends : ℕ) : 
  total_pieces - (friends_pieces * friends) = total_pieces - (friends_pieces * friends) := by
  sorry

#check lyndees_chicken_pieces 11 2 5

end lyndees_chicken_pieces_l2151_215130


namespace first_store_unload_percentage_l2151_215143

def initial_load : ℝ := 50000
def second_store_percentage : ℝ := 0.20
def final_load : ℝ := 36000

theorem first_store_unload_percentage :
  ∃ x : ℝ, 
    x ≥ 0 ∧ x ≤ 1 ∧
    (1 - x) * initial_load * (1 - second_store_percentage) = final_load ∧
    x = 0.1 := by
  sorry

end first_store_unload_percentage_l2151_215143


namespace estate_distribution_l2151_215147

theorem estate_distribution (a b c d : ℝ) : 
  a > 0 ∧ 
  b = 1.20 * a ∧ 
  c = 1.20 * b ∧ 
  d = 1.20 * c ∧ 
  d - a = 19520 →
  (b = 32176 ∨ c = 32176 ∨ d = 32176) := by
sorry

end estate_distribution_l2151_215147


namespace car_rate_problem_l2151_215149

/-- Given two cars starting at the same time and point, with one car traveling at 60 mph,
    if after 3 hours the distance between them is 30 miles,
    then the rate of the other car is 50 mph. -/
theorem car_rate_problem (rate1 : ℝ) : 
  (60 * 3 = rate1 * 3 + 30) → rate1 = 50 := by
  sorry

end car_rate_problem_l2151_215149


namespace video_game_expenditure_l2151_215137

theorem video_game_expenditure (total : ℝ) (books snacks stationery shoes : ℝ) :
  total = 50 →
  books = (1 / 4) * total →
  snacks = (1 / 5) * total →
  stationery = (1 / 10) * total →
  shoes = (3 / 10) * total →
  total - (books + snacks + stationery + shoes) = 7.5 := by
  sorry

end video_game_expenditure_l2151_215137


namespace unique_solution_implies_k_equals_one_l2151_215124

/-- The set of real solutions to the quadratic equation kx^2 + 4x + 4 = 0 -/
def A (k : ℝ) : Set ℝ := {x : ℝ | k * x^2 + 4 * x + 4 = 0}

/-- Theorem: If the set A contains only one element, then k = 1 -/
theorem unique_solution_implies_k_equals_one (k : ℝ) : (∃! x, x ∈ A k) → k = 1 := by
  sorry

end unique_solution_implies_k_equals_one_l2151_215124


namespace one_plus_i_fourth_power_l2151_215170

/-- The imaginary unit i -/
noncomputable def i : ℂ := Complex.I

/-- The fourth power of (1 + i) equals -4 -/
theorem one_plus_i_fourth_power : (1 + i)^4 = -4 := by sorry

end one_plus_i_fourth_power_l2151_215170


namespace intersection_of_A_and_B_l2151_215172

def A : Set ℝ := {x | x + 1 > 0}
def B : Set ℝ := {x | x < 0}

theorem intersection_of_A_and_B : A ∩ B = {x | -1 < x ∧ x < 0} := by sorry

end intersection_of_A_and_B_l2151_215172


namespace inequality_solution_set_l2151_215141

theorem inequality_solution_set (x : ℝ) :
  (3 ≤ x / (2 * x - 5) ∧ x / (2 * x - 5) < 8) ↔ (8/3 < x ∧ x ≤ 3) :=
by sorry

end inequality_solution_set_l2151_215141


namespace parallel_lines_d_value_l2151_215136

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The value of d for which the lines y = 3x + 5 and y = (4d)x + 3 are parallel -/
theorem parallel_lines_d_value :
  (∀ x y : ℝ, y = 3 * x + 5 ↔ y = (4 * d) * x + 3) → d = 3 / 4 :=
by sorry

end parallel_lines_d_value_l2151_215136


namespace soccer_league_games_l2151_215158

theorem soccer_league_games (n : ℕ) (h : n = 25) : n * (n - 1) / 2 = 300 := by
  sorry

end soccer_league_games_l2151_215158


namespace sqrt_three_irrational_l2151_215194

theorem sqrt_three_irrational : Irrational (Real.sqrt 3) := by
  sorry

end sqrt_three_irrational_l2151_215194


namespace removed_digit_not_power_of_two_l2151_215161

def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

def remove_middle_digit (n : ℕ) : ℕ := 
  -- Implementation details omitted
  sorry

theorem removed_digit_not_power_of_two (N : ℕ) (h : is_power_of_two N) :
  ¬ is_power_of_two (remove_middle_digit N) := by
  sorry

end removed_digit_not_power_of_two_l2151_215161


namespace existence_of_x_with_abs_f_ge_2_l2151_215117

theorem existence_of_x_with_abs_f_ge_2 (a b : ℝ) :
  ∃ x₀ ∈ Set.Icc 1 9, |a * x₀ + b + 9 / x₀| ≥ 2 := by
  sorry

end existence_of_x_with_abs_f_ge_2_l2151_215117


namespace tv_sales_decrease_l2151_215184

theorem tv_sales_decrease (original_price original_quantity : ℝ) 
  (original_price_positive : original_price > 0)
  (original_quantity_positive : original_quantity > 0) :
  let new_price := 1.30 * original_price
  let new_revenue := 1.04 * (original_price * original_quantity)
  let sales_decrease_percentage := 
    100 * (1 - (new_revenue / new_price) / original_quantity)
  sales_decrease_percentage = 20 := by
sorry

end tv_sales_decrease_l2151_215184


namespace fly_path_distance_l2151_215125

theorem fly_path_distance (r : ℝ) (s : ℝ) (h1 : r = 58) (h2 : s = 80) : 
  let d := 2 * r
  let x := Real.sqrt (d^2 - s^2)
  d + x + s = 280 :=
by sorry

end fly_path_distance_l2151_215125


namespace regular_polygon_sides_l2151_215173

theorem regular_polygon_sides (n : ℕ) (h : n > 2) :
  (∀ θ : ℝ, θ = 160 * π / 180 → (n - 2) * π = n * θ) → n = 18 := by
  sorry

end regular_polygon_sides_l2151_215173


namespace competition_probabilities_l2151_215110

-- Define the possible grades
inductive Grade : Type
  | Qualified
  | Good
  | Excellent

-- Define the probabilities for each participant
def probA : Grade → ℝ
  | Grade.Qualified => 0.6
  | Grade.Good => 0.3
  | Grade.Excellent => 0.1

def probB : Grade → ℝ
  | Grade.Qualified => 0.4
  | Grade.Good => 0.4
  | Grade.Excellent => 0.2

-- Define a function to check if one grade is higher than another
def isHigher : Grade → Grade → Bool
  | Grade.Excellent, Grade.Excellent => false
  | Grade.Excellent, _ => true
  | Grade.Good, Grade.Excellent => false
  | Grade.Good, _ => true
  | Grade.Qualified, Grade.Qualified => false
  | Grade.Qualified, _ => false

-- Define the probability that A's grade is higher than B's in one round
def probAHigherThanB : ℝ := 0.2

-- Define the probability that A's grade is higher than B's in at least two out of three rounds
def probAHigherThanBTwiceInThree : ℝ := 0.104

theorem competition_probabilities :
  (probAHigherThanB = 0.2) ∧
  (probAHigherThanBTwiceInThree = 0.104) := by
  sorry


end competition_probabilities_l2151_215110


namespace cole_trip_time_l2151_215107

/-- Proves that given a round trip where the outbound journey is at 75 km/h,
    the return journey is at 105 km/h, and the total trip time is 4 hours,
    the time taken for the outbound journey is 140 minutes. -/
theorem cole_trip_time (distance : ℝ) :
  distance / 75 + distance / 105 = 4 →
  distance / 75 * 60 = 140 := by
sorry

end cole_trip_time_l2151_215107


namespace s_five_value_l2151_215167

theorem s_five_value (x : ℝ) (h : x + 1/x = 4) : x^5 + 1/x^5 = 724 := by
  sorry

end s_five_value_l2151_215167


namespace xyz_sum_product_bounds_l2151_215146

theorem xyz_sum_product_bounds (x y z : ℝ) 
  (h : 3 * (x + y + z) = x^2 + y^2 + z^2) : 
  let f := x*y + x*z + y*z
  ∃ (M m : ℝ), 
    (∀ a b c : ℝ, 3*(a + b + c) = a^2 + b^2 + c^2 → a*b + a*c + b*c ≤ M) ∧
    (∀ a b c : ℝ, 3*(a + b + c) = a^2 + b^2 + c^2 → m ≤ a*b + a*c + b*c) ∧
    f ≤ M ∧ 
    m ≤ f ∧
    M = 27 ∧ 
    m = -9/8 ∧ 
    M + 5*m = 126/8 :=
by
  sorry

end xyz_sum_product_bounds_l2151_215146


namespace f_expression_m_values_l2151_215109

/-- A quadratic function satisfying certain properties -/
def f (x : ℝ) : ℝ := sorry

/-- The properties of the quadratic function -/
axiom f_diff (x : ℝ) : f (x + 1) - f x = 2 * x - 1
axiom f_zero : f 0 = 3

/-- The expression of f(x) -/
theorem f_expression (x : ℝ) : f x = x^2 - 2*x + 3 := sorry

/-- The function y in terms of x and m -/
def y (x m : ℝ) : ℝ := f (Real.log x / Real.log 3 + m)

/-- The set of x values -/
def X : Set ℝ := Set.Icc (1/3) 3

/-- The theorem about the values of m -/
theorem m_values :
  ∀ m : ℝ, (∀ x ∈ X, y x m ≥ 3) ∧ (∃ x ∈ X, y x m = 3) →
  m = -1 ∨ m = 3 := sorry

end f_expression_m_values_l2151_215109


namespace increase_by_percentage_l2151_215122

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (final : ℝ) :
  initial = 110 → percentage = 50 → final = initial * (1 + percentage / 100) → final = 165 := by
  sorry

end increase_by_percentage_l2151_215122


namespace nonagon_diagonals_l2151_215145

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

/-- Theorem: A regular nine-sided polygon (nonagon) contains 27 diagonals -/
theorem nonagon_diagonals :
  num_diagonals 9 = 27 := by
  sorry

end nonagon_diagonals_l2151_215145


namespace skipping_odometer_conversion_l2151_215153

/-- Represents an odometer that skips digits 3 and 4 --/
def SkippingOdometer := Nat → Nat

/-- Converts a regular number to its representation on the skipping odometer --/
def toSkippingOdometer : Nat → Nat :=
  sorry

/-- Converts a number from the skipping odometer to its actual value --/
def fromSkippingOdometer : Nat → Nat :=
  sorry

theorem skipping_odometer_conversion :
  ∃ (odo : SkippingOdometer),
    (toSkippingOdometer 1029 = 002006) ∧
    (fromSkippingOdometer 002006 = 1029) := by
  sorry

end skipping_odometer_conversion_l2151_215153


namespace largest_negative_angle_solution_l2151_215121

theorem largest_negative_angle_solution :
  let θ : ℝ := -π/2
  let eq (x : ℝ) := (1 - Real.sin x + Real.cos x) / (1 - Real.sin x - Real.cos x) +
                    (1 - Real.sin x - Real.cos x) / (1 - Real.sin x + Real.cos x) = 2
  (eq θ) ∧ 
  (∀ φ, φ < 0 → φ > θ → ¬(eq φ)) :=
by sorry

end largest_negative_angle_solution_l2151_215121


namespace prob_through_C_eq_25_63_l2151_215156

/-- Represents a point in the city grid -/
structure Point where
  x : ℕ
  y : ℕ

/-- The probability of choosing either direction at an intersection -/
def choice_prob : ℚ := 1/2

/-- The starting point A -/
def A : Point := ⟨0, 0⟩

/-- The intermediate point C -/
def C : Point := ⟨3, 2⟩

/-- The ending point B -/
def B : Point := ⟨5, 5⟩

/-- Calculates the number of paths between two points -/
def num_paths (start finish : Point) : ℕ :=
  Nat.choose (finish.x - start.x + finish.y - start.y) (finish.x - start.x)

/-- The probability of walking from A to B through C -/
def prob_through_C : ℚ :=
  (num_paths A C * num_paths C B : ℚ) / num_paths A B

theorem prob_through_C_eq_25_63 : prob_through_C = 25/63 := by
  sorry

end prob_through_C_eq_25_63_l2151_215156


namespace fraction_to_decimal_l2151_215129

theorem fraction_to_decimal : (5 : ℚ) / 125 = (4 : ℚ) / 100 := by
  sorry

end fraction_to_decimal_l2151_215129


namespace pentomino_tiling_l2151_215199

/-- A pentomino is a shape that covers exactly 5 squares. -/
def Pentomino : Type := Unit

/-- A rectangle of size 5 × m. -/
structure Rectangle (m : ℕ) :=
  (width : Fin 5)
  (height : Fin m)

/-- Predicate to determine if a rectangle can be tiled by a pentomino. -/
def IsTileable (m : ℕ) : Prop := sorry

theorem pentomino_tiling (m : ℕ) : 
  IsTileable m ↔ Even m := by sorry

end pentomino_tiling_l2151_215199


namespace two_real_roots_iff_nonneg_discriminant_quadratic_always_two_real_roots_l2151_215195

/-- The discriminant of a quadratic equation ax^2 + bx + c = 0 is b^2 - 4ac -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- A quadratic equation has two real roots if and only if its discriminant is non-negative -/
theorem two_real_roots_iff_nonneg_discriminant (a b c : ℝ) (ha : a ≠ 0) :
  ∃ x y : ℝ, a*x^2 + b*x + c = 0 ∧ a*y^2 + b*y + c = 0 ∧ x ≠ y ↔ discriminant a b c ≥ 0 :=
sorry

theorem quadratic_always_two_real_roots (k : ℝ) :
  discriminant 1 (-(k+4)) (4*k) ≥ 0 :=
sorry

end two_real_roots_iff_nonneg_discriminant_quadratic_always_two_real_roots_l2151_215195


namespace arithmetic_sequence_of_powers_no_infinite_arithmetic_sequence_of_powers_l2151_215115

/-- For any positive integer n, there exists an arithmetic sequence of n different elements 
    where every term is a power of a positive integer greater than 1. -/
theorem arithmetic_sequence_of_powers (n : ℕ+) : 
  ∃ (a : ℕ → ℕ) (d : ℕ), 
    (∀ i j, i ≠ j → a i ≠ a j) ∧ 
    (∀ i, ∃ (b c : ℕ), c > 1 ∧ a i = b^c) ∧
    (∀ i, a (i + 1) = a i + d) :=
sorry

/-- There does not exist an infinite arithmetic sequence where every term is a power 
    of a positive integer greater than 1. -/
theorem no_infinite_arithmetic_sequence_of_powers : 
  ¬∃ (a : ℕ → ℕ) (d : ℕ), 
    (∀ i j, i ≠ j → a i ≠ a j) ∧ 
    (∀ i, ∃ (b c : ℕ), c > 1 ∧ a i = b^c) ∧
    (∀ i, a (i + 1) = a i + d) :=
sorry

end arithmetic_sequence_of_powers_no_infinite_arithmetic_sequence_of_powers_l2151_215115


namespace conic_sections_from_equation_l2151_215169

/-- The equation y^4 - 8x^4 = 4y^2 - 4 represents two conic sections -/
theorem conic_sections_from_equation :
  ∃ (f g : ℝ → ℝ → Prop),
    (∀ x y, y^4 - 8*x^4 = 4*y^2 - 4 ↔ f x y ∨ g x y) ∧
    (∃ a b c d e : ℝ, ∀ x y, f x y ↔ (x^2 / a^2) - (y^2 / b^2) = 1) ∧
    (∃ a b c d e : ℝ, ∀ x y, g x y ↔ (x^2 / c^2) + (y^2 / d^2) = 1) :=
sorry

end conic_sections_from_equation_l2151_215169


namespace apple_cost_calculation_l2151_215150

/-- The cost of apples in dollars per 3 pounds -/
def apple_cost_per_3_pounds : ℚ := 3

/-- The weight of apples in pounds that we want to calculate the cost for -/
def apple_weight : ℚ := 18

/-- Theorem stating that the cost of 18 pounds of apples is 18 dollars -/
theorem apple_cost_calculation : 
  (apple_weight / 3) * apple_cost_per_3_pounds = 18 := by
  sorry


end apple_cost_calculation_l2151_215150


namespace cubic_sum_equals_linear_sum_l2151_215151

theorem cubic_sum_equals_linear_sum (k : ℝ) : 
  (∀ r s : ℝ, 3 * r^2 + 6 * r + k = 0 ∧ 3 * s^2 + 6 * s + k = 0 → r^3 + s^3 = r + s) ↔ 
  k = 3 := by
sorry

end cubic_sum_equals_linear_sum_l2151_215151


namespace floor_sqrt_50_squared_l2151_215186

theorem floor_sqrt_50_squared : ⌊Real.sqrt 50⌋^2 = 49 := by sorry

end floor_sqrt_50_squared_l2151_215186


namespace parallel_lines_circle_chord_l2151_215119

theorem parallel_lines_circle_chord (r : ℝ) (d : ℝ) : 
  r > 0 ∧ d > 0 ∧ 
  36 * r^2 = 648 + 9 * d^2 ∧ 
  40 * r^2 = 800 + 90 * d^2 → 
  d = 67/10 := by
sorry

end parallel_lines_circle_chord_l2151_215119


namespace total_jars_is_72_l2151_215160

/-- Represents the number of jars of each size -/
def num_jars : ℕ := 24

/-- Represents the total volume of water in gallons -/
def total_volume : ℚ := 42

/-- Represents the volume of a quart jar in gallons -/
def quart_volume : ℚ := 1/4

/-- Represents the volume of a half-gallon jar in gallons -/
def half_gallon_volume : ℚ := 1/2

/-- Represents the volume of a one-gallon jar in gallons -/
def gallon_volume : ℚ := 1

/-- The theorem stating that given the conditions, the total number of jars is 72 -/
theorem total_jars_is_72 :
  (num_jars : ℚ) * (quart_volume + half_gallon_volume + gallon_volume) = total_volume ∧
  num_jars * 3 = 72 := by
  sorry

#check total_jars_is_72

end total_jars_is_72_l2151_215160


namespace pta_fundraiser_remaining_money_l2151_215118

theorem pta_fundraiser_remaining_money (initial_amount : ℝ) : 
  initial_amount = 400 → 
  (initial_amount - initial_amount / 4) / 2 = 150 :=
by
  sorry

end pta_fundraiser_remaining_money_l2151_215118


namespace emily_egg_collection_l2151_215181

theorem emily_egg_collection (baskets : ℕ) (eggs_per_basket : ℕ) 
  (h1 : baskets = 303) (h2 : eggs_per_basket = 28) : 
  baskets * eggs_per_basket = 8484 := by
  sorry

end emily_egg_collection_l2151_215181


namespace product_of_roots_plus_one_l2151_215166

theorem product_of_roots_plus_one (a b c : ℂ) : 
  (a^3 - 15*a^2 + 22*a - 8 = 0) → 
  (b^3 - 15*b^2 + 22*b - 8 = 0) → 
  (c^3 - 15*c^2 + 22*c - 8 = 0) → 
  (1 + a) * (1 + b) * (1 + c) = 46 := by
  sorry

end product_of_roots_plus_one_l2151_215166


namespace repeating_decimal_to_fraction_l2151_215102

/-- The repeating decimal 0.̅5̅6̅ is equal to the fraction 56/99 -/
theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (∀ (n : ℕ), x * 10^(2*n+2) - ⌊x * 10^(2*n+2)⌋ = 0.56) ∧ x = 56/99 := by
  sorry

end repeating_decimal_to_fraction_l2151_215102


namespace frank_lawn_money_l2151_215104

/-- The amount of money Frank made mowing lawns -/
def lawn_money : ℕ := 19

/-- The cost of mower blades -/
def blade_cost : ℕ := 11

/-- The number of games Frank could buy -/
def num_games : ℕ := 4

/-- The cost of each game -/
def game_cost : ℕ := 2

/-- Theorem stating that the money Frank made mowing lawns is correct -/
theorem frank_lawn_money :
  lawn_money = blade_cost + num_games * game_cost :=
by sorry

end frank_lawn_money_l2151_215104


namespace marys_marbles_count_l2151_215191

def dans_marbles : ℕ := 5
def marys_marbles_multiplier : ℕ := 2

theorem marys_marbles_count : dans_marbles * marys_marbles_multiplier = 10 := by
  sorry

end marys_marbles_count_l2151_215191


namespace fundraising_ratio_l2151_215148

-- Define the fundraising goal
def goal : ℕ := 4000

-- Define Ken's collection
def ken_collection : ℕ := 600

-- Define the amount they exceeded the goal by
def excess : ℕ := 600

-- Define the total amount collected
def total_collected : ℕ := goal + excess

-- Define Mary's collection as a function of Ken's
def mary_collection (x : ℚ) : ℚ := x * ken_collection

-- Define Scott's collection as a function of Mary's
def scott_collection (x : ℚ) : ℚ := (1 / 3) * mary_collection x

-- State the theorem
theorem fundraising_ratio : 
  ∃ x : ℚ, 
    scott_collection x + mary_collection x + ken_collection = total_collected ∧ 
    mary_collection x / ken_collection = 5 := by
  sorry

end fundraising_ratio_l2151_215148


namespace wire_length_proof_l2151_215127

theorem wire_length_proof (shorter_piece longer_piece total_length : ℝ) : 
  shorter_piece = 20 →
  shorter_piece = (2 / 5) * longer_piece →
  total_length = shorter_piece + longer_piece →
  total_length = 70 := by
sorry

end wire_length_proof_l2151_215127


namespace problem_1_problem_2_l2151_215180

-- Problem 1
theorem problem_1 : (3 + Real.sqrt 5) * (Real.sqrt 5 - 2) = Real.sqrt 5 - 1 := by
  sorry

-- Problem 2
theorem problem_2 : (Real.sqrt 12 + Real.sqrt 27) / Real.sqrt 3 = 5 := by
  sorry

end problem_1_problem_2_l2151_215180


namespace boat_speed_ratio_l2151_215157

theorem boat_speed_ratio (boat_speed : ℝ) (current_speed : ℝ) (distance : ℝ) 
  (h1 : boat_speed = 20)
  (h2 : current_speed = 4)
  (h3 : distance = 2) : 
  (2 * distance) / ((distance / (boat_speed + current_speed)) + (distance / (boat_speed - current_speed))) / boat_speed = 24 / 25 := by
  sorry

end boat_speed_ratio_l2151_215157


namespace arithmetic_sequence_property_l2151_215187

/-- An arithmetic sequence -/
def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : arithmeticSequence a)
  (h_sum : a 4 + a 6 + a 8 + a 10 + a 12 = 120) :
  2 * a 10 - a 12 = 24 := by
  sorry

end arithmetic_sequence_property_l2151_215187


namespace arithmetic_sequence_common_difference_l2151_215128

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) (d : ℝ) 
  (h1 : a 1 + a 5 = 10) 
  (h2 : a 4 = 7) 
  (h3 : arithmetic_sequence a d) : 
  d = 2 := by
sorry

end arithmetic_sequence_common_difference_l2151_215128


namespace power_of_power_l2151_215152

theorem power_of_power (x : ℝ) : (x^2)^3 = x^6 := by
  sorry

end power_of_power_l2151_215152


namespace cos_330_degrees_l2151_215193

theorem cos_330_degrees : Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_330_degrees_l2151_215193
