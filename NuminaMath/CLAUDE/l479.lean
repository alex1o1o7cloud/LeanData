import Mathlib

namespace percentage_of_difference_l479_47901

theorem percentage_of_difference (x y : ℝ) (P : ℝ) :
  P / 100 * (x - y) = 15 / 100 * (x + y) →
  y = 14.285714285714285 / 100 * x →
  P = 16 := by
  sorry

end percentage_of_difference_l479_47901


namespace equation1_solutions_equation2_solution_l479_47951

-- Define the equations
def equation1 (x : ℝ) : Prop := 3 * (x + 1)^2 - 2 = 25
def equation2 (x : ℝ) : Prop := (x - 1)^3 = 64

-- Theorem for equation1
theorem equation1_solutions :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation1 x₁ ∧ equation1 x₂ ∧ x₁ = 2 ∧ x₂ = -4 :=
sorry

-- Theorem for equation2
theorem equation2_solution :
  ∃ x : ℝ, equation2 x ∧ x = 5 :=
sorry

end equation1_solutions_equation2_solution_l479_47951


namespace only_C_is_random_event_l479_47902

-- Define the structure for an event
structure Event where
  description : String
  is_possible : Bool
  is_certain : Bool

-- Define the events
def event_A : Event := ⟨"Scoring 105 points in a percentile-based exam", false, false⟩
def event_B : Event := ⟨"Area of a rectangle with sides a and b is ab", true, true⟩
def event_C : Event := ⟨"Taking out 2 parts from 100 parts (2 defective, 98 non-defective), both are defective", true, false⟩
def event_D : Event := ⟨"Tossing a coin, it lands with either heads or tails up", true, true⟩

-- Define what a random event is
def is_random_event (e : Event) : Prop := e.is_possible ∧ ¬e.is_certain

-- Theorem stating that only event C is a random event
theorem only_C_is_random_event : 
  ¬is_random_event event_A ∧ 
  ¬is_random_event event_B ∧ 
  is_random_event event_C ∧ 
  ¬is_random_event event_D := by sorry

end only_C_is_random_event_l479_47902


namespace amelia_painted_faces_l479_47926

/-- The number of faces on a single cuboid -/
def faces_per_cuboid : ℕ := 6

/-- The number of cuboids Amelia painted -/
def number_of_cuboids : ℕ := 6

/-- The total number of faces painted by Amelia -/
def total_faces_painted : ℕ := faces_per_cuboid * number_of_cuboids

theorem amelia_painted_faces :
  total_faces_painted = 36 :=
by sorry

end amelia_painted_faces_l479_47926


namespace justin_tim_games_l479_47915

/-- The total number of players in the four-square league -/
def total_players : ℕ := 12

/-- The number of players in each game -/
def players_per_game : ℕ := 6

/-- Justin and Tim are two specific players -/
def justin_and_tim : ℕ := 2

/-- The number of remaining players after Justin and Tim -/
def remaining_players : ℕ := total_players - justin_and_tim

/-- The number of additional players needed in a game with Justin and Tim -/
def additional_players : ℕ := players_per_game - justin_and_tim

theorem justin_tim_games (total_players : ℕ) (players_per_game : ℕ) (justin_and_tim : ℕ) 
  (remaining_players : ℕ) (additional_players : ℕ) :
  total_players = 12 →
  players_per_game = 6 →
  justin_and_tim = 2 →
  remaining_players = total_players - justin_and_tim →
  additional_players = players_per_game - justin_and_tim →
  Nat.choose remaining_players additional_players = 210 :=
by sorry

end justin_tim_games_l479_47915


namespace min_value_f_prime_2_l479_47975

theorem min_value_f_prime_2 (a : ℝ) (h : a > 0) :
  let f := fun x : ℝ => x^3 + 2*a*x^2 + (1/a)*x
  let f_prime := fun x : ℝ => 3*x^2 + 4*a*x + 1/a
  ∀ x : ℝ, f_prime 2 ≥ 12 + 4*Real.sqrt 2 :=
by sorry

end min_value_f_prime_2_l479_47975


namespace exists_min_value_in_interval_l479_47907

-- Define the function
def f (x : ℝ) : ℝ := x^3 + x^2 - x + 1

-- State the theorem
theorem exists_min_value_in_interval :
  ∃ (m : ℝ), ∀ (x : ℝ), -2 ≤ x ∧ x ≤ 1 → m ≤ f x :=
sorry

end exists_min_value_in_interval_l479_47907


namespace meaningful_expression_l479_47966

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = (x - 1)^(0 : ℕ) / Real.sqrt (x + 2)) ↔ x > -2 ∧ x ≠ 1 := by
  sorry

end meaningful_expression_l479_47966


namespace michael_record_score_l479_47919

/-- Given a basketball team's total score and the average score of other players,
    calculate Michael's score that set the new school record. -/
theorem michael_record_score (total_score : ℕ) (other_players : ℕ) (avg_score : ℕ) 
    (h1 : total_score = 75)
    (h2 : other_players = 5)
    (h3 : avg_score = 6) :
    total_score - (other_players * avg_score) = 45 := by
  sorry

#check michael_record_score

end michael_record_score_l479_47919


namespace students_not_in_biology_l479_47980

theorem students_not_in_biology (total_students : ℕ) (biology_percentage : ℚ) 
  (h1 : total_students = 880) 
  (h2 : biology_percentage = 325 / 1000) : 
  total_students - (total_students * biology_percentage).floor = 594 := by
  sorry

end students_not_in_biology_l479_47980


namespace salmon_sales_ratio_l479_47909

/-- Given the first week's salmon sales and the total sales over two weeks,
    prove that the ratio of the second week's sales to the first week's sales is 3:1 -/
theorem salmon_sales_ratio (first_week : ℝ) (total : ℝ) :
  first_week = 50 →
  total = 200 →
  (total - first_week) / first_week = 3 := by
sorry

end salmon_sales_ratio_l479_47909


namespace geometric_sequence_grouping_l479_47944

/-- Given a geometric sequence with common ratio q ≠ 1, prove that the sequence
    formed by grouping every three terms is also geometric with ratio q^3 -/
theorem geometric_sequence_grouping (q : ℝ) (hq : q ≠ 1) :
  ∀ (a : ℕ → ℝ), (∀ n, a (n + 1) = q * a n) →
  ∃ (b : ℕ → ℝ), (∀ n, b n = a (3*n - 2) + a (3*n - 1) + a (3*n)) ∧
                 (∀ n, b (n + 1) = q^3 * b n) :=
by sorry

end geometric_sequence_grouping_l479_47944


namespace correct_number_of_sons_l479_47957

/-- Represents the problem of dividing land among sons --/
structure LandDivision where
  total_land : ℝ  -- Total land in hectares
  hectare_to_sqm : ℝ  -- Conversion factor from hectare to square meters
  profit_area : ℝ  -- Area in square meters that yields a certain profit
  profit_per_quarter : ℝ  -- Profit in dollars per quarter for profit_area
  son_yearly_profit : ℝ  -- Yearly profit for each son in dollars

/-- Calculate the number of sons based on land division --/
def calculate_sons (ld : LandDivision) : ℕ :=
  sorry

/-- Theorem stating the correct number of sons --/
theorem correct_number_of_sons (ld : LandDivision) 
  (h1 : ld.total_land = 3)
  (h2 : ld.hectare_to_sqm = 10000)
  (h3 : ld.profit_area = 750)
  (h4 : ld.profit_per_quarter = 500)
  (h5 : ld.son_yearly_profit = 10000) :
  calculate_sons ld = 8 := by
    sorry

end correct_number_of_sons_l479_47957


namespace smallest_valid_k_l479_47906

def sum_to(m : ℕ) : ℕ := m * (m + 1) / 2

def is_valid_k(k : ℕ) : Prop :=
  ∃ n : ℕ, n > k ∧ sum_to k = sum_to n - sum_to k

theorem smallest_valid_k :
  (∀ k : ℕ, k > 6 ∧ k < 9 → ¬is_valid_k k) ∧
  is_valid_k 9 :=
sorry

end smallest_valid_k_l479_47906


namespace green_ball_probability_l479_47970

structure Container where
  red : ℕ
  green : ℕ

def X : Container := { red := 5, green := 7 }
def Y : Container := { red := 7, green := 5 }
def Z : Container := { red := 7, green := 5 }

def total_containers : ℕ := 3

def prob_select_container : ℚ := 1 / total_containers

def prob_green (c : Container) : ℚ := c.green / (c.red + c.green)

def total_prob_green : ℚ := 
  prob_select_container * prob_green X + 
  prob_select_container * prob_green Y + 
  prob_select_container * prob_green Z

theorem green_ball_probability : total_prob_green = 17 / 36 := by
  sorry

end green_ball_probability_l479_47970


namespace parabola_properties_l479_47913

-- Define the parabola
def parabola (x : ℝ) : ℝ := 2 * x^2 + 4 * x - 6

-- Define the vertex of the parabola
def vertex : ℝ × ℝ := (-1, -8)

-- Define the shift
def m : ℝ := 3

-- Theorem statement
theorem parabola_properties :
  (∀ x, parabola x ≥ parabola (vertex.1)) ∧ 
  parabola (vertex.1) = vertex.2 ∧
  parabola (m - 0) = 0 :=
sorry

end parabola_properties_l479_47913


namespace polynomial_condition_implies_monomial_l479_47981

/-- A polynomial with nonnegative coefficients and degree ≤ n -/
def NonNegPolynomial (n : ℕ) := {p : Polynomial ℝ // p.degree ≤ n ∧ ∀ i, 0 ≤ p.coeff i}

theorem polynomial_condition_implies_monomial {n : ℕ} (P : NonNegPolynomial n) :
  (∀ x : ℝ, x > 0 → P.val.eval x * P.val.eval (1/x) ≤ (P.val.eval 1)^2) →
  ∃ (k : ℕ) (a : ℝ), k ≤ n ∧ a ≥ 0 ∧ P.val = Polynomial.monomial k a :=
sorry

end polynomial_condition_implies_monomial_l479_47981


namespace M_union_S_eq_M_l479_47922

-- Define set M
def M : Set ℝ := {y | ∃ x, y = Real.exp (x * Real.log 2)}

-- Define set S
def S : Set ℝ := {x | x > 1}

-- Theorem to prove
theorem M_union_S_eq_M : M ∪ S = M := by
  sorry

end M_union_S_eq_M_l479_47922


namespace inequality_proof_l479_47984

theorem inequality_proof (x y z : ℝ) 
  (h1 : 0 < x) (h2 : x < y) (h3 : y < z) (h4 : z < π/2) : 
  π/2 + 2 * Real.sin x * Real.cos y + 2 * Real.sin y * Real.cos z > 
  Real.sin (2*x) + Real.sin (2*y) + Real.sin (2*z) := by
  sorry

end inequality_proof_l479_47984


namespace pentagonal_prism_edges_l479_47945

/-- A pentagonal prism is a three-dimensional shape with two pentagonal bases connected by lateral edges. -/
structure PentagonalPrism where
  base_edges : ℕ  -- Number of edges in one pentagonal base
  lateral_edges : ℕ  -- Number of lateral edges connecting the two bases

/-- Theorem: A pentagonal prism has 15 edges. -/
theorem pentagonal_prism_edges (p : PentagonalPrism) : 
  p.base_edges = 5 → p.lateral_edges = 5 → p.base_edges * 2 + p.lateral_edges = 15 := by
  sorry

#check pentagonal_prism_edges

end pentagonal_prism_edges_l479_47945


namespace speed_ratio_l479_47997

-- Define the speeds of A and B
def v_A : ℝ := sorry
def v_B : ℝ := sorry

-- Define the initial position of B
def initial_B_position : ℝ := -800

-- Define the equidistant condition at 3 minutes
def equidistant_3min : Prop :=
  3 * v_A = abs (initial_B_position + 3 * v_B)

-- Define the equidistant condition at 9 minutes
def equidistant_9min : Prop :=
  9 * v_A = abs (initial_B_position + 9 * v_B)

-- Theorem statement
theorem speed_ratio :
  equidistant_3min →
  equidistant_9min →
  v_A / v_B = 9 / 10 :=
by
  sorry

end speed_ratio_l479_47997


namespace least_skilled_painter_is_granddaughter_l479_47940

-- Define the family members
inductive FamilyMember
  | Grandmother
  | Niece
  | Nephew
  | Granddaughter

-- Define the skill levels
inductive SkillLevel
  | Best
  | Least

-- Define the gender
inductive Gender
  | Male
  | Female

-- Function to get the gender of a family member
def gender (m : FamilyMember) : Gender :=
  match m with
  | FamilyMember.Grandmother => Gender.Female
  | FamilyMember.Niece => Gender.Female
  | FamilyMember.Nephew => Gender.Male
  | FamilyMember.Granddaughter => Gender.Female

-- Function to determine if two family members can be twins
def canBeTwins (m1 m2 : FamilyMember) : Prop :=
  (m1 = FamilyMember.Niece ∧ m2 = FamilyMember.Nephew) ∨
  (m1 = FamilyMember.Nephew ∧ m2 = FamilyMember.Niece) ∨
  (m1 = FamilyMember.Granddaughter ∧ m2 = FamilyMember.Nephew) ∨
  (m1 = FamilyMember.Nephew ∧ m2 = FamilyMember.Granddaughter)

-- Function to determine if two family members can be the same age
def canBeSameAge (m1 m2 : FamilyMember) : Prop :=
  (m1 = FamilyMember.Niece ∧ m2 = FamilyMember.Nephew) ∨
  (m1 = FamilyMember.Nephew ∧ m2 = FamilyMember.Niece) ∨
  (m1 = FamilyMember.Niece ∧ m2 = FamilyMember.Granddaughter) ∨
  (m1 = FamilyMember.Granddaughter ∧ m2 = FamilyMember.Niece)

-- Theorem statement
theorem least_skilled_painter_is_granddaughter :
  ∀ (best least : FamilyMember),
    (gender best ≠ gender least) →
    (∃ twin, canBeTwins twin least ∧ twin ≠ least) →
    canBeSameAge best least →
    least = FamilyMember.Granddaughter :=
by
  sorry

end least_skilled_painter_is_granddaughter_l479_47940


namespace triangle_inequality_sum_l479_47986

theorem triangle_inequality_sum (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  Real.sqrt (a / (b + c - a)) + Real.sqrt (b / (c + a - b)) + Real.sqrt (c / (a + b - c)) ≥ 3 := by
  sorry

end triangle_inequality_sum_l479_47986


namespace unique_solution_for_equation_l479_47992

theorem unique_solution_for_equation (b : ℝ) (hb : b ≠ 0) :
  ∃! a : ℝ, a ≠ 0 ∧ (a^2 / b + b^2 / a = (a + b)^2 / (a + b)) :=
by
  sorry

end unique_solution_for_equation_l479_47992


namespace wood_length_after_sawing_l479_47965

theorem wood_length_after_sawing (original_length sawing_length : ℝ) 
  (h1 : original_length = 8.9)
  (h2 : sawing_length = 2.3) : 
  original_length - sawing_length = 6.6 := by
  sorry

end wood_length_after_sawing_l479_47965


namespace allan_brought_two_balloons_l479_47914

/-- The number of balloons Allan and Jake had in total -/
def total_balloons : ℕ := 6

/-- The number of balloons Jake brought -/
def jake_balloons : ℕ := 4

/-- The number of balloons Allan brought -/
def allan_balloons : ℕ := total_balloons - jake_balloons

theorem allan_brought_two_balloons : allan_balloons = 2 := by
  sorry

end allan_brought_two_balloons_l479_47914


namespace area_of_specific_trapezoid_l479_47931

/-- An isosceles trapezoid with an inscribed circle -/
structure IsoscelesTrapezoidWithInscribedCircle where
  /-- The length of each lateral side -/
  lateral_side : ℝ
  /-- The radius of the inscribed circle -/
  inscribed_radius : ℝ

/-- The area of an isosceles trapezoid with an inscribed circle -/
def area (t : IsoscelesTrapezoidWithInscribedCircle) : ℝ :=
  2 * t.lateral_side * t.inscribed_radius

/-- Theorem: The area of an isosceles trapezoid with lateral side length 9 and an inscribed circle of radius 4 is 72 -/
theorem area_of_specific_trapezoid :
  let t : IsoscelesTrapezoidWithInscribedCircle := ⟨9, 4⟩
  area t = 72 := by sorry

end area_of_specific_trapezoid_l479_47931


namespace quadratic_equation_roots_l479_47972

/-- Given a quadratic equation x^2 - 4x + m = 0 with one root x₁ = 1, 
    prove that the other root x₂ = 3 -/
theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = 3 ∧ 
   ∀ x : ℝ, x^2 - 4*x + m = 0 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end quadratic_equation_roots_l479_47972


namespace root_product_equation_l479_47939

theorem root_product_equation (a b m p r : ℝ) : 
  (a^2 - m*a + 3 = 0) →
  (b^2 - m*b + 3 = 0) →
  ((a^2 + 1/b^2)^2 - p*(a^2 + 1/b^2) + r = 0) →
  ((b^2 + 1/a^2)^2 - p*(b^2 + 1/a^2) + r = 0) →
  r = 100/9 := by
sorry

end root_product_equation_l479_47939


namespace no_integer_solution_for_equation_l479_47983

theorem no_integer_solution_for_equation : ∀ x y : ℤ, x^2 - y^2 ≠ 210 := by
  sorry

end no_integer_solution_for_equation_l479_47983


namespace jar_weight_percentage_l479_47941

theorem jar_weight_percentage (jar_weight : ℝ) (full_weight : ℝ) 
  (h1 : jar_weight = 0.4 * full_weight)
  (h2 : jar_weight > 0)
  (h3 : full_weight > jar_weight) :
  let beans_weight := full_weight - jar_weight
  let remaining_beans_weight := (1/3) * beans_weight
  let new_total_weight := jar_weight + remaining_beans_weight
  new_total_weight / full_weight = 0.6 := by
sorry

end jar_weight_percentage_l479_47941


namespace inequality_equivalence_l479_47930

theorem inequality_equivalence (x : ℝ) : (1 / (x - 1) > 1) ↔ (1 < x ∧ x < 2) :=
sorry

end inequality_equivalence_l479_47930


namespace kitchen_area_is_265_l479_47904

def total_area : ℕ := 1110
def num_bedrooms : ℕ := 4
def bedroom_length : ℕ := 11
def num_bathrooms : ℕ := 2
def bathroom_length : ℕ := 6
def bathroom_width : ℕ := 8

def bedroom_area : ℕ := bedroom_length * bedroom_length
def bathroom_area : ℕ := bathroom_length * bathroom_width
def total_bedroom_area : ℕ := num_bedrooms * bedroom_area
def total_bathroom_area : ℕ := num_bathrooms * bathroom_area
def remaining_area : ℕ := total_area - (total_bedroom_area + total_bathroom_area)

theorem kitchen_area_is_265 : remaining_area / 2 = 265 := by
  sorry

end kitchen_area_is_265_l479_47904


namespace sokka_fish_count_l479_47988

theorem sokka_fish_count (aang_fish : ℕ) (toph_fish : ℕ) (average_fish : ℕ) (total_people : ℕ) :
  aang_fish = 7 →
  toph_fish = 12 →
  average_fish = 8 →
  total_people = 3 →
  ∃ sokka_fish : ℕ, sokka_fish = total_people * average_fish - (aang_fish + toph_fish) :=
by
  sorry

end sokka_fish_count_l479_47988


namespace polynomial_simplification_l479_47996

theorem polynomial_simplification (x : ℝ) : 
  (12 * x^10 - 3 * x^9 + 8 * x^8 - 5 * x^7) - 
  (2 * x^10 + 2 * x^9 - x^8 + x^7 + 4 * x^4 + 6 * x^2 + 9) = 
  10 * x^10 - 5 * x^9 + 9 * x^8 - 6 * x^7 - 4 * x^4 - 6 * x^2 - 9 := by
  sorry

end polynomial_simplification_l479_47996


namespace min_p_plus_q_l479_47985

def is_repeating_decimal (p q : ℕ+) : Prop :=
  (p : ℚ) / q = 0.198

theorem min_p_plus_q (p q : ℕ+) (h : is_repeating_decimal p q) 
  (h_min : ∀ (p' q' : ℕ+), is_repeating_decimal p' q' → q ≤ q') : 
  p + q = 121 := by
  sorry

end min_p_plus_q_l479_47985


namespace largest_inscribed_square_l479_47920

theorem largest_inscribed_square (outer_square_side : ℝ) 
  (h_outer_square : outer_square_side = 12) : ℝ :=
  let triangle_side := 4 * Real.sqrt 6
  let inscribed_square_side := 6 - 2 * Real.sqrt 3
  inscribed_square_side

#check largest_inscribed_square

end largest_inscribed_square_l479_47920


namespace octahedron_sphere_probability_l479_47976

/-- Represents a regular octahedron with inscribed and circumscribed spheres -/
structure OctahedronWithSpheres where
  /-- Radius of the circumscribed sphere -/
  R : ℝ
  /-- Radius of the inscribed sphere -/
  r : ℝ
  /-- Assumption that the inscribed sphere radius is one-third of the circumscribed sphere radius -/
  h_r_eq : r = R / 3

/-- The probability that a randomly chosen point in the circumscribed sphere
    lies inside one of the nine smaller spheres (one inscribed and eight tangent to faces) -/
theorem octahedron_sphere_probability (o : OctahedronWithSpheres) :
  (volume_ratio : ℝ) = 1 / 3 :=
sorry

end octahedron_sphere_probability_l479_47976


namespace cereal_box_price_calculation_l479_47999

theorem cereal_box_price_calculation 
  (initial_price : ℕ) 
  (price_reduction : ℕ) 
  (num_boxes : ℕ) : 
  initial_price = 104 → 
  price_reduction = 24 → 
  num_boxes = 20 → 
  (initial_price - price_reduction) * num_boxes = 1600 := by
sorry

end cereal_box_price_calculation_l479_47999


namespace polygon_area_is_144_l479_47925

/-- A polygon with perpendicular adjacent sides -/
structure PerpendicularPolygon where
  sides : ℕ
  side_length : ℝ
  perimeter : ℝ
  area : ℝ

/-- Our specific polygon -/
def our_polygon : PerpendicularPolygon where
  sides := 36
  side_length := 2
  perimeter := 72
  area := 144

theorem polygon_area_is_144 (p : PerpendicularPolygon) 
  (h1 : p.sides = 36) 
  (h2 : p.perimeter = 72) 
  (h3 : p.side_length = p.perimeter / p.sides) : 
  p.area = 144 := by
  sorry

#check polygon_area_is_144

end polygon_area_is_144_l479_47925


namespace tan_alpha_three_implies_sin_2alpha_over_cos_squared_alpha_eq_six_l479_47948

theorem tan_alpha_three_implies_sin_2alpha_over_cos_squared_alpha_eq_six
  (α : Real)
  (h : Real.tan α = 3) :
  (Real.sin (2 * α)) / (Real.cos α)^2 = 6 := by
  sorry

end tan_alpha_three_implies_sin_2alpha_over_cos_squared_alpha_eq_six_l479_47948


namespace perfect_square_trinomial_l479_47927

theorem perfect_square_trinomial (k : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 - k*x + 9 = (x - a)^2) → (k = 6 ∨ k = -6) := by
  sorry

end perfect_square_trinomial_l479_47927


namespace park_boats_l479_47908

theorem park_boats (total_boats : ℕ) (large_capacity : ℕ) (small_capacity : ℕ) 
  (h1 : total_boats = 42)
  (h2 : large_capacity = 6)
  (h3 : small_capacity = 4)
  (h4 : ∃ (large_boats small_boats : ℕ), 
    large_boats + small_boats = total_boats ∧ 
    large_capacity * large_boats = 2 * small_capacity * small_boats) :
  ∃ (large_boats small_boats : ℕ), 
    large_boats = 24 ∧ 
    small_boats = 18 ∧ 
    large_boats + small_boats = total_boats ∧ 
    large_capacity * large_boats = 2 * small_capacity * small_boats :=
by sorry

end park_boats_l479_47908


namespace tangent_fifteen_degrees_ratio_l479_47991

theorem tangent_fifteen_degrees_ratio (π : Real) :
  let tan15 := Real.tan (15 * π / 180)
  (1 + tan15) / (1 - tan15) = Real.sqrt 3 :=
by sorry

end tangent_fifteen_degrees_ratio_l479_47991


namespace matrix_power_zero_l479_47989

theorem matrix_power_zero (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : A ^ 4 = 0) : A ^ 3 = 0 := by
  sorry

end matrix_power_zero_l479_47989


namespace fractional_equation_to_polynomial_l479_47959

theorem fractional_equation_to_polynomial (x y : ℝ) (h1 : (2*x - 1)/x^2 + x^2/(2*x - 1) = 5) (h2 : (2*x - 1)/x^2 = y) : y^2 - 5*y + 1 = 0 := by
  sorry

end fractional_equation_to_polynomial_l479_47959


namespace cos_alpha_minus_pi_l479_47938

theorem cos_alpha_minus_pi (α : Real) 
  (h1 : π / 2 < α) 
  (h2 : α < π) 
  (h3 : 3 * Real.sin (2 * α) = 2 * Real.cos α) : 
  Real.cos (α - π) = 2 * Real.sqrt 2 / 3 := by
  sorry

end cos_alpha_minus_pi_l479_47938


namespace popcorn_cost_l479_47903

/-- The cost of each box of popcorn for three friends splitting movie expenses -/
theorem popcorn_cost (ticket_price movie_tickets popcorn_boxes milktea_price milktea_cups individual_contribution : ℚ) :
  (ticket_price = 7) →
  (movie_tickets = 3) →
  (popcorn_boxes = 2) →
  (milktea_price = 3) →
  (milktea_cups = 3) →
  (individual_contribution = 11) →
  (((ticket_price * movie_tickets) + (milktea_price * milktea_cups) + 
    (popcorn_boxes * ((individual_contribution * 3) - 
    (ticket_price * movie_tickets) - (milktea_price * milktea_cups)) / popcorn_boxes)) / 3 = individual_contribution) →
  ((individual_contribution * 3) - (ticket_price * movie_tickets) - (milktea_price * milktea_cups)) / popcorn_boxes = (3/2 : ℚ) := by
  sorry


end popcorn_cost_l479_47903


namespace quadratic_equation_solution_l479_47969

theorem quadratic_equation_solution (k : ℚ) : 
  (∀ x : ℚ, k * x^2 + 8 * x + 15 = 0 ↔ (x = -3 ∨ x = -5/2)) → k = 11/4 := by
  sorry

end quadratic_equation_solution_l479_47969


namespace distance_between_foci_l479_47961

-- Define the three known endpoints of the ellipse's axes
def point1 : ℝ × ℝ := (-3, 5)
def point2 : ℝ × ℝ := (4, -3)
def point3 : ℝ × ℝ := (9, 5)

-- Define the ellipse based on these points
def ellipse_from_points (p1 p2 p3 : ℝ × ℝ) : Type := sorry

-- Theorem stating the distance between foci
theorem distance_between_foci 
  (e : ellipse_from_points point1 point2 point3) : 
  ∃ (f1 f2 : ℝ × ℝ), dist f1 f2 = 4 * Real.sqrt 7 :=
sorry

end distance_between_foci_l479_47961


namespace base_10_to_base_7_l479_47958

theorem base_10_to_base_7 : 
  (2 * 7^3 + 2 * 7^2 + 0 * 7^1 + 5 * 7^0 : ℕ) = 789 := by
sorry

end base_10_to_base_7_l479_47958


namespace min_value_of_x_plus_reciprocal_min_value_achieved_l479_47978

theorem min_value_of_x_plus_reciprocal (x : ℝ) (h : x > 1) :
  x + 1 / (x - 1) ≥ 3 :=
sorry

theorem min_value_achieved (x : ℝ) (h : x > 1) :
  x + 1 / (x - 1) = 3 ↔ x = 2 :=
sorry

end min_value_of_x_plus_reciprocal_min_value_achieved_l479_47978


namespace fractional_equation_root_l479_47924

theorem fractional_equation_root (x m : ℝ) : 
  ((x - 5) / (x + 2) = m / (x + 2) ∧ x + 2 ≠ 0) → m = -7 := by
  sorry

end fractional_equation_root_l479_47924


namespace aero_tees_count_l479_47935

/-- The number of people golfing -/
def num_people : ℕ := 4

/-- The number of tees in a package of generic tees -/
def generic_package_size : ℕ := 12

/-- The maximum number of generic packages Bill will buy -/
def max_generic_packages : ℕ := 2

/-- The minimum number of tees needed per person -/
def min_tees_per_person : ℕ := 20

/-- The number of aero flight tee packages Bill must purchase -/
def aero_packages : ℕ := 28

/-- The number of aero flight tees in one package -/
def aero_tees_per_package : ℕ := 2

theorem aero_tees_count : 
  num_people * min_tees_per_person ≤ 
  max_generic_packages * generic_package_size + 
  aero_packages * aero_tees_per_package ∧
  num_people * min_tees_per_person > 
  max_generic_packages * generic_package_size + 
  aero_packages * (aero_tees_per_package - 1) :=
by sorry

end aero_tees_count_l479_47935


namespace first_sample_is_three_l479_47953

/-- Represents a systematic sampling scenario -/
structure SystematicSampling where
  totalPopulation : Nat
  sampleSize : Nat
  lastSample : Nat

/-- Calculates the first sample in a systematic sampling scenario -/
def firstSample (s : SystematicSampling) : Nat :=
  s.lastSample - (s.sampleSize - 1) * (s.totalPopulation / s.sampleSize)

/-- Theorem: In the given systematic sampling scenario, the first sample is 3 -/
theorem first_sample_is_three :
  let s : SystematicSampling := ⟨300, 60, 298⟩
  firstSample s = 3 := by sorry

end first_sample_is_three_l479_47953


namespace fraction_sum_l479_47973

theorem fraction_sum (x y : ℚ) (h : x / y = 4 / 7) : (x + y) / y = 11 / 7 := by
  sorry

end fraction_sum_l479_47973


namespace relationship_abc_l479_47987

theorem relationship_abc : 
  let a : ℝ := (3/7)^(2/7)
  let b : ℝ := (2/7)^(3/7)
  let c : ℝ := (2/7)^(2/7)
  a > c ∧ c > b := by sorry

end relationship_abc_l479_47987


namespace min_value_of_expression_l479_47954

theorem min_value_of_expression (x y : ℝ) :
  (x + y + x * y)^2 + (x - y - x * y)^2 ≥ 0 ∧
  ∃ a b : ℝ, (a + b + a * b)^2 + (a - b - a * b)^2 = 0 :=
sorry

end min_value_of_expression_l479_47954


namespace max_angles_less_than_108_is_4_l479_47994

/-- The maximum number of angles less than 108° in a convex polygon -/
def max_angles_less_than_108 (n : ℕ) : ℕ := 4

/-- Theorem stating that the maximum number of angles less than 108° in a convex n-gon is 4 -/
theorem max_angles_less_than_108_is_4 (n : ℕ) (h : n ≥ 3) :
  max_angles_less_than_108 n = 4 := by sorry

end max_angles_less_than_108_is_4_l479_47994


namespace geometric_sequence_sum_4_to_7_l479_47962

/-- The sum of the 4th to 7th terms of a geometric sequence with first term 1 and common ratio 3 is 1080 -/
theorem geometric_sequence_sum_4_to_7 :
  let a : ℕ → ℝ := λ n => 1 * (3 : ℝ) ^ (n - 1)
  (a 4) + (a 5) + (a 6) + (a 7) = 1080 := by
  sorry

end geometric_sequence_sum_4_to_7_l479_47962


namespace complex_equation_implies_exponent_one_l479_47977

theorem complex_equation_implies_exponent_one (x y : ℝ) 
  (h : (x + y) * Complex.I = x - 1) : 
  (2 : ℝ) ^ (x + y) = 1 := by
  sorry

end complex_equation_implies_exponent_one_l479_47977


namespace medical_supply_transport_l479_47967

/-- Given two locations A and B that are 360 kilometers apart, a truck carrying 6 boxes of medical supplies
    traveling from A to B at 40 km/h, and a motorcycle departing from B towards the truck at 80 km/h,
    this theorem proves that the total time needed to transport all 6 boxes to location B is 26/3 hours
    and the total distance traveled by the motorcycle is 2080/3 kilometers. -/
theorem medical_supply_transport (distance_AB : ℝ) (truck_speed : ℝ) (motorcycle_speed : ℝ) 
  (boxes : ℕ) (boxes_per_trip : ℕ) :
  distance_AB = 360 →
  truck_speed = 40 →
  motorcycle_speed = 80 →
  boxes = 6 →
  boxes_per_trip = 2 →
  ∃ (total_time : ℝ) (total_distance : ℝ),
    total_time = 26/3 ∧
    total_distance = 2080/3 :=
by sorry


end medical_supply_transport_l479_47967


namespace ninth_root_of_unity_sum_l479_47955

theorem ninth_root_of_unity_sum (ω : ℂ) (h1 : ω ^ 9 = 1) (h2 : ω ≠ 1) :
  ω^18 + ω^27 + ω^36 + ω^45 + ω^54 + ω^63 + ω^72 + ω^81 + ω^90 = 0 :=
by sorry

end ninth_root_of_unity_sum_l479_47955


namespace karen_cookies_l479_47947

/-- Given Karen's cookie distribution, prove she kept 10 for herself -/
theorem karen_cookies (total : ℕ) (grandparents : ℕ) (class_size : ℕ) (per_person : ℕ)
  (h1 : total = 50)
  (h2 : grandparents = 8)
  (h3 : class_size = 16)
  (h4 : per_person = 2) :
  total - (grandparents + class_size * per_person) = 10 := by
  sorry

#eval 50 - (8 + 16 * 2)  -- Expected output: 10

end karen_cookies_l479_47947


namespace initial_deer_families_l479_47982

/-- The number of deer families that stayed in the area -/
def families_stayed : ℕ := 45

/-- The number of deer families that moved out of the area -/
def families_moved_out : ℕ := 34

/-- The initial number of deer families in the area -/
def initial_families : ℕ := families_stayed + families_moved_out

theorem initial_deer_families : initial_families = 79 := by
  sorry

end initial_deer_families_l479_47982


namespace bake_sale_group_composition_l479_47990

theorem bake_sale_group_composition (p : ℕ) : 
  (p : ℚ) / 2 = (((p : ℚ) / 2 - 5) / p) * 100 → p / 2 = 25 := by
  sorry

end bake_sale_group_composition_l479_47990


namespace pennies_found_l479_47993

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- The number of quarters found -/
def num_quarters : ℕ := 12

/-- The total value in cents -/
def total_value : ℕ := 307

/-- The number of pennies found -/
def num_pennies : ℕ := (total_value - num_quarters * quarter_value) / penny_value

theorem pennies_found : num_pennies = 7 := by
  sorry

end pennies_found_l479_47993


namespace five_digit_sum_l479_47968

def is_valid_digit (x : ℕ) : Prop := 1 ≤ x ∧ x ≤ 9

def sum_of_digits (x : ℕ) : ℕ := 120 * (1 + 3 + 4 + 6 + x)

theorem five_digit_sum (x : ℕ) (h1 : is_valid_digit x) (h2 : sum_of_digits x = 2640) : x = 8 := by
  sorry

end five_digit_sum_l479_47968


namespace arithmetic_calculations_l479_47979

theorem arithmetic_calculations :
  ((54 + 38) * 15 = 1380) ∧
  (1500 - 32 * 45 = 60) ∧
  (157 * (70 / 35) = 314) := by
  sorry

end arithmetic_calculations_l479_47979


namespace weighted_average_two_groups_l479_47943

/-- Weighted average calculation for two groups of students -/
theorem weighted_average_two_groups 
  (x y : ℝ) -- x and y are real numbers representing average scores
  (total_students : ℕ := 25) -- total number of students
  (group_a_students : ℕ := 15) -- number of students in Group A
  (group_b_students : ℕ := 10) -- number of students in Group B
  (h1 : total_students = group_a_students + group_b_students) -- condition: total students is sum of both groups
  : (group_a_students * x + group_b_students * y) / total_students = (3 * x + 2 * y) / 5 :=
by
  sorry

#check weighted_average_two_groups

end weighted_average_two_groups_l479_47943


namespace mitchs_family_milk_consumption_l479_47912

/-- The total milk consumption of Mitch's family in one week -/
def total_milk_consumption (regular_milk soy_milk almond_milk oat_milk : ℝ) : ℝ :=
  regular_milk + soy_milk + almond_milk + oat_milk

/-- Theorem stating the total milk consumption of Mitch's family -/
theorem mitchs_family_milk_consumption :
  total_milk_consumption 1.75 0.85 1.25 0.65 = 4.50 := by
  sorry

end mitchs_family_milk_consumption_l479_47912


namespace daves_initial_apps_daves_initial_apps_proof_l479_47934

theorem daves_initial_apps : ℕ :=
  let initial_files : ℕ := 9
  let final_files : ℕ := 5
  let final_apps : ℕ := 12
  let app_file_difference : ℕ := 7

  have h1 : final_apps = final_files + app_file_difference := by sorry
  have h2 : ∃ (initial_apps : ℕ), initial_apps - final_apps = initial_files - final_files := by sorry

  16

theorem daves_initial_apps_proof : daves_initial_apps = 16 := by sorry

end daves_initial_apps_daves_initial_apps_proof_l479_47934


namespace coordinates_of_G_l479_47923

/-- Given a line segment OH with O at (0, 0) and H at (12, 0), 
    and a point G on the same vertical line as H,
    if the line from G through the midpoint M of OH intersects the y-axis at P(0, -4),
    then G has coordinates (12, 4) -/
theorem coordinates_of_G (O H G M P : ℝ × ℝ) : 
  O = (0, 0) →
  H = (12, 0) →
  G.1 = H.1 →
  M = ((O.1 + H.1) / 2, (O.2 + H.2) / 2) →
  P = (0, -4) →
  (∃ t : ℝ, G = t • (M - P) + P) →
  G = (12, 4) := by
  sorry

end coordinates_of_G_l479_47923


namespace flow_rate_increase_l479_47952

/-- Proves that the percentage increase in flow rate from the first to the second hour is 50% -/
theorem flow_rate_increase (r1 r2 r3 : ℝ) : 
  r2 = 36 →  -- Second hour flow rate
  r3 = 1.25 * r2 →  -- Third hour flow rate is 25% more than second
  r1 + r2 + r3 = 105 →  -- Total flow for all three hours
  r1 < r2 →  -- Second hour rate faster than first
  (r2 - r1) / r1 * 100 = 50 := by sorry

end flow_rate_increase_l479_47952


namespace least_number_with_remainders_l479_47998

theorem least_number_with_remainders : ∃ n : ℕ, 
  n > 0 ∧
  n % 5 = 3 ∧
  n % 6 = 3 ∧
  n % 7 = 3 ∧
  n % 8 = 3 ∧
  n % 9 = 0 ∧
  (∀ m : ℕ, m > 0 ∧ m % 5 = 3 ∧ m % 6 = 3 ∧ m % 7 = 3 ∧ m % 8 = 3 ∧ m % 9 = 0 → n ≤ m) ∧
  n = 1683 :=
by sorry

end least_number_with_remainders_l479_47998


namespace sum_of_f_zero_and_f_neg_two_l479_47974

def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x + f y = f (x + y)

theorem sum_of_f_zero_and_f_neg_two (f : ℝ → ℝ) 
  (h1 : functional_equation f) 
  (h2 : f 2 = 4) : 
  f 0 + f (-2) = -4 := by
  sorry

end sum_of_f_zero_and_f_neg_two_l479_47974


namespace square_sum_theorem_l479_47971

theorem square_sum_theorem (x y : ℝ) 
  (h1 : 3 * x + 4 * y = 8) 
  (h2 : x * y = -6) : 
  9 * x^2 + 16 * y^2 = 208 := by
  sorry

end square_sum_theorem_l479_47971


namespace max_profit_is_120_l479_47933

/-- Profit function for location A -/
def L₁ (x : ℕ) : ℤ := -x^2 + 21*x

/-- Profit function for location B -/
def L₂ (x : ℕ) : ℤ := 2*x

/-- Total profit function -/
def L (x : ℕ) : ℤ := L₁ x + L₂ (15 - x)

theorem max_profit_is_120 :
  ∃ x : ℕ, x ≤ 15 ∧ L x = 120 ∧ ∀ y : ℕ, y ≤ 15 → L y ≤ 120 :=
sorry

end max_profit_is_120_l479_47933


namespace solution_set_implies_b_value_l479_47950

theorem solution_set_implies_b_value (b : ℝ) : 
  (∀ x : ℝ, x^2 - b*x + 6 < 0 ↔ 2 < x ∧ x < 3) → b = 5 := by
  sorry

end solution_set_implies_b_value_l479_47950


namespace intersection_in_fourth_quadrant_implies_a_greater_half_l479_47956

/-- The intersection point of two lines is in the fourth quadrant if and only if
    its x-coordinate is positive and its y-coordinate is negative. -/
def in_fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- Given two lines y = -x + 1 and y = x - 2a, their intersection point
    is in the fourth quadrant implies a > 1/2 -/
theorem intersection_in_fourth_quadrant_implies_a_greater_half (a : ℝ) :
  (∃ x y : ℝ, y = -x + 1 ∧ y = x - 2 * a ∧ in_fourth_quadrant x y) →
  a > 1/2 :=
by sorry

end intersection_in_fourth_quadrant_implies_a_greater_half_l479_47956


namespace problem_solution_l479_47963

theorem problem_solution (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (∀ a b, a * b ≤ 1) ∧
  (∀ a b, 1 / a + 1 / b ≥ 2) ∧
  (∀ m : ℝ, (∀ x : ℝ, |x + m| - |x + 1| ≤ 1 / a + 1 / b) ↔ m ∈ Set.Icc (-1) 3) :=
by sorry

end problem_solution_l479_47963


namespace tetrahedral_pile_remaining_marbles_l479_47929

/-- The number of marbles in a tetrahedral pile of height k -/
def tetrahedralPile (k : ℕ) : ℕ := k * (k + 1) * (k + 2) / 6

/-- The total number of marbles -/
def totalMarbles : ℕ := 60

/-- The height of the largest possible tetrahedral pile -/
def maxHeight : ℕ := 6

/-- The number of remaining marbles -/
def remainingMarbles : ℕ := totalMarbles - tetrahedralPile maxHeight

theorem tetrahedral_pile_remaining_marbles :
  remainingMarbles = 4 := by sorry

end tetrahedral_pile_remaining_marbles_l479_47929


namespace range_of_circle_l479_47921

theorem range_of_circle (x y : ℝ) (h : x^2 + y^2 = 4*x) :
  ∃ (z : ℝ), z = x^2 + y^2 ∧ 0 ≤ z ∧ z ≤ 16 :=
sorry

end range_of_circle_l479_47921


namespace function_value_2009_l479_47960

theorem function_value_2009 (f : ℝ → ℝ) 
  (h1 : f 3 = -Real.sqrt 3) 
  (h2 : ∀ x : ℝ, f (x + 2) * (1 - f x) = 1 + f x) : 
  f 2009 = 2 + Real.sqrt 3 := by
sorry

end function_value_2009_l479_47960


namespace power_multiplication_l479_47917

theorem power_multiplication (x : ℝ) : x^2 * x^3 = x^5 := by
  sorry

end power_multiplication_l479_47917


namespace total_money_is_36_l479_47937

/-- Given Joanna's money, calculate the total money of Joanna, her brother, and her sister -/
def total_money (joanna_money : ℕ) : ℕ :=
  joanna_money + 3 * joanna_money + joanna_money / 2

/-- Theorem: The total money of Joanna, her brother, and her sister is $36 when Joanna has $8 -/
theorem total_money_is_36 : total_money 8 = 36 := by
  sorry

end total_money_is_36_l479_47937


namespace children_at_track_meet_l479_47964

theorem children_at_track_meet (total_seats : ℕ) (empty_seats : ℕ) (adults : ℕ) 
  (h1 : total_seats = 95)
  (h2 : empty_seats = 14)
  (h3 : adults = 29) :
  total_seats - empty_seats - adults = 52 := by
  sorry

end children_at_track_meet_l479_47964


namespace regular_octagon_area_l479_47916

theorem regular_octagon_area (s : ℝ) (h : s = Real.sqrt 2) :
  let square_side : ℝ := 2 + s
  let octagon_area : ℝ := square_side ^ 2 - 4 * (1 / 2)
  octagon_area = 4 + 4 * s := by sorry

end regular_octagon_area_l479_47916


namespace cos_sum_fifth_circle_l479_47900

theorem cos_sum_fifth_circle : Real.cos (2 * Real.pi / 5) + Real.cos (4 * Real.pi / 5) = -1/2 := by
  sorry

end cos_sum_fifth_circle_l479_47900


namespace solution_value_l479_47946

theorem solution_value (a b : ℝ) (h : a * 3^2 - b * 3 = 6) : 2023 - 6 * a + 2 * b = 2019 := by
  sorry

end solution_value_l479_47946


namespace fgh_supermarkets_in_us_l479_47928

theorem fgh_supermarkets_in_us (total : ℕ) (difference : ℕ) (us_count : ℕ) : 
  total = 60 →
  difference = 22 →
  us_count = total - difference →
  us_count = 41 :=
by
  sorry

end fgh_supermarkets_in_us_l479_47928


namespace square_sum_product_l479_47932

theorem square_sum_product (a b : ℝ) (ha : a = Real.sqrt 2 + 1) (hb : b = Real.sqrt 2 - 1) :
  a^2 + a*b + b^2 = 7 := by
  sorry

end square_sum_product_l479_47932


namespace hall_length_l479_47911

/-- The length of a hall given its breadth, number of stones, and stone dimensions -/
theorem hall_length (breadth : ℝ) (num_stones : ℕ) (stone_length stone_width : ℝ) :
  breadth = 15 ∧ 
  num_stones = 5400 ∧
  stone_length = 0.2 ∧
  stone_width = 0.5 →
  (num_stones * stone_length * stone_width) / breadth = 36 := by
sorry


end hall_length_l479_47911


namespace fraction_simplification_l479_47905

theorem fraction_simplification (a b : ℕ) (h : b ≠ 0) : 
  (4 * a) / (4 * b) = a / b :=
sorry

end fraction_simplification_l479_47905


namespace fraction_simplification_l479_47942

theorem fraction_simplification : (-150 + 50) / (-50) = 2 := by
  sorry

end fraction_simplification_l479_47942


namespace cheese_cost_is_seven_l479_47995

/-- The cost of a pound of cheese, given Tony's initial amount, beef cost, and remaining amount after purchase. -/
def cheese_cost (initial_amount beef_cost remaining_amount : ℚ) : ℚ :=
  (initial_amount - beef_cost - remaining_amount) / 3

/-- Theorem stating that the cost of a pound of cheese is $7 under the given conditions. -/
theorem cheese_cost_is_seven :
  let initial_amount : ℚ := 87
  let beef_cost : ℚ := 5
  let remaining_amount : ℚ := 61
  cheese_cost initial_amount beef_cost remaining_amount = 7 := by
  sorry

end cheese_cost_is_seven_l479_47995


namespace binary_11111011111_equals_2015_l479_47918

def binary_to_decimal (binary : List Bool) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem binary_11111011111_equals_2015 :
  binary_to_decimal [true, true, true, true, true, false, true, true, true, true, true] = 2015 := by
  sorry

end binary_11111011111_equals_2015_l479_47918


namespace chocolate_differences_l479_47949

/-- Given the number of chocolates eaten by Robert, Nickel, and Jessica,
    prove the differences between Robert's and Nickel's chocolates,
    and Jessica's and Nickel's chocolates. -/
theorem chocolate_differences (robert nickel jessica : ℕ) 
    (h_robert : robert = 23)
    (h_nickel : nickel = 8)
    (h_jessica : jessica = 15) :
    robert - nickel = 15 ∧ jessica - nickel = 7 := by
  sorry

end chocolate_differences_l479_47949


namespace rectangle_area_l479_47910

theorem rectangle_area (square_area : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) : 
  square_area = 36 →
  rectangle_width = Real.sqrt square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 108 := by
sorry

end rectangle_area_l479_47910


namespace quadratic_roots_relation_l479_47936

theorem quadratic_roots_relation (k : ℝ) : 
  (∀ x y : ℝ, x^2 + k*x + 12 = 0 ∧ y^2 - k*y + 12 = 0 → y = x + 7) → 
  k = -7 := by
sorry

end quadratic_roots_relation_l479_47936
