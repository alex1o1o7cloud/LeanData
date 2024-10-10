import Mathlib

namespace smallest_factor_sum_l2803_280330

theorem smallest_factor_sum (b : ℕ) (p q : ℤ) : 
  (∀ x, x^2 + b*x + 2040 = (x + p) * (x + q)) →
  (∀ b' : ℕ, b' < b → 
    ¬∃ p' q' : ℤ, ∀ x, x^2 + b'*x + 2040 = (x + p') * (x + q')) →
  b = 94 :=
sorry

end smallest_factor_sum_l2803_280330


namespace product_of_three_terms_l2803_280304

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = 2 * a n

theorem product_of_three_terms
  (a : ℕ → ℝ)
  (h1 : geometric_sequence a)
  (h2 : a 5 = 4) :
  a 4 * a 5 * a 6 = 64 := by
sorry

end product_of_three_terms_l2803_280304


namespace m_plus_n_is_zero_l2803_280329

-- Define the complex function f
def f (m n z : ℂ) : ℂ := z^2 + m*z + n

-- State the theorem
theorem m_plus_n_is_zero (m n : ℂ) 
  (h : ∀ z : ℂ, Complex.abs z = 1 → Complex.abs (f m n z) = 1) : 
  m + n = 0 := by
  sorry

end m_plus_n_is_zero_l2803_280329


namespace remainder_theorem_l2803_280347

theorem remainder_theorem (n : ℕ) : (2 * n) % 4 = 2 → n % 4 = 1 := by
  sorry

end remainder_theorem_l2803_280347


namespace trig_sum_equals_one_l2803_280311

theorem trig_sum_equals_one : 4 * Real.cos (Real.pi / 3) + 8 * Real.sin (Real.pi / 6) - 5 * Real.tan (Real.pi / 4) = 1 := by
  sorry

end trig_sum_equals_one_l2803_280311


namespace vector_function_properties_l2803_280392

noncomputable def a (x : ℝ) : ℝ × ℝ := (1 + Real.sin (2 * x), Real.sin x - Real.cos x)

noncomputable def b (x : ℝ) : ℝ × ℝ := (1, Real.sin x + Real.cos x)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem vector_function_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ ∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧
  (∃ (M : ℝ), ∀ (x : ℝ), f x ≤ M ∧ ∃ (x : ℝ), f x = M) ∧
  (∃ (S : Set ℝ), S = {x | ∃ (k : ℤ), x = 3 * Real.pi / 8 + k * Real.pi} ∧
    ∀ (x : ℝ), x ∈ S ↔ f x = Real.sqrt 2 + 1) := by
  sorry

end vector_function_properties_l2803_280392


namespace complement_intersection_theorem_l2803_280333

open Set

def U : Set Nat := {0, 1, 2, 3, 4}
def M : Set Nat := {0, 1, 2}
def N : Set Nat := {2, 3}

theorem complement_intersection_theorem :
  (U \ M) ∩ N = {3} := by sorry

end complement_intersection_theorem_l2803_280333


namespace set_equivalence_l2803_280377

def M : Set ℝ := {x | -3 < x ∧ x < 1}
def N : Set ℝ := {x | x ≤ -3}

theorem set_equivalence : {x : ℝ | x ≥ 1} = (Set.univ : Set ℝ) \ (M ∪ N) := by sorry

end set_equivalence_l2803_280377


namespace line_separates_points_l2803_280332

/-- Given that the origin (0,0) and the point (1,1) are on opposite sides of the line x+y=a,
    prove that the range of values for a is 0 < a < 2. -/
theorem line_separates_points (a : ℝ) : 
  (0 + 0 - a) * (1 + 1 - a) < 0 → 0 < a ∧ a < 2 :=
by sorry

end line_separates_points_l2803_280332


namespace f_min_max_l2803_280331

-- Define the function
def f (x : ℝ) : ℝ := 1 + 3*x - x^3

-- State the theorem
theorem f_min_max : 
  (∃ x : ℝ, f x = -1) ∧ 
  (∀ x : ℝ, f x ≥ -1) ∧ 
  (∃ x : ℝ, f x = 3) ∧ 
  (∀ x : ℝ, f x ≤ 3) := by sorry

end f_min_max_l2803_280331


namespace inequality_proof_l2803_280376

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^2 + y^4 + z^6 ≥ x*y^2 + y^2*z^3 + x*z^3 := by
  sorry

end inequality_proof_l2803_280376


namespace part_one_part_two_l2803_280389

-- Define the quadratic functions p and q
def p (x : ℝ) : Prop := x^2 - 7*x + 10 < 0
def q (m x : ℝ) : Prop := x^2 - 4*m*x + 3*m^2 < 0

-- Theorem for part (1)
theorem part_one (m : ℝ) (h : m = 4) :
  (∀ x, p x ∧ q m x ↔ 4 < x ∧ x < 5) :=
sorry

-- Theorem for part (2)
theorem part_two :
  (∀ m, (∀ x, ¬(q m x) → ¬(p x)) ∧ (∃ x, ¬(p x) ∧ q m x)) ↔ (5/3 ≤ m ∧ m ≤ 2) :=
sorry

end part_one_part_two_l2803_280389


namespace oil_leak_during_work_l2803_280320

/-- The amount of oil leaked while engineers were working, given the total amount leaked and the amount leaked before they started. -/
theorem oil_leak_during_work (total_leak : ℕ) (pre_work_leak : ℕ) 
  (h1 : total_leak = 11687)
  (h2 : pre_work_leak = 6522) :
  total_leak - pre_work_leak = 5165 := by
  sorry

end oil_leak_during_work_l2803_280320


namespace heartsuit_nested_equals_fourteen_l2803_280316

-- Define the ⊛ operation for positive real numbers
def heartsuit (x y : ℝ) : ℝ := x + 2 * y

-- State the theorem
theorem heartsuit_nested_equals_fourteen :
  heartsuit 2 (heartsuit 2 2) = 14 := by
  sorry

end heartsuit_nested_equals_fourteen_l2803_280316


namespace smallest_base_for_fourth_power_l2803_280341

/-- Given an integer N represented as 777 in base b, 
    prove that 18 is the smallest positive integer b 
    such that N is the fourth power of a decimal integer -/
theorem smallest_base_for_fourth_power (N : ℤ) (b : ℕ+) : 
  (N = 7 * b^2 + 7 * b + 7) →
  (∃ (x : ℤ), N = x^4) →
  (∀ (b' : ℕ+), b' < b → ¬∃ (x : ℤ), 7 * b'^2 + 7 * b' + 7 = x^4) →
  b = 18 := by
sorry

end smallest_base_for_fourth_power_l2803_280341


namespace exists_large_remainder_sum_l2803_280351

/-- Given positive integers N and a, generates a sequence of remainders by repeatedly dividing N by the last remainder, starting with a, until 0 is reached. -/
def remainderSequence (N a : ℕ+) : List ℕ :=
  sorry

/-- The theorem states that there exist positive integers N and a such that the sum of the remainder sequence is greater than 100N. -/
theorem exists_large_remainder_sum : ∃ N a : ℕ+, 
  (remainderSequence N a).sum > 100 * N.val := by
  sorry

end exists_large_remainder_sum_l2803_280351


namespace simplify_expression_l2803_280303

theorem simplify_expression (x : ℝ) (h : x ≠ -1) :
  (x - 1 - 8 / (x + 1)) / ((x + 3) / (x + 1)) = x - 3 := by
  sorry

end simplify_expression_l2803_280303


namespace total_soccer_balls_l2803_280314

/-- Given the following conditions:
  - The school purchased 10 boxes of soccer balls
  - Each box contains 8 packages
  - Each package has 13 soccer balls
  Prove that the total number of soccer balls purchased is 1040 -/
theorem total_soccer_balls (num_boxes : ℕ) (packages_per_box : ℕ) (balls_per_package : ℕ)
  (h1 : num_boxes = 10)
  (h2 : packages_per_box = 8)
  (h3 : balls_per_package = 13) :
  num_boxes * packages_per_box * balls_per_package = 1040 := by
  sorry

end total_soccer_balls_l2803_280314


namespace unique_congruence_in_range_l2803_280345

theorem unique_congruence_in_range : ∃! n : ℤ, 10 ≤ n ∧ n ≤ 15 ∧ n ≡ 12345 [ZMOD 7] := by
  sorry

end unique_congruence_in_range_l2803_280345


namespace special_heptagon_perturbation_l2803_280321

/-- A point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A heptagon represented by its vertices -/
structure Heptagon :=
  (vertices : Fin 7 → Point)

/-- Predicate to check if a heptagon is convex -/
def is_convex (h : Heptagon) : Prop := sorry

/-- Predicate to check if three lines intersect at a single point -/
def intersect_at_point (l1 l2 l3 : Point × Point) (p : Point) : Prop := sorry

/-- Predicate to check if a heptagon is special -/
def is_special (h : Heptagon) : Prop :=
  ∃ (i j k : Fin 7) (p : Point),
    i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    intersect_at_point
      (h.vertices i, h.vertices ((i + 3) % 7))
      (h.vertices j, h.vertices ((j + 3) % 7))
      (h.vertices k, h.vertices ((k + 3) % 7))
      p

/-- Definition of a small perturbation -/
def small_perturbation (h1 h2 : Heptagon) (ε : ℝ) : Prop :=
  ∃ (i : Fin 7),
    ∀ (j : Fin 7),
      if i = j then
        (h1.vertices j).x - ε < (h2.vertices j).x ∧ (h2.vertices j).x < (h1.vertices j).x + ε ∧
        (h1.vertices j).y - ε < (h2.vertices j).y ∧ (h2.vertices j).y < (h1.vertices j).y + ε
      else
        h1.vertices j = h2.vertices j

/-- The main theorem -/
theorem special_heptagon_perturbation (h : Heptagon) (hconv : is_convex h) (hspec : is_special h) :
  ∃ (h' : Heptagon) (ε : ℝ), ε > 0 ∧ small_perturbation h h' ε ∧ is_convex h' ∧ ¬is_special h' :=
sorry

end special_heptagon_perturbation_l2803_280321


namespace min_cos_C_in_special_triangle_l2803_280375

theorem min_cos_C_in_special_triangle (A B C : ℝ) (h1 : 0 < A ∧ A < π)
  (h2 : 0 < B ∧ B < π) (h3 : 0 < C ∧ C < π) (h4 : A + B + C = π)
  (h5 : ∃ k : ℝ, (1 / Real.tan A) + k = 2 / Real.tan C ∧
                 2 / Real.tan C + k = 1 / Real.tan B) :
  ∃ (cosC : ℝ), cosC = Real.cos C ∧ cosC ≥ 1/3 ∧
  ∀ (cosC' : ℝ), cosC' = Real.cos C → cosC' ≥ 1/3 :=
by sorry

end min_cos_C_in_special_triangle_l2803_280375


namespace max_x5_value_l2803_280324

theorem max_x5_value (x₁ x₂ x₃ x₄ x₅ : ℕ) 
  (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧ x₅ > 0)
  (h_eq : x₁ + x₂ + x₃ + x₄ + x₅ = x₁ * x₂ * x₃ * x₄ * x₅) :
  x₅ ≤ 5 := by
  sorry

#check max_x5_value

end max_x5_value_l2803_280324


namespace philatelist_stamps_problem_l2803_280362

theorem philatelist_stamps_problem :
  ∃! x : ℕ,
    x % 2 = 1 ∧
    x % 3 = 1 ∧
    x % 5 = 3 ∧
    x % 9 = 7 ∧
    150 < x ∧
    x ≤ 300 ∧
    x = 223 := by sorry

end philatelist_stamps_problem_l2803_280362


namespace amount_owed_l2803_280353

-- Define the rate per room
def rate_per_room : ℚ := 11 / 2

-- Define the number of rooms cleaned
def rooms_cleaned : ℚ := 7 / 3

-- Theorem statement
theorem amount_owed : rate_per_room * rooms_cleaned = 77 / 6 := by
  sorry

end amount_owed_l2803_280353


namespace regular_tetrahedron_unordered_pairs_l2803_280319

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  /-- The number of edges in a regular tetrahedron -/
  num_edges : ℕ
  /-- The property that any two edges determine the same plane -/
  edges_same_plane : Unit

/-- The number of unordered pairs of edges in a regular tetrahedron -/
def num_unordered_pairs (t : RegularTetrahedron) : ℕ :=
  (t.num_edges * (t.num_edges - 1)) / 2

/-- Theorem stating that the number of unordered pairs of edges in a regular tetrahedron is 15 -/
theorem regular_tetrahedron_unordered_pairs :
  ∀ t : RegularTetrahedron, t.num_edges = 6 → num_unordered_pairs t = 15 :=
by
  sorry


end regular_tetrahedron_unordered_pairs_l2803_280319


namespace some_pens_not_vens_l2803_280328

-- Define the universe
variable (U : Type)

-- Define the predicates
variable (Pen Den Ven : U → Prop)

-- Define the hypotheses
variable (h1 : ∀ x, Pen x → Den x)
variable (h2 : ∃ x, Den x ∧ ¬Ven x)

-- State the theorem
theorem some_pens_not_vens : ∃ x, Pen x ∧ ¬Ven x :=
sorry

end some_pens_not_vens_l2803_280328


namespace at_most_two_rational_points_l2803_280326

/-- A point in the 2D plane -/
structure Point where
  x : ℚ
  y : ℚ

/-- A circle in the 2D plane -/
structure Circle where
  center_x : ℚ
  center_y : ℝ
  radius : ℝ

/-- A point is on a circle if it satisfies the circle equation -/
def Point.onCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center_x)^2 + (p.y - c.center_y)^2 = c.radius^2

/-- The main theorem: there are at most two rational points on a circle with irrational y-coordinate of the center -/
theorem at_most_two_rational_points (c : Circle) 
    (h : Irrational c.center_y) :
    ∃ (p1 p2 : Point), ∀ (p : Point), 
      p.onCircle c → p = p1 ∨ p = p2 := by
  sorry

end at_most_two_rational_points_l2803_280326


namespace cos_alpha_minus_pi_l2803_280339

theorem cos_alpha_minus_pi (α : Real) 
  (h1 : π / 2 < α) 
  (h2 : α < π) 
  (h3 : 3 * Real.sin (2 * α) = 2 * Real.cos α) : 
  Real.cos (α - π) = 2 * Real.sqrt 2 / 3 := by
sorry

end cos_alpha_minus_pi_l2803_280339


namespace f_continuity_and_discontinuity_l2803_280393

-- Define the piecewise function
noncomputable def f (x : ℝ) : ℝ :=
  if x < -3 then (x^2 + 3*x - 1) / (x + 2)
  else if x ≤ 4 then (x + 2)^2
  else 9*x + 1

-- Define continuity at a point
def continuous_at (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - a| < δ → |f x - f a| < ε

-- Define left and right limits
def has_limit_at_left (f : ℝ → ℝ) (a : ℝ) (L : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x, a - δ < x ∧ x < a → |f x - L| < ε

def has_limit_at_right (f : ℝ → ℝ) (a : ℝ) (L : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x, a < x ∧ x < a + δ → |f x - L| < ε

-- Define jump discontinuity
def jump_discontinuity (f : ℝ → ℝ) (a : ℝ) (jump : ℝ) : Prop :=
  ∃ L₁ L₂, has_limit_at_left f a L₁ ∧ has_limit_at_right f a L₂ ∧ L₂ - L₁ = jump

-- Theorem statement
theorem f_continuity_and_discontinuity :
  continuous_at f (-3) ∧ jump_discontinuity f 4 1 :=
sorry

end f_continuity_and_discontinuity_l2803_280393


namespace fourth_player_wins_probability_l2803_280317

def roll_probability : ℚ := 1 / 6

def other_roll_probability : ℚ := 1 - roll_probability

def num_players : ℕ := 4

def first_cycle_probability : ℚ := (other_roll_probability ^ (num_players - 1)) * roll_probability

def cycle_continuation_probability : ℚ := other_roll_probability ^ num_players

theorem fourth_player_wins_probability :
  let a := first_cycle_probability
  let r := cycle_continuation_probability
  (a / (1 - r)) = 125 / 671 := by sorry

end fourth_player_wins_probability_l2803_280317


namespace min_m_value_l2803_280308

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin x ^ 2 + 4 * Real.sqrt 3 * Real.sin x * Real.cos x + 5

theorem min_m_value (m : ℝ) :
  (∃ x, x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x ≤ m) →
  m ≥ 5 :=
by sorry

end min_m_value_l2803_280308


namespace perpendicular_line_equation_l2803_280322

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_line_equation (given_line : Line) (point : Point) :
  given_line = Line.mk 2 (-3) 4 →
  point = Point.mk (-1) 2 →
  ∃ (result_line : Line),
    result_line.perpendicular given_line ∧
    point.liesOn result_line ∧
    result_line = Line.mk 3 2 (-1) := by
  sorry


end perpendicular_line_equation_l2803_280322


namespace fruit_difference_l2803_280300

theorem fruit_difference (total : ℕ) (apples : ℕ) : 
  total = 913 → apples = 514 → apples - (total - apples) = 115 := by
  sorry

end fruit_difference_l2803_280300


namespace female_red_ants_percentage_l2803_280378

/-- Given an ant colony where 85% of the population is red and 46.75% of the total population
    are male red ants, prove that 45% of the red ants are females. -/
theorem female_red_ants_percentage
  (total_population : ℝ)
  (red_ants_percentage : ℝ)
  (male_red_ants_percentage : ℝ)
  (h1 : red_ants_percentage = 85)
  (h2 : male_red_ants_percentage = 46.75)
  (h3 : total_population > 0) :
  let total_red_ants := red_ants_percentage * total_population / 100
  let male_red_ants := male_red_ants_percentage * total_population / 100
  let female_red_ants := total_red_ants - male_red_ants
  female_red_ants / total_red_ants * 100 = 45 :=
by
  sorry


end female_red_ants_percentage_l2803_280378


namespace x_value_proof_l2803_280343

theorem x_value_proof (x y : ℝ) 
  (eq1 : x^2 - 4*x + y = 0) 
  (eq2 : y = 4) : 
  x = 2 := by
sorry

end x_value_proof_l2803_280343


namespace cube_edges_sum_l2803_280309

/-- Given a cube-shaped toy made up of 27 small cubes, with the total length of all edges
    of the large cube being 82.8 cm, prove that the sum of the length of one edge of the
    large cube and one edge of a small cube is 9.2 cm. -/
theorem cube_edges_sum (total_edge_length : ℝ) (num_small_cubes : ℕ) :
  total_edge_length = 82.8 ∧ num_small_cubes = 27 →
  ∃ (large_edge small_edge : ℝ),
    large_edge = total_edge_length / 12 ∧
    small_edge = large_edge / 3 ∧
    large_edge + small_edge = 9.2 :=
by sorry

end cube_edges_sum_l2803_280309


namespace remainder_sum_factorials_60_l2803_280356

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem remainder_sum_factorials_60 (h : ∀ k ≥ 5, 15 ∣ factorial k) :
  sum_factorials 60 % 15 = (factorial 1 + factorial 2 + factorial 3 + factorial 4) % 15 := by
  sorry

end remainder_sum_factorials_60_l2803_280356


namespace grass_sheet_cost_per_cubic_meter_l2803_280366

/-- The cost of a grass sheet per cubic meter, given the area of a playground,
    the depth of the grass sheet, and the total cost to cover the playground. -/
theorem grass_sheet_cost_per_cubic_meter
  (area : ℝ) (depth_cm : ℝ) (total_cost : ℝ)
  (h_area : area = 5900)
  (h_depth : depth_cm = 1)
  (h_total_cost : total_cost = 165.2) :
  total_cost / (area * depth_cm / 100) = 2.8 := by
  sorry

end grass_sheet_cost_per_cubic_meter_l2803_280366


namespace sqrt_equation_unique_solution_l2803_280348

theorem sqrt_equation_unique_solution :
  ∃! x : ℝ, Real.sqrt (x + 15) - 7 / Real.sqrt (x + 15) = 6 :=
by
  -- The proof would go here
  sorry

end sqrt_equation_unique_solution_l2803_280348


namespace ounces_per_cup_ounces_per_cup_is_eight_l2803_280325

/-- The number of ounces in a cup, given Cassie's water consumption habits -/
theorem ounces_per_cup : ℕ :=
  let cups_per_day : ℕ := 12
  let bottle_capacity : ℕ := 16
  let refills_per_day : ℕ := 6
  (refills_per_day * bottle_capacity) / cups_per_day

/-- Proof that the number of ounces in a cup is 8 -/
theorem ounces_per_cup_is_eight : ounces_per_cup = 8 := by
  sorry

end ounces_per_cup_ounces_per_cup_is_eight_l2803_280325


namespace square_diagonal_length_l2803_280338

/-- The diagonal length of a square with area 72 square meters is 12 meters. -/
theorem square_diagonal_length (area : ℝ) (side : ℝ) (diagonal : ℝ) : 
  area = 72 → 
  area = side ^ 2 → 
  diagonal ^ 2 = 2 * side ^ 2 → 
  diagonal = 12 := by
  sorry


end square_diagonal_length_l2803_280338


namespace perpendicular_vectors_m_value_l2803_280364

/-- Given two 2D vectors a and b, if a is perpendicular to (a + m*b), then m = 2/5 -/
theorem perpendicular_vectors_m_value (a b : ℝ × ℝ) (m : ℝ) 
  (ha : a = (1, -1)) 
  (hb : b = (-2, 3)) 
  (h_perp : a.1 * (a.1 + m * b.1) + a.2 * (a.2 + m * b.2) = 0) : 
  m = 2/5 := by
sorry

end perpendicular_vectors_m_value_l2803_280364


namespace point_on_curve_l2803_280349

/-- Curve C is defined by the parametric equations x = 4t² and y = t -/
def curve_C (t : ℝ) : ℝ × ℝ := (4 * t^2, t)

/-- Point P has coordinates (m, 2) -/
def point_P (m : ℝ) : ℝ × ℝ := (m, 2)

/-- Theorem: If point P(m, 2) lies on curve C, then m = 16 -/
theorem point_on_curve (m : ℝ) : 
  (∃ t : ℝ, curve_C t = point_P m) → m = 16 := by
  sorry

end point_on_curve_l2803_280349


namespace painted_faces_count_l2803_280391

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : ℕ := n

/-- Represents a painted cube -/
structure PaintedCube (n : ℕ) extends Cube n where
  is_painted : Bool := true

/-- Calculates the number of smaller cubes with at least two painted faces -/
def cubes_with_two_or_more_painted_faces (c : PaintedCube 4) : ℕ :=
  sorry

/-- Theorem stating that a 4x4x4 painted cube cut into 1x1x1 cubes has 32 smaller cubes with at least two painted faces -/
theorem painted_faces_count (c : PaintedCube 4) : 
  cubes_with_two_or_more_painted_faces c = 32 := by sorry

end painted_faces_count_l2803_280391


namespace orthogonal_vectors_l2803_280383

/-- Two vectors in ℝ² are orthogonal if their dot product is zero -/
def orthogonal (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- The vector a -/
def a : ℝ × ℝ := (4, 2)

/-- The vector b -/
def b (y : ℝ) : ℝ × ℝ := (6, y)

/-- Theorem: If a and b are orthogonal, then y = -12 -/
theorem orthogonal_vectors (y : ℝ) :
  orthogonal a (b y) → y = -12 := by
  sorry

end orthogonal_vectors_l2803_280383


namespace board_numbers_l2803_280337

theorem board_numbers (N : ℕ) (numbers : Finset ℝ) : 
  (N ≥ 9) →
  (Finset.card numbers = N) →
  (∀ x ∈ numbers, 0 ≤ x ∧ x < 1) →
  (∀ subset : Finset ℝ, subset ⊆ numbers → Finset.card subset = 8 → 
    ∃ y ∈ numbers, y ∉ subset ∧ 
    ∃ z : ℤ, (Finset.sum subset (λ i => i) + y = z)) →
  N = 9 := by
sorry

end board_numbers_l2803_280337


namespace fraction_equality_l2803_280334

theorem fraction_equality (x y z : ℝ) (h : (x - y) / (z - y) = -10) :
  (x - z) / (y - z) = 11 := by sorry

end fraction_equality_l2803_280334


namespace power_sum_five_l2803_280367

theorem power_sum_five (x : ℝ) (h : x + 1/x = 5) : x^5 + 1/x^5 = 2520 := by
  sorry

end power_sum_five_l2803_280367


namespace number_equation_l2803_280346

theorem number_equation (x : ℝ) : 3 * x - 1 = 2 * x ↔ x = 1 := by
  sorry

end number_equation_l2803_280346


namespace parallel_vectors_condition_solution_set_correct_l2803_280302

/-- Two vectors are parallel if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- The vectors a and b as functions of x -/
def a (x : ℝ) : ℝ × ℝ := (1, x)
def b (x : ℝ) : ℝ × ℝ := (x - 1, 2)

/-- Theorem stating the conditions for a and b to be parallel -/
theorem parallel_vectors_condition :
  ∀ x : ℝ, are_parallel (a x) (b x) ↔ x = 2 ∨ x = -1 :=
by
  sorry

/-- The solution set for x -/
def solution_set : Set ℝ := {2, -1}

/-- Theorem stating that the solution set is correct -/
theorem solution_set_correct :
  ∀ x : ℝ, x ∈ solution_set ↔ are_parallel (a x) (b x) :=
by
  sorry

end parallel_vectors_condition_solution_set_correct_l2803_280302


namespace double_average_l2803_280327

theorem double_average (n : ℕ) (original_avg : ℝ) (h1 : n = 25) (h2 : original_avg = 70) :
  let total_marks := n * original_avg
  let doubled_marks := 2 * total_marks
  let new_avg := doubled_marks / n
  new_avg = 140 := by
sorry

end double_average_l2803_280327


namespace abs_reciprocal_of_neg_three_halves_l2803_280394

theorem abs_reciprocal_of_neg_three_halves :
  |(((-1 : ℚ) - (1 : ℚ) / (2 : ℚ))⁻¹)| = (2 : ℚ) / (3 : ℚ) := by
  sorry

end abs_reciprocal_of_neg_three_halves_l2803_280394


namespace sum_coordinates_of_B_l2803_280312

/-- Given that M(4,4) is the midpoint of AB and A has coordinates (8,4),
    prove that the sum of the coordinates of B is 4. -/
theorem sum_coordinates_of_B (A B M : ℝ × ℝ) : 
  M = (4, 4) →
  A = (8, 4) →
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  B.1 + B.2 = 4 := by
  sorry

end sum_coordinates_of_B_l2803_280312


namespace even_function_iff_a_eq_zero_l2803_280379

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x

theorem even_function_iff_a_eq_zero (a : ℝ) :
  (∀ x : ℝ, f a x = f a (-x)) ↔ a = 0 := by
  sorry

end even_function_iff_a_eq_zero_l2803_280379


namespace factorial_fraction_equality_l2803_280395

theorem factorial_fraction_equality : (4 * Nat.factorial 6 + 20 * Nat.factorial 5) / Nat.factorial 7 = 22 / 21 := by
  sorry

end factorial_fraction_equality_l2803_280395


namespace five_lines_max_sections_l2803_280372

/-- The maximum number of sections created by drawing n line segments through a rectangle,
    given that the first line segment separates the rectangle into 2 sections. -/
def max_sections (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 2
  else 2 + (n - 1) * n / 2

/-- Theorem: The maximum number of sections created by drawing 5 line segments
    through a rectangle is 16, given that the first line segment separates
    the rectangle into 2 sections. -/
theorem five_lines_max_sections :
  max_sections 5 = 16 := by
  sorry

end five_lines_max_sections_l2803_280372


namespace negation_of_exists_lt_is_forall_ge_l2803_280306

theorem negation_of_exists_lt_is_forall_ge :
  ¬(∃ x : ℝ, x^2 - 2*x + 1 < 0) ↔ (∀ x : ℝ, x^2 - 2*x + 1 ≥ 0) :=
by sorry

end negation_of_exists_lt_is_forall_ge_l2803_280306


namespace money_ratio_proof_l2803_280301

/-- Proves that the ratio of Nataly's money to Raquel's money is 3:1 given the problem conditions -/
theorem money_ratio_proof (tom nataly raquel : ℚ) : 
  tom = (1 / 4) * nataly →  -- Tom has 1/4 as much money as Nataly
  nataly = raquel * (nataly / raquel) →  -- Nataly has a certain multiple of Raquel's money
  tom + raquel + nataly = 190 →  -- Total money is $190
  raquel = 40 →  -- Raquel has $40
  nataly / raquel = 3 := by
sorry

end money_ratio_proof_l2803_280301


namespace inequality_proof_l2803_280370

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  1 / (a^3 + b*c) + 1 / (b^3 + c*a) + 1 / (c^3 + a*b) ≤ (a*b + b*c + c*a)^2 / 6 := by
  sorry

end inequality_proof_l2803_280370


namespace intersection_value_l2803_280380

theorem intersection_value (a : ℝ) : 
  let M : Set ℝ := {a^2, a+1, -3}
  let N : Set ℝ := {a-3, 2*a-1, a^2+1}
  M ∩ N = {-3} → a = -1 := by
sorry

end intersection_value_l2803_280380


namespace original_price_calculation_l2803_280390

/-- Proves that if an article is sold for $130 with a 30% gain, then its original price was $100. -/
theorem original_price_calculation (sale_price : ℝ) (gain_percent : ℝ) : 
  sale_price = 130 ∧ gain_percent = 30 → 
  sale_price = (100 : ℝ) * (1 + gain_percent / 100) := by
sorry

end original_price_calculation_l2803_280390


namespace quadratic_function_properties_l2803_280381

def f (x : ℝ) := -2 * (x + 3)^2 + 1

theorem quadratic_function_properties :
  let opens_downward := ∀ x y : ℝ, f ((x + y) / 2) > (f x + f y) / 2
  let axis_of_symmetry := 3
  let vertex := (3, 1)
  let decreases_after_three := ∀ x₁ x₂ : ℝ, x₁ > 3 → x₂ > x₁ → f x₂ < f x₁
  
  (opens_downward ∧ ¬(f axis_of_symmetry = f (-axis_of_symmetry)) ∧
   ¬(f (vertex.1) = vertex.2) ∧ decreases_after_three) :=
by sorry

end quadratic_function_properties_l2803_280381


namespace arithmetic_sequence_properties_max_sum_at_25_l2803_280386

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  second_eighth_sum : a 2 + a 8 = 82
  sum_equality : S 41 = S 9

/-- Properties of the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∃ d : ℝ, ∀ n : ℕ, seq.a n = 51 - 2 * n) ∧
  (∃ n : ℕ, seq.S n = 625 ∧ ∀ m : ℕ, seq.S m ≤ seq.S n) := by
  sorry

/-- The maximum value of S_n occurs when n = 25 -/
theorem max_sum_at_25 (seq : ArithmeticSequence) :
  seq.S 25 = 625 ∧ ∀ n : ℕ, seq.S n ≤ seq.S 25 := by
  sorry

end arithmetic_sequence_properties_max_sum_at_25_l2803_280386


namespace largest_multiple_of_15_less_than_500_l2803_280387

theorem largest_multiple_of_15_less_than_500 : 
  ∃ n : ℕ, n * 15 = 495 ∧ 
  495 < 500 ∧ 
  ∀ m : ℕ, m * 15 < 500 → m * 15 ≤ 495 := by
sorry

end largest_multiple_of_15_less_than_500_l2803_280387


namespace inequality_solution_inequality_system_solution_l2803_280371

-- Part 1: Inequality solution
theorem inequality_solution (x : ℝ) :
  (2 * x - 1) / 2 ≥ 1 - (x + 1) / 3 ↔ x ≥ 7 / 8 :=
by sorry

-- Part 2: System of inequalities solution
theorem inequality_system_solution (x : ℝ) :
  (-2 * x ≤ -3 ∧ x / 2 < 2) ↔ (3 / 2 ≤ x ∧ x < 4) :=
by sorry

end inequality_solution_inequality_system_solution_l2803_280371


namespace point_on_line_l2803_280352

/-- A point (x, y) lies on the line passing through (2, -4) and (8, 16) if and only if y = (10/3)x - 32/3 -/
theorem point_on_line (x y : ℝ) : 
  (y = (10/3)*x - 32/3) ↔ 
  (∃ t : ℝ, x = 2 + 6*t ∧ y = -4 + 20*t) :=
by sorry

end point_on_line_l2803_280352


namespace arrangements_combinations_ratio_l2803_280315

/-- Number of arrangements of n items taken r at a time -/
def A (n : ℕ) (r : ℕ) : ℚ := (Nat.factorial n) / (Nat.factorial (n - r))

/-- Number of combinations of n items taken r at a time -/
def C (n : ℕ) (r : ℕ) : ℚ := (Nat.factorial n) / ((Nat.factorial r) * (Nat.factorial (n - r)))

theorem arrangements_combinations_ratio : (A 7 2) / (C 10 2) = 14 / 15 := by
  sorry

end arrangements_combinations_ratio_l2803_280315


namespace max_candies_consumed_max_candies_is_1225_l2803_280336

/-- The number of initial ones on the board -/
def initial_ones : ℕ := 50

/-- The number of minutes the process continues -/
def total_minutes : ℕ := 50

/-- The number of candies consumed is equal to the number of edges in a complete graph -/
theorem max_candies_consumed (n : ℕ) (h : n = initial_ones) :
  (n * (n - 1)) / 2 = total_minutes * (total_minutes - 1) / 2 := by sorry

/-- The maximum number of candies consumed after the process -/
def max_candies : ℕ := (initial_ones * (initial_ones - 1)) / 2

/-- Proof that the maximum number of candies consumed is 1225 -/
theorem max_candies_is_1225 : max_candies = 1225 := by sorry

end max_candies_consumed_max_candies_is_1225_l2803_280336


namespace nina_taller_than_lena_probability_l2803_280359

-- Define the set of friends
inductive Friend
| Masha
| Nina
| Lena
| Olya

-- Define the height relation
def taller_than (a b : Friend) : Prop := sorry

-- Define the conditions
axiom different_heights :
  ∀ (a b : Friend), a ≠ b → (taller_than a b ∨ taller_than b a)

axiom nina_shorter_than_masha :
  taller_than Friend.Masha Friend.Nina

axiom lena_taller_than_olya :
  taller_than Friend.Lena Friend.Olya

-- Define the probability function
noncomputable def probability (event : Prop) : ℝ := sorry

-- Theorem to prove
theorem nina_taller_than_lena_probability :
  probability (taller_than Friend.Nina Friend.Lena) = 0 := by sorry

end nina_taller_than_lena_probability_l2803_280359


namespace no_infinite_arithmetic_progression_in_squares_l2803_280305

theorem no_infinite_arithmetic_progression_in_squares :
  ¬ ∃ (a d : ℕ) (f : ℕ → ℕ),
    (∀ n, f n < f (n + 1)) ∧
    (∀ n, ∃ k, f n = k^2) ∧
    (∀ n, f (n + 1) - f n = d) :=
sorry

end no_infinite_arithmetic_progression_in_squares_l2803_280305


namespace local_extrema_sum_l2803_280310

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 27

-- State the theorem
theorem local_extrema_sum (a b : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-1 - ε) (-1 + ε), f a b (-1) ≥ f a b x) ∧
  (∃ ε > 0, ∀ x ∈ Set.Ioo (3 - ε) (3 + ε), f a b 3 ≤ f a b x) →
  a + b = -12 := by
  sorry

end local_extrema_sum_l2803_280310


namespace iris_jacket_purchase_l2803_280397

theorem iris_jacket_purchase (jacket_price shorts_price pants_price : ℕ)
  (shorts_quantity pants_quantity : ℕ) (total_spent : ℕ) :
  jacket_price = 10 →
  shorts_price = 6 →
  pants_price = 12 →
  shorts_quantity = 2 →
  pants_quantity = 4 →
  total_spent = 90 →
  ∃ (jacket_quantity : ℕ), 
    jacket_quantity * jacket_price + 
    shorts_quantity * shorts_price + 
    pants_quantity * pants_price = total_spent ∧
    jacket_quantity = 3 :=
by sorry

end iris_jacket_purchase_l2803_280397


namespace desk_purchase_price_l2803_280358

/-- Proves that the purchase price of a desk is $100 given the specified conditions -/
theorem desk_purchase_price (purchase_price selling_price : ℝ) : 
  selling_price = purchase_price + 0.5 * selling_price →
  selling_price - purchase_price = 100 →
  purchase_price = 100 := by
sorry

end desk_purchase_price_l2803_280358


namespace correct_scores_theorem_l2803_280382

/-- Represents a class with exam scores -/
structure ExamClass where
  studentCount : Nat
  initialAverage : ℝ
  initialVariance : ℝ
  studentAInitialScore : ℝ
  studentAActualScore : ℝ
  studentBInitialScore : ℝ
  studentBActualScore : ℝ

/-- Calculates the new average and variance after correcting two scores -/
def correctScores (c : ExamClass) : ℝ × ℝ :=
  let newAverage := c.initialAverage
  let newVariance := c.initialVariance - 25
  (newAverage, newVariance)

theorem correct_scores_theorem (c : ExamClass) 
  (h1 : c.studentCount = 48)
  (h2 : c.initialAverage = 70)
  (h3 : c.initialVariance = 75)
  (h4 : c.studentAInitialScore = 50)
  (h5 : c.studentAActualScore = 80)
  (h6 : c.studentBInitialScore = 100)
  (h7 : c.studentBActualScore = 70) :
  correctScores c = (70, 50) := by
  sorry

end correct_scores_theorem_l2803_280382


namespace carol_achieves_target_average_l2803_280361

-- Define the inverse relationship between exercise time and test score
def inverse_relation (exercise_time : ℝ) (test_score : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ exercise_time * test_score = k

-- Define Carol's first test results
def first_test_exercise_time : ℝ := 45
def first_test_score : ℝ := 80

-- Define Carol's target average score
def target_average_score : ℝ := 85

-- Define Carol's exercise time for the second test
def second_test_exercise_time : ℝ := 40

-- Theorem to prove
theorem carol_achieves_target_average :
  inverse_relation first_test_exercise_time first_test_score →
  inverse_relation second_test_exercise_time ((2 * target_average_score * 2) - first_test_score) →
  (first_test_score + ((2 * target_average_score * 2) - first_test_score)) / 2 = target_average_score :=
by
  sorry

end carol_achieves_target_average_l2803_280361


namespace tea_bags_in_box_l2803_280384

theorem tea_bags_in_box : ∀ n : ℕ,
  (2 * n ≤ 41 ∧ 41 ≤ 3 * n) ∧
  (2 * n ≤ 58 ∧ 58 ≤ 3 * n) →
  n = 20 := by
sorry

end tea_bags_in_box_l2803_280384


namespace odd_function_with_property_M_even_function_with_property_M_l2803_280340

def has_property_M (f : ℝ → ℝ) (A : Set ℝ) :=
  ∃ c : ℝ, ∀ x ∈ A, Real.exp x * (f x - Real.exp x) = c

def is_odd (f : ℝ → ℝ) :=
  ∀ x, f (-x) = -f x

def is_even (f : ℝ → ℝ) :=
  ∀ x, f (-x) = f x

theorem odd_function_with_property_M (f : ℝ → ℝ) (h1 : is_odd f) (h2 : has_property_M f Set.univ) :
  ∀ x, f x = Real.exp x - 1 / Real.exp x := by sorry

theorem even_function_with_property_M (g : ℝ → ℝ) (h1 : is_even g)
    (h2 : has_property_M g (Set.Icc (-1) 1))
    (h3 : ∀ x ∈ Set.Icc (-1) 1, g (2 * x) - 2 * Real.exp 1 * g x + n > 0) :
  n > Real.exp 2 + 2 := by sorry

end odd_function_with_property_M_even_function_with_property_M_l2803_280340


namespace school_merger_ratio_l2803_280354

theorem school_merger_ratio (a b : ℕ) (ha : a > 0) (hb : b > 0) : 
  (8 * a) / (7 * a) = 8 / 7 →
  (30 * b) / (31 * b) = 30 / 31 →
  (8 * a + 30 * b) / (7 * a + 31 * b) = 27 / 26 →
  (8 * a + 7 * a) / (30 * b + 31 * b) = 27 / 26 :=
by sorry

end school_merger_ratio_l2803_280354


namespace museum_discount_percentage_l2803_280307

/-- Represents the discount percentage for people 18 years old or younger -/
def discount_percentage : ℝ := 30

/-- Represents the regular ticket cost -/
def regular_ticket_cost : ℝ := 10

/-- Represents Dorothy's initial amount of money -/
def dorothy_initial_money : ℝ := 70

/-- Represents Dorothy's remaining money after the trip -/
def dorothy_remaining_money : ℝ := 26

/-- Represents the number of people in Dorothy's family -/
def family_size : ℕ := 5

/-- Represents the number of adults (paying full price) in Dorothy's family -/
def num_adults : ℕ := 3

/-- Represents the number of children (eligible for discount) in Dorothy's family -/
def num_children : ℕ := 2

theorem museum_discount_percentage :
  let total_spent := dorothy_initial_money - dorothy_remaining_money
  let adult_cost := num_adults * regular_ticket_cost
  let children_cost := total_spent - adult_cost
  let discounted_ticket_cost := regular_ticket_cost * (1 - discount_percentage / 100)
  children_cost = num_children * discounted_ticket_cost :=
by sorry

#check museum_discount_percentage

end museum_discount_percentage_l2803_280307


namespace elephant_ratio_is_three_l2803_280368

/-- The number of elephants at We Preserve For Future park -/
def we_preserve_elephants : ℕ := 70

/-- The total number of elephants in both parks -/
def total_elephants : ℕ := 280

/-- The ratio of elephants at Gestures For Good park to We Preserve For Future park -/
def elephant_ratio : ℚ := (total_elephants - we_preserve_elephants) / we_preserve_elephants

theorem elephant_ratio_is_three : elephant_ratio = 3 := by
  sorry

end elephant_ratio_is_three_l2803_280368


namespace negation_of_implication_l2803_280388

theorem negation_of_implication (a b : ℝ) : 
  ¬(ab = 0 → a = 0 ∨ b = 0) ↔ (ab ≠ 0 → a ≠ 0 ∧ b ≠ 0) := by
  sorry

end negation_of_implication_l2803_280388


namespace scooter_price_calculation_l2803_280323

/-- Calculates the selling price of a scooter given its purchase price, repair costs, and gain percentage. -/
def scooter_selling_price (purchase_price repair_costs : ℚ) (gain_percent : ℚ) : ℚ :=
  let total_cost := purchase_price + repair_costs
  let gain := gain_percent / 100
  let profit := total_cost * gain
  total_cost + profit

/-- Theorem stating that the selling price of the scooter is $5800 given the specified conditions. -/
theorem scooter_price_calculation :
  scooter_selling_price 4700 800 (5454545454545454 / 100000000000000) = 5800 := by
  sorry

end scooter_price_calculation_l2803_280323


namespace union_complement_equality_l2803_280344

def U : Finset Nat := {0, 1, 2, 4, 6, 8}
def M : Finset Nat := {0, 4, 6}
def N : Finset Nat := {0, 1, 6}

theorem union_complement_equality : M ∪ (U \ N) = {0, 2, 4, 6, 8} := by sorry

end union_complement_equality_l2803_280344


namespace simplify_and_evaluate_evaluate_at_2023_l2803_280357

theorem simplify_and_evaluate (x : ℝ) : (x + 1)^2 - x * (x + 1) = x + 1 :=
  sorry

theorem evaluate_at_2023 : (2023 + 1)^2 - 2023 * (2023 + 1) = 2024 :=
  sorry

end simplify_and_evaluate_evaluate_at_2023_l2803_280357


namespace constant_term_value_l2803_280355

/-- The constant term in the expansion of (3x + 2/x)^8 -/
def constant_term : ℕ :=
  let binomial_coeff := (8 : ℕ).choose 4
  let x_power_term := 3^4 * 2^4
  binomial_coeff * x_power_term

/-- The constant term in the expansion of (3x + 2/x)^8 is 90720 -/
theorem constant_term_value : constant_term = 90720 := by
  sorry

#eval constant_term -- This will evaluate the constant term

end constant_term_value_l2803_280355


namespace winter_ball_attendance_l2803_280360

theorem winter_ball_attendance 
  (total_students : ℕ) 
  (ball_attendees : ℕ) 
  (girls : ℕ) 
  (boys : ℕ) : 
  total_students = 1500 →
  ball_attendees = 900 →
  girls + boys = total_students →
  (3 * girls / 4 : ℚ) + (2 * boys / 3 : ℚ) = ball_attendees →
  3 * girls / 4 = 900 :=
by sorry

end winter_ball_attendance_l2803_280360


namespace distribute_5_3_l2803_280335

/-- The number of ways to distribute n college graduates to k employers,
    with each employer receiving at least 1 graduate -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 college graduates to 3 employers,
    with each employer receiving at least 1 graduate, is 150 -/
theorem distribute_5_3 : distribute 5 3 = 150 := by sorry

end distribute_5_3_l2803_280335


namespace scientific_notation_410000_l2803_280365

theorem scientific_notation_410000 : 410000 = 4.1 * (10 ^ 5) := by
  sorry

end scientific_notation_410000_l2803_280365


namespace rectangle_diagonal_l2803_280350

/-- A rectangle with perimeter 14 cm and area 12 square cm has a diagonal of length 5 cm -/
theorem rectangle_diagonal (l w : ℝ) : 
  (2 * l + 2 * w = 14) →  -- Perimeter condition
  (l * w = 12) →          -- Area condition
  Real.sqrt (l^2 + w^2) = 5 := by
sorry

end rectangle_diagonal_l2803_280350


namespace product_remainder_by_ten_l2803_280373

theorem product_remainder_by_ten : (1734 * 5389 * 80607) % 10 = 2 := by
  sorry

end product_remainder_by_ten_l2803_280373


namespace quadratic_equation_solution_l2803_280398

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 + 6*x + 2
  ∃ x1 x2 : ℝ, x1 = -3 + Real.sqrt 7 ∧ x2 = -3 - Real.sqrt 7 ∧ f x1 = 0 ∧ f x2 = 0 :=
by
  sorry

end quadratic_equation_solution_l2803_280398


namespace perpendicular_lines_parallel_l2803_280318

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp : Line → Plane → Prop)

-- Define the parallel relation between two lines
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_parallel (a b : Line) (α : Plane) :
  perp a α → perp b α → parallel a b := by sorry

end perpendicular_lines_parallel_l2803_280318


namespace geometric_series_sum_l2803_280374

theorem geometric_series_sum (a b : ℝ) (h : ∑' n, a / b^n = 6) :
  ∑' n, 2*a / (a + b)^n = 12/7 := by sorry

end geometric_series_sum_l2803_280374


namespace geometric_series_sum_l2803_280342

/-- The sum of an infinite geometric series with first term 1 and common ratio 1/4 is 4/3 -/
theorem geometric_series_sum : 
  let a : ℝ := 1
  let r : ℝ := 1/4
  let S : ℝ := ∑' n, a * r^n
  S = 4/3 := by sorry

end geometric_series_sum_l2803_280342


namespace semicircle_radius_l2803_280399

theorem semicircle_radius (p : ℝ) (h : p = 108) : 
  ∃ r : ℝ, p = r * (Real.pi + 2) ∧ r = p / (Real.pi + 2) := by
  sorry

end semicircle_radius_l2803_280399


namespace index_difference_proof_l2803_280369

/-- Calculates the index for a subgroup within a larger group -/
def calculate_index (n k x : ℕ) : ℚ :=
  (n - k : ℚ) / n * (n - x : ℚ) / n

theorem index_difference_proof (n k x_f x_m : ℕ) 
  (h_n : n = 25)
  (h_k : k = 8)
  (h_x_f : x_f = 6)
  (h_x_m : x_m = 10) :
  calculate_index n k x_f - calculate_index n (n - k) x_m = 203 / 625 := by
  sorry

#eval calculate_index 25 8 6 - calculate_index 25 17 10

end index_difference_proof_l2803_280369


namespace four_Z_three_equals_37_l2803_280363

-- Define the Z operation
def Z (a b : ℕ) : ℕ := a^2 + a*b + b^2

-- Theorem to prove
theorem four_Z_three_equals_37 : Z 4 3 = 37 := by sorry

end four_Z_three_equals_37_l2803_280363


namespace no_solution_when_p_divides_x_l2803_280385

theorem no_solution_when_p_divides_x (p : ℕ) (hp : p.Prime) (hp_gt_3 : p > 3) :
  ∀ (x y : ℕ), x > 0 → y > 0 → p ∣ x → x^2 - 1 ≠ y^p := by
  sorry

end no_solution_when_p_divides_x_l2803_280385


namespace fraction_simplest_form_l2803_280313

theorem fraction_simplest_form (x y : ℝ) : 
  ¬∃ (a b : ℝ), (x - y) / (x^2 + y^2) = a / b ∧ (a ≠ x - y ∨ b ≠ x^2 + y^2) :=
sorry

end fraction_simplest_form_l2803_280313


namespace initial_markup_percentage_l2803_280396

theorem initial_markup_percentage (initial_price : ℝ) (price_increase : ℝ) : 
  initial_price = 36 →
  price_increase = 4 →
  (initial_price + price_increase) / (initial_price - initial_price * 0.8) = 2 →
  (initial_price - (initial_price - initial_price * 0.8)) / (initial_price - initial_price * 0.8) = 0.8 := by
  sorry

end initial_markup_percentage_l2803_280396
