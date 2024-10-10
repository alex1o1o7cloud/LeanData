import Mathlib

namespace town_trash_cans_l3684_368465

theorem town_trash_cans (street_cans : ℕ) (store_cans : ℕ) : 
  street_cans = 14 →
  store_cans = 2 * street_cans →
  street_cans + store_cans = 42 := by
sorry

end town_trash_cans_l3684_368465


namespace brothers_catch_up_time_l3684_368405

/-- The time taken for the older brother to catch up with the younger brother -/
def catchUpTime (olderTime youngerTime delay : ℚ) : ℚ :=
  let relativeSpeed := 1 / olderTime - 1 / youngerTime
  let distanceCovered := delay / youngerTime
  delay + distanceCovered / relativeSpeed

/-- Theorem stating the catch-up time for the given problem -/
theorem brothers_catch_up_time :
  catchUpTime 12 20 5 = 25/2 := by
  sorry

#eval catchUpTime 12 20 5

end brothers_catch_up_time_l3684_368405


namespace impossibleTransformation_l3684_368498

/-- Represents a pile of stones -/
structure Pile :=
  (count : ℕ)

/-- Represents the state of all piles -/
structure PileState :=
  (piles : List Pile)

/-- Allowed operations on piles -/
inductive Operation
  | Combine : Pile → Pile → Operation
  | Split : Pile → Operation

/-- Applies an operation to a pile state -/
def applyOperation (state : PileState) (op : Operation) : PileState :=
  sorry

/-- Checks if a pile state is the desired final state -/
def isFinalState (state : PileState) : Prop :=
  state.piles.length = 105 ∧ state.piles.all (fun p => p.count = 1)

/-- The main theorem -/
theorem impossibleTransformation :
  ∀ (operations : List Operation),
    let initialState : PileState := ⟨[⟨51⟩, ⟨49⟩, ⟨5⟩]⟩
    let finalState := operations.foldl applyOperation initialState
    ¬(isFinalState finalState) := by
  sorry

end impossibleTransformation_l3684_368498


namespace parking_fee_calculation_l3684_368401

/-- Calculates the parking fee based on the given fee structure and parking duration. -/
def parking_fee (initial_fee : ℕ) (additional_fee : ℕ) (initial_duration : ℕ) (increment : ℕ) (total_duration : ℕ) : ℕ :=
  let extra_duration := total_duration - initial_duration
  let extra_increments := (extra_duration + increment - 1) / increment
  initial_fee + additional_fee * extra_increments

/-- Theorem stating that the parking fee for 80 minutes is 1500 won given the specified fee structure. -/
theorem parking_fee_calculation : parking_fee 500 200 30 10 80 = 1500 := by
  sorry

#eval parking_fee 500 200 30 10 80

end parking_fee_calculation_l3684_368401


namespace original_number_equation_l3684_368450

/-- Given a number x, prove that when it's doubled, 15 is added, and the result is trebled, it equals 75 -/
theorem original_number_equation (x : ℝ) : 3 * (2 * x + 15) = 75 ↔ x = 5 := by
  sorry

end original_number_equation_l3684_368450


namespace pyramid_layers_l3684_368409

/-- Represents a pyramid with layers of sandstone blocks -/
structure Pyramid where
  total_blocks : ℕ
  layer_ratio : ℕ
  top_layer_blocks : ℕ

/-- Calculates the number of layers in a pyramid -/
def num_layers (p : Pyramid) : ℕ :=
  sorry

/-- Theorem stating that a pyramid with 40 blocks, 3:1 layer ratio, and single top block has 4 layers -/
theorem pyramid_layers (p : Pyramid) 
  (h1 : p.total_blocks = 40)
  (h2 : p.layer_ratio = 3)
  (h3 : p.top_layer_blocks = 1) :
  num_layers p = 4 := by
  sorry

end pyramid_layers_l3684_368409


namespace square_construction_theorem_l3684_368446

/-- A line in a plane -/
structure Line :=
  (point : ℝ × ℝ)
  (direction : ℝ × ℝ)

/-- A square in a plane -/
structure Square :=
  (center : ℝ × ℝ)
  (side_length : ℝ)
  (rotation : ℝ)

/-- Check if a point lies on a line -/
def point_on_line (p : ℝ × ℝ) (l : Line) : Prop := sorry

/-- Check if a line intersects a square (including its extensions) -/
def line_intersects_square (l : Line) (s : Square) : Prop := sorry

/-- The main theorem -/
theorem square_construction_theorem 
  (L : Line) 
  (A B C D : ℝ × ℝ) 
  (h_distinct : A ≠ B ∧ B ≠ C ∧ C ≠ D)
  (h_order : point_on_line A L ∧ point_on_line B L ∧ point_on_line C L ∧ point_on_line D L) :
  ∃ (S : Square), 
    (∃ (p q : ℝ × ℝ), line_intersects_square L S ∧ p ≠ q ∧ 
      ((p = A ∧ q = B) ∨ (p = B ∧ q = A))) ∧
    (∃ (r s : ℝ × ℝ), line_intersects_square L S ∧ r ≠ s ∧ 
      ((r = C ∧ s = D) ∨ (r = D ∧ s = C))) :=
sorry

end square_construction_theorem_l3684_368446


namespace polynomial_divisibility_l3684_368419

theorem polynomial_divisibility (p q : ℤ) : 
  (∀ x : ℝ, (x + 3) * (x - 2) ∣ (x^5 - 2*x^4 + 3*x^3 - p*x^2 + q*x + 12)) →
  p = -28 ∧ q = -74 := by
sorry

end polynomial_divisibility_l3684_368419


namespace wall_bricks_count_l3684_368407

/-- Represents the number of bricks in the wall -/
def total_bricks : ℕ := 720

/-- Time taken by the first bricklayer to complete the wall alone (in hours) -/
def time_worker1 : ℕ := 12

/-- Time taken by the second bricklayer to complete the wall alone (in hours) -/
def time_worker2 : ℕ := 15

/-- Productivity decrease when working together (in bricks per hour) -/
def productivity_decrease : ℕ := 12

/-- Time taken when both workers work together (in hours) -/
def time_together : ℕ := 6

/-- Theorem stating that the number of bricks in the wall is 720 -/
theorem wall_bricks_count :
  (total_bricks / time_worker1 + total_bricks / time_worker2 - productivity_decrease) * time_together = total_bricks := by
  sorry

end wall_bricks_count_l3684_368407


namespace colberts_treehouse_l3684_368462

theorem colberts_treehouse (total : ℕ) (storage : ℕ) (parents : ℕ) (store : ℕ) (friends : ℕ) : 
  total = 200 →
  storage = total / 4 →
  parents = total / 2 →
  store = 30 →
  total = storage + parents + store + friends →
  friends = 20 := by
sorry

end colberts_treehouse_l3684_368462


namespace ellipse_condition_l3684_368441

/-- Represents an ellipse with foci on the y-axis -/
def is_ellipse_y_axis (m n : ℝ) : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ m = 1 / (a ^ 2) ∧ n = 1 / (b ^ 2)

/-- The main theorem stating that m > n > 0 is necessary and sufficient for mx^2 + ny^2 = 1 
    to represent an ellipse with foci on the y-axis -/
theorem ellipse_condition (m n : ℝ) : 
  (m > n ∧ n > 0) ↔ is_ellipse_y_axis m n := by sorry

end ellipse_condition_l3684_368441


namespace time_expression_l3684_368436

/-- Given V = 3gt + V₀ and S = (3/2)gt² + V₀t + (1/2)at², where a is another constant acceleration,
    prove that t = 9gS / (2(V - V₀)² + 3V₀(V - V₀)) -/
theorem time_expression (V V₀ g a S t : ℝ) 
  (hV : V = 3 * g * t + V₀)
  (hS : S = (3/2) * g * t^2 + V₀ * t + (1/2) * a * t^2) :
  t = (9 * g * S) / (2 * (V - V₀)^2 + 3 * V₀ * (V - V₀)) :=
by sorry

end time_expression_l3684_368436


namespace arithmetic_sequence_problem_l3684_368486

/-- Given an arithmetic sequence {a_n} where a_5 = 8 and a_9 = 24, prove that a_4 = 4 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (h_a5 : a 5 = 8)
  (h_a9 : a 9 = 24) : 
  a 4 = 4 := by
sorry

end arithmetic_sequence_problem_l3684_368486


namespace system_of_equations_l3684_368432

theorem system_of_equations (x y a : ℝ) : 
  (2 * x + y = 2 * a + 1) → 
  (x + 2 * y = a - 1) → 
  (x - y = 4) → 
  (a = 2) := by
sorry

end system_of_equations_l3684_368432


namespace perpendicular_parallel_relations_l3684_368416

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Line → Prop)
variable (perpPlane : Line → Plane → Prop)
variable (para : Line → Line → Prop)
variable (paraPlane : Line → Plane → Prop)

-- Given: l is perpendicular to α
variable (l : Line) (α : Plane)
variable (h : perpPlane l α)

-- Theorem to prove
theorem perpendicular_parallel_relations :
  (∀ m : Line, perpPlane m α → para m l) ∧
  (∀ m : Line, paraPlane m α → perp m l) ∧
  (∀ m : Line, para m l → perpPlane m α) :=
sorry

end perpendicular_parallel_relations_l3684_368416


namespace altitudes_sum_lt_perimeter_l3684_368431

/-- A triangle with side lengths and altitudes -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : ℝ
  hb : ℝ
  hc : ℝ
  side_positive : 0 < a ∧ 0 < b ∧ 0 < c
  altitude_positive : 0 < ha ∧ 0 < hb ∧ 0 < hc
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

/-- The sum of altitudes is less than the perimeter in any triangle -/
theorem altitudes_sum_lt_perimeter (t : Triangle) : t.ha + t.hb + t.hc < t.a + t.b + t.c := by
  sorry

end altitudes_sum_lt_perimeter_l3684_368431


namespace colins_class_girls_l3684_368480

theorem colins_class_girls (total : ℕ) (boys : ℕ) (girls : ℕ) :
  total = 35 →
  boys > 15 →
  girls + boys = total →
  4 * girls = 3 * boys →
  girls = 15 :=
by sorry

end colins_class_girls_l3684_368480


namespace juan_cars_count_l3684_368459

theorem juan_cars_count (num_bicycles num_pickup_trucks num_tricycles total_tires : ℕ)
  (h1 : num_bicycles = 3)
  (h2 : num_pickup_trucks = 8)
  (h3 : num_tricycles = 1)
  (h4 : total_tires = 101)
  (h5 : ∀ (num_cars : ℕ), total_tires = 4 * num_cars + 2 * num_bicycles + 4 * num_pickup_trucks + 3 * num_tricycles) :
  ∃ (num_cars : ℕ), num_cars = 15 ∧ total_tires = 4 * num_cars + 2 * num_bicycles + 4 * num_pickup_trucks + 3 * num_tricycles :=
by
  sorry

end juan_cars_count_l3684_368459


namespace vertex_segments_different_colors_l3684_368434

structure ColoredTriangle where
  n : ℕ  -- number of marked points inside the triangle
  k₁ : ℕ  -- number of segments of first color connected to vertices
  k₂ : ℕ  -- number of segments of second color connected to vertices
  k₃ : ℕ  -- number of segments of third color connected to vertices
  sum_k : k₁ + k₂ + k₃ = 3
  valid_k : 0 ≤ k₁ ∧ k₁ ≤ 3 ∧ 0 ≤ k₂ ∧ k₂ ≤ 3 ∧ 0 ≤ k₃ ∧ k₃ ≤ 3
  even_sum : Even (n + k₁) ∧ Even (n + k₂) ∧ Even (n + k₃)

theorem vertex_segments_different_colors (t : ColoredTriangle) : t.k₁ = 1 ∧ t.k₂ = 1 ∧ t.k₃ = 1 := by
  sorry

end vertex_segments_different_colors_l3684_368434


namespace complex_arithmetic_equality_l3684_368425

theorem complex_arithmetic_equality : (90 + 5) * (12 / (180 / (3^2))) = 57 := by
  sorry

end complex_arithmetic_equality_l3684_368425


namespace shirts_returned_l3684_368421

/-- Given that Haley bought 11 shirts initially and ended up with 5 shirts,
    prove that she returned 6 shirts. -/
theorem shirts_returned (initial_shirts : ℕ) (final_shirts : ℕ) (h1 : initial_shirts = 11) (h2 : final_shirts = 5) :
  initial_shirts - final_shirts = 6 := by
  sorry

end shirts_returned_l3684_368421


namespace closed_polyline_theorem_l3684_368437

/-- Represents a rectangle on a unit grid --/
structure Rectangle where
  m : ℕ  -- Width
  n : ℕ  -- Height

/-- Determines if a closed polyline exists for a given rectangle --/
def closedPolylineExists (rect : Rectangle) : Prop :=
  Odd rect.m ∨ Odd rect.n

/-- Calculates the length of the closed polyline if it exists --/
def polylineLength (rect : Rectangle) : ℕ :=
  (rect.n + 1) * (rect.m + 1)

/-- Main theorem about the existence and length of the closed polyline --/
theorem closed_polyline_theorem (rect : Rectangle) :
  closedPolylineExists rect ↔ 
    ∃ (length : ℕ), length = polylineLength rect ∧ 
      (∀ (i j : ℕ), i ≤ rect.m ∧ j ≤ rect.n → 
        ∃ (unique_visit : Prop), unique_visit) :=
by sorry

end closed_polyline_theorem_l3684_368437


namespace product_of_roots_l3684_368422

theorem product_of_roots (x : ℝ) : 
  (24 * x^2 - 72 * x + 200 = 0) → 
  (∃ r₁ r₂ : ℝ, (24 * r₁^2 - 72 * r₁ + 200 = 0) ∧ 
                (24 * r₂^2 - 72 * r₂ + 200 = 0) ∧ 
                (r₁ * r₂ = 25 / 3)) := by
  sorry

end product_of_roots_l3684_368422


namespace meeting_point_distance_l3684_368484

/-- A problem about two people meeting on a road --/
theorem meeting_point_distance
  (total_distance : ℝ)
  (distance_B_to_C : ℝ)
  (h1 : total_distance = 1000)
  (h2 : distance_B_to_C = 400) :
  total_distance - distance_B_to_C = 600 :=
by sorry

end meeting_point_distance_l3684_368484


namespace smallest_sum_of_perfect_squares_l3684_368467

theorem smallest_sum_of_perfect_squares (x y : ℕ) : 
  x^2 - y^2 = 143 → ∀ a b : ℕ, a^2 - b^2 = 143 → x^2 + y^2 ≤ a^2 + b^2 → x^2 + y^2 = 145 :=
by sorry

end smallest_sum_of_perfect_squares_l3684_368467


namespace problem_statement_l3684_368482

theorem problem_statement (x y : ℤ) (h1 : x = 7) (h2 : y = x + 5) :
  (x - y) * (x + y) = -95 := by
  sorry

end problem_statement_l3684_368482


namespace customers_in_other_countries_l3684_368452

/-- Given a cell phone company with a total of 7422 customers,
    of which 723 live in the United States,
    prove that 6699 customers live in other countries. -/
theorem customers_in_other_countries
  (total : ℕ)
  (usa : ℕ)
  (h1 : total = 7422)
  (h2 : usa = 723) :
  total - usa = 6699 := by
  sorry

end customers_in_other_countries_l3684_368452


namespace f_2011_equals_6_l3684_368455

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop := ∀ x, f (a - x) = f (a + x)

theorem f_2011_equals_6 (f : ℝ → ℝ) 
    (h_even : is_even_function f)
    (h_sym : symmetric_about f 2)
    (h_sum : f 2011 + 2 * f 1 = 18) :
  f 2011 = 6 := by
sorry

end f_2011_equals_6_l3684_368455


namespace hcl_moles_formed_l3684_368479

/-- Represents the chemical reaction NH4Cl + H2O → NH4OH + HCl -/
structure ChemicalReaction where
  nh4cl_mass : ℝ
  h2o_moles : ℝ
  nh4oh_moles : ℝ
  hcl_moles : ℝ

/-- The molar mass of NH4Cl in g/mol -/
def nh4cl_molar_mass : ℝ := 53.49

/-- Theorem stating that in the given reaction, 1 mole of HCl is formed -/
theorem hcl_moles_formed (reaction : ChemicalReaction) 
  (h1 : reaction.nh4cl_mass = 53)
  (h2 : reaction.h2o_moles = 1)
  (h3 : reaction.nh4oh_moles = 1) :
  reaction.hcl_moles = 1 := by
sorry

end hcl_moles_formed_l3684_368479


namespace board_zeros_l3684_368466

theorem board_zeros (n : ℕ) (pos neg zero : ℕ) : 
  n = 10 → 
  pos + neg + zero = n → 
  pos * neg = 15 → 
  zero = 2 := by sorry

end board_zeros_l3684_368466


namespace symmetry_of_shifted_even_function_l3684_368447

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Define the concept of axis of symmetry
def AxisOfSymmetry (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f (a + x) = f (a - x)

-- State the theorem
theorem symmetry_of_shifted_even_function (f : ℝ → ℝ) (h : EvenFunction f) :
  AxisOfSymmetry (fun x ↦ f (x + 1)) (-1) :=
sorry

end symmetry_of_shifted_even_function_l3684_368447


namespace hyperbola_asymptote_slope_l3684_368453

/-- The hyperbola equation -/
def hyperbola_eq (x y : ℝ) : Prop :=
  (y - 1)^2 / 16 - (x + 2)^2 / 9 = 1

/-- The asymptote equation -/
def asymptote_eq (x y m : ℝ) : Prop :=
  y - 1 = m * x + 2 * m ∨ y - 1 = -m * x - 2 * m

/-- Theorem stating that the value of m for the given hyperbola is 4/3 -/
theorem hyperbola_asymptote_slope :
  ∃ m : ℝ, (∀ x y : ℝ, hyperbola_eq x y → asymptote_eq x y m) ∧ m = 4/3 := by
  sorry

end hyperbola_asymptote_slope_l3684_368453


namespace not_always_true_parallel_intersection_l3684_368435

-- Define the types for planes and lines
variable (Plane : Type) (Line : Type)

-- Define the relations
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)

-- Define the planes and lines
variable (α β : Plane)
variable (m n : Line)

-- State the theorem
theorem not_always_true_parallel_intersection
  (h_diff_planes : α ≠ β)
  (h_diff_lines : m ≠ n)
  (h_m_parallel_α : parallel_line_plane m α)
  (h_intersect : intersect α β n) :
  ¬ (parallel_lines m n) :=
sorry

end not_always_true_parallel_intersection_l3684_368435


namespace no_consecutive_squares_l3684_368497

/-- The number of positive integer divisors of n -/
def tau (n : ℕ) : ℕ := (Nat.divisors n).card

/-- The sequence defined by a_(n+1) = a_n + τ(n) -/
def a : ℕ → ℕ
  | 0 => 1  -- Arbitrary starting value
  | n + 1 => a n + tau n

/-- Two consecutive terms of the sequence cannot both be perfect squares -/
theorem no_consecutive_squares (n : ℕ) : ¬(∃ k m : ℕ, a n = k^2 ∧ a (n + 1) = m^2) := by
  sorry


end no_consecutive_squares_l3684_368497


namespace quadratic_rational_root_even_coefficient_l3684_368494

theorem quadratic_rational_root_even_coefficient 
  (a b c : ℤ) (hα : a ≠ 0) : 
  (∃ (x : ℚ), a * x^2 + b * x + c = 0) → 
  (Even a ∨ Even b ∨ Even c) := by
  sorry

end quadratic_rational_root_even_coefficient_l3684_368494


namespace triangle_trig_expression_l3684_368477

theorem triangle_trig_expression (D E F : Real) (h1 : 0 < D) (h2 : 0 < E) (h3 : 0 < F)
  (h4 : D + E + F = Real.pi) 
  (h5 : Real.sin F * 8 = 7) (h6 : Real.sin D * 5 = 8) (h7 : Real.sin E * 7 = 5) :
  (Real.cos ((D - E) / 2) / Real.sin (F / 2)) - (Real.sin ((D - E) / 2) / Real.cos (F / 2)) = 
  1 / Real.sqrt ((1 + Real.sqrt (15 / 64)) / 2) := by
  sorry

end triangle_trig_expression_l3684_368477


namespace exercise_time_distribution_l3684_368492

theorem exercise_time_distribution (total_time : ℕ) (aerobics_ratio : ℕ) (weight_ratio : ℕ) 
  (h1 : total_time = 250)
  (h2 : aerobics_ratio = 3)
  (h3 : weight_ratio = 2) :
  ∃ (aerobics_time weight_time : ℕ),
    aerobics_time + weight_time = total_time ∧
    aerobics_time * weight_ratio = weight_time * aerobics_ratio ∧
    aerobics_time = 150 ∧
    weight_time = 100 := by
  sorry

end exercise_time_distribution_l3684_368492


namespace triangle_area_l3684_368485

/-- Given a triangle with perimeter 36, inradius 2.5, and sides in ratio 3:4:5, its area is 45 -/
theorem triangle_area (a b c : ℝ) (perimeter inradius : ℝ) : 
  perimeter = 36 →
  inradius = 2.5 →
  ∃ (k : ℝ), a = 3*k ∧ b = 4*k ∧ c = 5*k →
  a + b + c = perimeter →
  (a + b + c) / 2 * inradius = 45 := by
sorry


end triangle_area_l3684_368485


namespace max_value_theorem_l3684_368440

-- Define the constraint function
def constraint (x y : ℝ) : Prop := x^2 + y^2 = 2

-- Define the objective function
def objective (x y : ℝ) : ℝ := 3*x + 4*y

-- Theorem statement
theorem max_value_theorem :
  ∃ (max : ℝ), max = 5 * Real.sqrt 2 ∧
  (∀ x y : ℝ, constraint x y → objective x y ≤ max) ∧
  (∃ x y : ℝ, constraint x y ∧ objective x y = max) :=
sorry

end max_value_theorem_l3684_368440


namespace right_triangle_area_l3684_368457

theorem right_triangle_area (a b c : ℝ) (h_right : a^2 + b^2 = c^2)
  (h_ratio : a / b = 7 / 24)
  (h_distance : (c / 2) * ((c / 2) - 2 * ((a + b - c) / 2)) = 1) :
  (1 / 2) * a * b = 336 / 325 := by
  sorry

end right_triangle_area_l3684_368457


namespace dolls_count_l3684_368483

/-- The number of dolls Hannah's sister has -/
def sister_dolls : ℝ := 8.5

/-- The ratio of Hannah's dolls to her sister's dolls -/
def hannah_ratio : ℝ := 5.5

/-- The ratio of cousin's dolls to Hannah and her sister's combined dolls -/
def cousin_ratio : ℝ := 7

/-- The total number of dolls Hannah, her sister, and their cousin have -/
def total_dolls : ℝ := 
  sister_dolls + (hannah_ratio * sister_dolls) + (cousin_ratio * (sister_dolls + hannah_ratio * sister_dolls))

theorem dolls_count : total_dolls = 442 := by
  sorry

end dolls_count_l3684_368483


namespace astronaut_distribution_l3684_368408

/-- The number of ways to distribute n astronauts among k distinct modules,
    with each module containing at least min and at most max astronauts. -/
def distribute_astronauts (n k min max : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that there are 450 ways to distribute
    6 astronauts among 3 distinct modules, with each module containing
    at least 1 and at most 3 astronauts. -/
theorem astronaut_distribution :
  distribute_astronauts 6 3 1 3 = 450 :=
sorry

end astronaut_distribution_l3684_368408


namespace max_q_plus_r_for_1051_l3684_368442

theorem max_q_plus_r_for_1051 :
  ∀ q r : ℕ+,
  1051 = 23 * q + r →
  ∀ q' r' : ℕ+,
  1051 = 23 * q' + r' →
  q + r ≤ 61 :=
by sorry

end max_q_plus_r_for_1051_l3684_368442


namespace direction_vector_c_value_l3684_368418

/-- Given a line passing through two points and a direction vector, prove the value of c. -/
theorem direction_vector_c_value (p1 p2 : ℝ × ℝ) (h : p1 = (-3, 1) ∧ p2 = (0, 4)) :
  let v : ℝ × ℝ := (p2.1 - p1.1, p2.2 - p1.2)
  v.1 = 3 → v = (3, 3) :=
by sorry

end direction_vector_c_value_l3684_368418


namespace cost_price_percentage_l3684_368445

theorem cost_price_percentage (marked_price cost_price selling_price : ℝ) :
  selling_price = 0.88 * marked_price →
  selling_price = 1.375 * cost_price →
  cost_price / marked_price = 0.64 := by
  sorry

end cost_price_percentage_l3684_368445


namespace probability_mame_on_top_l3684_368413

/-- Represents a section of a folded paper -/
structure PaperSection :=
  (side : Fin 2)
  (quadrant : Fin 4)

/-- The total number of sections on a paper folded in quarters -/
def total_sections : ℕ := 8

/-- The probability of a specific section being on top when randomly refolded -/
def probability_on_top : ℚ := 1 / total_sections

theorem probability_mame_on_top :
  probability_on_top = 1 / 8 := by
  sorry

end probability_mame_on_top_l3684_368413


namespace train_station_distance_l3684_368430

theorem train_station_distance (speed1 speed2 : ℝ) (time_diff : ℝ) : 
  speed1 = 4 →
  speed2 = 5 →
  time_diff = 12 / 60 →
  (∃ d : ℝ, d / speed1 - d / speed2 = time_diff ∧ d = 4) :=
by sorry

end train_station_distance_l3684_368430


namespace division_problem_l3684_368404

theorem division_problem (dividend quotient remainder : ℕ) 
  (h1 : dividend = 690) 
  (h2 : quotient = 19) 
  (h3 : remainder = 6) :
  ∃ divisor : ℕ, 
    dividend = divisor * quotient + remainder ∧ 
    divisor = 36 := by
  sorry

end division_problem_l3684_368404


namespace credit_card_interest_rate_l3684_368444

theorem credit_card_interest_rate 
  (initial_balance : ℝ) 
  (payment : ℝ) 
  (new_balance : ℝ) 
  (h1 : initial_balance = 150)
  (h2 : payment = 50)
  (h3 : new_balance = 120) :
  (new_balance - (initial_balance - payment)) / initial_balance * 100 = 13.33 := by
sorry

end credit_card_interest_rate_l3684_368444


namespace monkey_nuts_problem_l3684_368412

theorem monkey_nuts_problem (n : ℕ) (x : ℕ) : 
  n > 1 → 
  x > 1 → 
  n * x - n * (n - 1) = 35 → 
  x = 11 :=
by sorry

end monkey_nuts_problem_l3684_368412


namespace leo_caught_40_l3684_368424

/-- The number of fish Leo caught -/
def leo_fish : ℕ := sorry

/-- The number of fish Agrey caught -/
def agrey_fish : ℕ := sorry

/-- Agrey caught 20 more fish than Leo -/
axiom agrey_more : agrey_fish = leo_fish + 20

/-- They caught a total of 100 fish together -/
axiom total_fish : leo_fish + agrey_fish = 100

/-- Prove that Leo caught 40 fish -/
theorem leo_caught_40 : leo_fish = 40 := by sorry

end leo_caught_40_l3684_368424


namespace vaishalis_total_stripes_l3684_368464

/-- Represents the types of stripes on hats --/
inductive StripeType
  | Solid
  | Zigzag
  | Wavy
  | Other

/-- Represents a hat with its stripe count and type --/
structure Hat :=
  (stripeCount : ℕ)
  (stripeType : StripeType)

/-- Determines if a stripe type should be counted --/
def countStripe (st : StripeType) : Bool :=
  match st with
  | StripeType.Solid => true
  | StripeType.Zigzag => true
  | StripeType.Wavy => true
  | _ => false

/-- Calculates the total number of counted stripes for a list of hats --/
def totalCountedStripes (hats : List Hat) : ℕ :=
  hats.foldl (fun acc hat => 
    if countStripe hat.stripeType then
      acc + hat.stripeCount
    else
      acc
  ) 0

/-- Vaishali's hat collection --/
def vaishalisHats : List Hat := [
  { stripeCount := 3, stripeType := StripeType.Solid },
  { stripeCount := 3, stripeType := StripeType.Solid },
  { stripeCount := 3, stripeType := StripeType.Solid },
  { stripeCount := 3, stripeType := StripeType.Solid },
  { stripeCount := 5, stripeType := StripeType.Zigzag },
  { stripeCount := 5, stripeType := StripeType.Zigzag },
  { stripeCount := 2, stripeType := StripeType.Wavy },
  { stripeCount := 2, stripeType := StripeType.Wavy },
  { stripeCount := 2, stripeType := StripeType.Wavy }
]

theorem vaishalis_total_stripes :
  totalCountedStripes vaishalisHats = 28 := by
  sorry

end vaishalis_total_stripes_l3684_368464


namespace pool_problem_l3684_368476

/-- Given a pool with humans and dogs, calculate the number of dogs -/
def number_of_dogs (total_legs_paws : ℕ) (num_humans : ℕ) (human_legs : ℕ) (dog_paws : ℕ) : ℕ :=
  ((total_legs_paws - (num_humans * human_legs)) / dog_paws)

theorem pool_problem :
  let total_legs_paws : ℕ := 24
  let num_humans : ℕ := 2
  let human_legs : ℕ := 2
  let dog_paws : ℕ := 4
  number_of_dogs total_legs_paws num_humans human_legs dog_paws = 5 := by
  sorry

end pool_problem_l3684_368476


namespace barefoot_kids_count_l3684_368426

theorem barefoot_kids_count (total kids_with_socks kids_with_shoes kids_with_both : ℕ) :
  total = 22 ∧ kids_with_socks = 12 ∧ kids_with_shoes = 8 ∧ kids_with_both = 6 →
  total - ((kids_with_socks - kids_with_both) + (kids_with_shoes - kids_with_both) + kids_with_both) = 8 := by
sorry

end barefoot_kids_count_l3684_368426


namespace geometric_sequence_minimum_l3684_368460

theorem geometric_sequence_minimum (a : ℕ → ℝ) (m n : ℕ) :
  (∀ k, a k > 0) →  -- positive sequence
  (∃ r > 0, ∀ k, a (k + 1) = r * a k) →  -- geometric sequence
  a 9 = 9 * a 7 →  -- given condition
  a m * a n = 9 * (a 1)^2 →  -- given condition
  (∀ i j : ℕ, (a i * a j = 9 * (a 1)^2) → 1/i + 9/j ≥ 1/m + 9/n) →  -- minimum condition
  1/m + 9/n = 4 :=
by sorry

end geometric_sequence_minimum_l3684_368460


namespace quadratic_function_properties_l3684_368454

-- Define the quadratic function f
def f : ℝ → ℝ := λ x => x^2 - 2*x - 1

-- State the theorem
theorem quadratic_function_properties :
  (∀ x, f x ≥ -2) ∧  -- minimum value is -2
  f 3 = 2 ∧ f (-1) = 2 ∧  -- given conditions
  (∀ t, f (2*t^2 - 4*t + 3) > f (t^2 + t + 3) ↔ t > 5 ∨ t < 0) := by
  sorry

end quadratic_function_properties_l3684_368454


namespace sector_area_l3684_368472

theorem sector_area (perimeter : ℝ) (central_angle : ℝ) (h1 : perimeter = 4) (h2 : central_angle = 2) :
  let radius := perimeter / (2 + central_angle)
  let arc_length := radius * central_angle
  let area := (1 / 2) * radius * arc_length
  area = 1 := by sorry

end sector_area_l3684_368472


namespace possible_values_of_a_l3684_368473

theorem possible_values_of_a (a b x : ℝ) 
  (h1 : a ≠ b) 
  (h2 : a^3 - b^3 = 19*x^3) 
  (h3 : a - b = x) : 
  a = 3*x ∨ a = -2*x := by
sorry

end possible_values_of_a_l3684_368473


namespace flag_design_count_l3684_368491

/-- The number of color choices for each stripe -/
def num_colors : ℕ := 3

/-- The number of stripes on the flag -/
def num_stripes : ℕ := 3

/-- The total number of possible flag designs -/
def total_flag_designs : ℕ := num_colors ^ num_stripes

/-- Theorem: The number of possible flag designs is 27 -/
theorem flag_design_count : total_flag_designs = 27 := by
  sorry

end flag_design_count_l3684_368491


namespace vacation_miles_theorem_l3684_368487

/-- Calculates the total miles driven during a vacation -/
def total_miles_driven (days : ℕ) (miles_per_day : ℕ) : ℕ :=
  days * miles_per_day

/-- Proves that a 5-day vacation driving 250 miles per day results in 1250 total miles -/
theorem vacation_miles_theorem : 
  total_miles_driven 5 250 = 1250 := by
  sorry

end vacation_miles_theorem_l3684_368487


namespace oatmeal_cookie_baggies_l3684_368410

theorem oatmeal_cookie_baggies 
  (total_cookies : ℝ) 
  (chocolate_chip_cookies : ℝ) 
  (cookies_per_bag : ℝ) 
  (h1 : total_cookies = 41.0) 
  (h2 : chocolate_chip_cookies = 13.0) 
  (h3 : cookies_per_bag = 9.0) :
  ⌊(total_cookies - chocolate_chip_cookies) / cookies_per_bag⌋ = 3 :=
by
  sorry

end oatmeal_cookie_baggies_l3684_368410


namespace saree_price_proof_l3684_368439

/-- Proves that given a product with two successive discounts of 10% and 5%, 
    if the final sale price is Rs. 513, then the original price was Rs. 600. -/
theorem saree_price_proof (original_price : ℝ) : 
  (original_price * (1 - 0.1) * (1 - 0.05) = 513) → original_price = 600 := by
  sorry

end saree_price_proof_l3684_368439


namespace parallel_line_m_value_l3684_368469

/-- Given two points A(-3,m) and B(m,5), and a line parallel to 3x+y-1=0, prove m = -7 -/
theorem parallel_line_m_value :
  ∀ m : ℝ,
  let A : ℝ × ℝ := (-3, m)
  let B : ℝ × ℝ := (m, 5)
  let parallel_line_slope : ℝ := -3
  (B.2 - A.2) / (B.1 - A.1) = parallel_line_slope →
  m = -7 :=
by
  sorry

end parallel_line_m_value_l3684_368469


namespace square_perimeter_from_area_l3684_368468

theorem square_perimeter_from_area (area : ℝ) (perimeter : ℝ) :
  area = 225 → perimeter = 60 :=
by
  sorry

end square_perimeter_from_area_l3684_368468


namespace rectangle_y_value_l3684_368470

/-- Given a rectangle with vertices at (-2, y), (10, y), (-2, 4), and (10, 4),
    where y is positive and the area is 108 square units, prove that y = 13. -/
theorem rectangle_y_value (y : ℝ) (h1 : y > 0) (h2 : (10 - (-2)) * (y - 4) = 108) : y = 13 := by
  sorry

end rectangle_y_value_l3684_368470


namespace platform_length_l3684_368493

/-- Given a train with speed 72 km/h and length 290.04 m, crossing a platform in 26 seconds,
    prove that the length of the platform is 229.96 m. -/
theorem platform_length (train_speed : ℝ) (train_length : ℝ) (crossing_time : ℝ) :
  train_speed = 72 →
  train_length = 290.04 →
  crossing_time = 26 →
  ∃ platform_length : ℝ,
    platform_length = 229.96 ∧
    platform_length = train_speed * (1000 / 3600) * crossing_time - train_length :=
by sorry

end platform_length_l3684_368493


namespace cylinder_volume_relation_l3684_368420

/-- Given two cylinders C and D with the specified properties, 
    prove that the volume of D is 9πh³ --/
theorem cylinder_volume_relation (h r : ℝ) : 
  h > 0 → r > 0 → 
  (π * h^2 * r) * 3 = π * r^2 * h → 
  π * r^2 * h = 9 * π * h^3 := by
  sorry

end cylinder_volume_relation_l3684_368420


namespace eighteenth_is_sunday_l3684_368429

/-- Represents days of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents a date in a month -/
structure Date where
  day : Nat
  dayOfWeek : DayOfWeek

/-- A month with three Fridays on even dates -/
structure SpecialMonth where
  dates : List Date
  three_even_fridays : ∃ (d1 d2 d3 : Date),
    d1 ∈ dates ∧ d2 ∈ dates ∧ d3 ∈ dates ∧
    d1.day ≠ d2.day ∧ d1.day ≠ d3.day ∧ d2.day ≠ d3.day ∧
    d1.dayOfWeek = DayOfWeek.Friday ∧
    d2.dayOfWeek = DayOfWeek.Friday ∧
    d3.dayOfWeek = DayOfWeek.Friday ∧
    d1.day % 2 = 0 ∧ d2.day % 2 = 0 ∧ d3.day % 2 = 0

/-- The 18th of a special month is a Sunday -/
theorem eighteenth_is_sunday (m : SpecialMonth) :
  ∃ (d : Date), d ∈ m.dates ∧ d.day = 18 ∧ d.dayOfWeek = DayOfWeek.Sunday :=
sorry

end eighteenth_is_sunday_l3684_368429


namespace jemma_grasshopper_count_l3684_368417

/-- The number of grasshoppers Jemma saw on her African daisy plant -/
def grasshoppers_on_plant : ℕ := 7

/-- The number of dozens of baby grasshoppers Jemma found under the plant -/
def dozens_of_baby_grasshoppers : ℕ := 2

/-- The number of grasshoppers in a dozen -/
def grasshoppers_per_dozen : ℕ := 12

/-- The total number of grasshoppers Jemma found -/
def total_grasshoppers : ℕ := grasshoppers_on_plant + dozens_of_baby_grasshoppers * grasshoppers_per_dozen

theorem jemma_grasshopper_count : total_grasshoppers = 31 := by
  sorry

end jemma_grasshopper_count_l3684_368417


namespace condition_for_reciprocal_less_than_one_l3684_368411

theorem condition_for_reciprocal_less_than_one (a : ℝ) :
  (a > 1 → (1 / a) < 1) ∧ (∃ b : ℝ, (1 / b) < 1 ∧ b ≤ 1) := by sorry

end condition_for_reciprocal_less_than_one_l3684_368411


namespace rationalize_cube_root_difference_l3684_368490

theorem rationalize_cube_root_difference : ∃ (A B C D : ℕ),
  (((1 : ℝ) / (5^(1/3) - 3^(1/3))) * ((5^(2/3) + 5^(1/3)*3^(1/3) + 3^(2/3)) / (5^(2/3) + 5^(1/3)*3^(1/3) + 3^(2/3)))) = 
  ((A : ℝ)^(1/3) + (B : ℝ)^(1/3) + (C : ℝ)^(1/3)) / (D : ℝ) ∧
  A + B + C + D = 51 := by
sorry

end rationalize_cube_root_difference_l3684_368490


namespace curve_symmetrical_about_y_axis_l3684_368414

/-- A curve defined by an equation f(x, y) = 0 is symmetrical about the y-axis
    if f(-x, y) = f(x, y) for all x and y. -/
def is_symmetrical_about_y_axis (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y, f (-x) y = f x y

/-- The equation x^2 - y^2 = 1 can be represented as a function f(x, y) = x^2 - y^2 - 1. -/
def f (x y : ℝ) : ℝ := x^2 - y^2 - 1

/-- Theorem: The curve defined by x^2 - y^2 = 1 is symmetrical about the y-axis. -/
theorem curve_symmetrical_about_y_axis : is_symmetrical_about_y_axis f := by
  sorry

end curve_symmetrical_about_y_axis_l3684_368414


namespace orange_pricing_theorem_l3684_368461

/-- Represents the price in cents for a pack of oranges -/
structure PackPrice :=
  (quantity : ℕ)
  (price : ℕ)

/-- Calculates the total cost for a given number of packs -/
def totalCost (pack : PackPrice) (numPacks : ℕ) : ℕ :=
  pack.price * numPacks

/-- Calculates the total number of oranges for a given number of packs -/
def totalOranges (pack : PackPrice) (numPacks : ℕ) : ℕ :=
  pack.quantity * numPacks

theorem orange_pricing_theorem (pack1 pack2 : PackPrice) 
    (h1 : pack1 = ⟨4, 15⟩)
    (h2 : pack2 = ⟨6, 25⟩)
    (h3 : totalOranges pack1 5 + totalOranges pack2 5 = 20) :
  (totalCost pack1 5 + totalCost pack2 5) / 20 = 10 := by
  sorry

end orange_pricing_theorem_l3684_368461


namespace union_complement_eq_specific_set_l3684_368400

open Set

def U : Finset ℕ := {0, 1, 2, 4, 6, 8}
def M : Finset ℕ := {0, 4, 6}
def N : Finset ℕ := {0, 1, 6}

theorem union_complement_eq_specific_set :
  M ∪ (U \ N) = {0, 2, 4, 6, 8} :=
sorry

end union_complement_eq_specific_set_l3684_368400


namespace circular_track_speed_l3684_368428

/-- The speed of person A in rounds per hour -/
def speed_A : ℝ := 7

/-- The speed of person B in rounds per hour -/
def speed_B : ℝ := 3

/-- The number of times A and B cross each other in 1 hour -/
def crossings : ℕ := 10

/-- The time period in hours -/
def time_period : ℝ := 1

theorem circular_track_speed :
  speed_A + speed_B = crossings / time_period :=
by sorry

end circular_track_speed_l3684_368428


namespace percentage_problem_l3684_368406

theorem percentage_problem (P : ℝ) : (P / 100 * 1265) / 6 = 543.95 ↔ P = 258 := by
  sorry

end percentage_problem_l3684_368406


namespace square_k_ascending_range_l3684_368449

/-- A function f is k-ascending on a set M if for all x in M, f(x + k) ≥ f(x) --/
def IsKAscending (f : ℝ → ℝ) (k : ℝ) (M : Set ℝ) : Prop :=
  ∀ x ∈ M, f (x + k) ≥ f x

theorem square_k_ascending_range {k : ℝ} (hk : k ≠ 0) :
  IsKAscending (fun x ↦ x^2) k (Set.Ioi (-1)) → k ≥ 2 := by
  sorry

end square_k_ascending_range_l3684_368449


namespace p_sequence_constant_difference_l3684_368499

/-- A P-sequence is a geometric sequence {a_n} where (a_1 + 1, a_2 + 2, a_3 + 3) also forms a geometric sequence -/
def is_p_sequence (a : ℝ) (a_n : ℕ → ℝ) : Prop :=
  (∀ n, a_n (n + 1) = a_n n * a_n 1) ∧
  (∃ r, (a_n 1 + 1) * r = a_n 2 + 2 ∧ (a_n 2 + 2) * r = a_n 3 + 3)

theorem p_sequence_constant_difference (a : ℝ) (h1 : 1/2 < a) (h2 : a < 1) :
  let a_n : ℕ → ℝ := λ n => a^(2*n - 1)
  let x_n : ℕ → ℝ := λ n => a_n n - 1 / (a_n n)
  is_p_sequence a a_n →
  ∀ n ≥ 2, x_n n^2 - x_n (n-1) * x_n (n+1) = 5 := by
sorry

end p_sequence_constant_difference_l3684_368499


namespace defective_rate_is_twenty_percent_l3684_368496

variable (n : ℕ)  -- number of defective items among 10 products

-- Define the probability of selecting one defective item out of two random selections
def prob_one_defective (n : ℕ) : ℚ :=
  (n * (10 - n)) / (10 * 9)

-- Theorem statement
theorem defective_rate_is_twenty_percent :
  n ≤ 10 ∧                     -- n is at most 10 (total number of products)
  prob_one_defective n = 16/45 ∧ -- probability of selecting one defective item is 16/45
  n ≤ 4 →                      -- defective rate does not exceed 40%
  n = 2                        -- implies that n = 2, which means 20% defective rate
  := by sorry

end defective_rate_is_twenty_percent_l3684_368496


namespace helmet_sales_theorem_l3684_368471

/-- Represents the helmet sales scenario -/
structure HelmetSales where
  originalPrice : ℝ
  originalSales : ℝ
  costPrice : ℝ
  salesIncrease : ℝ

/-- Calculates the monthly profit given a price reduction -/
def monthlyProfit (hs : HelmetSales) (priceReduction : ℝ) : ℝ :=
  (hs.originalPrice - priceReduction - hs.costPrice) * (hs.originalSales + hs.salesIncrease * priceReduction)

/-- The main theorem about helmet sales -/
theorem helmet_sales_theorem (hs : HelmetSales) 
    (h1 : hs.originalPrice = 80)
    (h2 : hs.originalSales = 200)
    (h3 : hs.costPrice = 50)
    (h4 : hs.salesIncrease = 10) :
  (∃ (x : ℝ), x ≥ 10 ∧ monthlyProfit hs x = 5250 ∧ hs.originalPrice - x = 65) ∧
  (∃ (maxProfit : ℝ), maxProfit = 6000 ∧ 
    ∀ (y : ℝ), y ≥ 10 → monthlyProfit hs y ≤ maxProfit ∧
    monthlyProfit hs 10 = maxProfit) := by
  sorry

end helmet_sales_theorem_l3684_368471


namespace count_two_digit_S_equal_l3684_368481

def S (n : ℕ) : ℕ :=
  (n % 2) + (n % 3) + (n % 4) + (n % 5)

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem count_two_digit_S_equal : 
  ∃ (l : List ℕ), (∀ n ∈ l, is_two_digit n ∧ S n = S (n + 1)) ∧ 
                  (∀ n, is_two_digit n → S n = S (n + 1) → n ∈ l) ∧
                  l.length = 6 :=
sorry

end count_two_digit_S_equal_l3684_368481


namespace rectangular_to_cubic_block_l3684_368456

/-- The edge length of a cube with the same volume as a rectangular block -/
def cube_edge_length (l w h : ℝ) : ℝ :=
  (l * w * h) ^ (1/3)

/-- Theorem stating that a 50cm x 8cm x 20cm rectangular block forged into a cube has an edge length of 20cm -/
theorem rectangular_to_cubic_block :
  cube_edge_length 50 8 20 = 20 := by
  sorry

end rectangular_to_cubic_block_l3684_368456


namespace percentage_sum_problem_l3684_368458

theorem percentage_sum_problem : (0.2 * 40) + (0.25 * 60) = 23 := by
  sorry

end percentage_sum_problem_l3684_368458


namespace arithmetic_sequence_sum_l3684_368495

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence, if a_1 + a_7 = 10, then a_3 + a_5 = 10 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : arithmetic_sequence a) (h1 : a 1 + a 7 = 10) :
  a 3 + a 5 = 10 := by
  sorry

end arithmetic_sequence_sum_l3684_368495


namespace container_problem_l3684_368415

theorem container_problem (x y : ℝ) :
  (∃ (large_capacity small_capacity : ℝ),
    large_capacity = x ∧
    small_capacity = y ∧
    5 * large_capacity + small_capacity = 3 ∧
    large_capacity + 5 * small_capacity = 2) →
  (5 * x + y = 3 ∧ x + 5 * y = 2) :=
by sorry

end container_problem_l3684_368415


namespace reciprocal_of_negative_one_l3684_368474

theorem reciprocal_of_negative_one :
  ∃ x : ℚ, x * (-1) = 1 ∧ x = -1 :=
sorry

end reciprocal_of_negative_one_l3684_368474


namespace number_difference_l3684_368448

theorem number_difference (L S : ℕ) (hL : L > S) (hDiv : L = 6 * S + 20) (hLValue : L = 1634) : 
  L - S = 1365 := by
sorry

end number_difference_l3684_368448


namespace seventh_term_of_geometric_sequence_l3684_368403

/-- Given a geometric sequence with 10 terms, first term 6, and last term 93312,
    prove that the 7th term is 279936 -/
theorem seventh_term_of_geometric_sequence :
  ∀ (a : ℕ → ℝ),
    (∀ i j, a (i + 1) / a i = a (j + 1) / a j) →  -- geometric sequence condition
    a 1 = 6 →                                     -- first term is 6
    a 10 = 93312 →                                -- last term is 93312
    a 7 = 279936 := by
  sorry

end seventh_term_of_geometric_sequence_l3684_368403


namespace marie_cash_register_cost_l3684_368463

/-- A bakery's daily sales and expenses -/
structure BakeryFinances where
  bread_quantity : ℕ
  bread_price : ℝ
  cake_quantity : ℕ
  cake_price : ℝ
  rent : ℝ
  electricity : ℝ

/-- Calculate the cost of a cash register based on daily sales and expenses -/
def cash_register_cost (finances : BakeryFinances) (days : ℕ) : ℝ :=
  let daily_sales := finances.bread_quantity * finances.bread_price + 
                     finances.cake_quantity * finances.cake_price
  let daily_expenses := finances.rent + finances.electricity
  let daily_profit := daily_sales - daily_expenses
  days * daily_profit

/-- Marie's bakery finances -/
def marie_finances : BakeryFinances :=
  { bread_quantity := 40
  , bread_price := 2
  , cake_quantity := 6
  , cake_price := 12
  , rent := 20
  , electricity := 2 }

/-- Theorem: The cost of Marie's cash register is $1040 -/
theorem marie_cash_register_cost :
  cash_register_cost marie_finances 8 = 1040 := by
  sorry

end marie_cash_register_cost_l3684_368463


namespace unknown_blanket_rate_l3684_368402

/-- Given the purchase of blankets with specific quantities and prices, 
    prove that the unknown rate for two blankets is 225 Rs. -/
theorem unknown_blanket_rate : 
  ∀ (unknown_rate : ℕ),
  (3 * 100 + 2 * 150 + 2 * unknown_rate) / 7 = 150 →
  unknown_rate = 225 := by
sorry

end unknown_blanket_rate_l3684_368402


namespace orange_groups_count_l3684_368489

/-- The number of oranges in Philip's collection -/
def total_oranges : ℕ := 356

/-- The number of oranges in each group -/
def oranges_per_group : ℕ := 2

/-- The number of groups of oranges -/
def orange_groups : ℕ := total_oranges / oranges_per_group

theorem orange_groups_count : orange_groups = 178 := by
  sorry

end orange_groups_count_l3684_368489


namespace factor_of_expression_l3684_368423

theorem factor_of_expression (x y z : ℝ) :
  ∃ (f : ℝ → ℝ → ℝ → ℝ), (x^2 - y^2 - z^2 + 2*y*z + x + y - z) = (x - y + z + 1) * f x y z := by
  sorry

end factor_of_expression_l3684_368423


namespace certain_number_equation_l3684_368451

theorem certain_number_equation (x : ℝ) : 0.85 * 40 = (4/5) * x + 14 → x = 25 := by
  sorry

end certain_number_equation_l3684_368451


namespace parametric_to_standard_equation_l3684_368478

theorem parametric_to_standard_equation (x y t : ℝ) :
  (∃ θ : ℝ, x = (1/2) * (Real.exp t + Real.exp (-t)) * Real.cos θ ∧
             y = (1/2) * (Real.exp t - Real.exp (-t)) * Real.sin θ) →
  x^2 * (Real.exp (2*t) - 2 + Real.exp (-2*t)) + 
  y^2 * (Real.exp (2*t) + 2 + Real.exp (-2*t)) = 
  Real.exp (6*t) - 2 * Real.exp (4*t) + Real.exp (2*t) + 
  2 * Real.exp (4*t) - 4 * Real.exp (2*t) + 2 + 
  Real.exp (2*t) - 2 * Real.exp (-2*t) + Real.exp (-4*t) :=
by sorry

end parametric_to_standard_equation_l3684_368478


namespace max_product_constraint_l3684_368443

theorem max_product_constraint (x y : ℝ) (h : x + y = 1) : x * y ≤ 1 / 4 := by
  sorry

end max_product_constraint_l3684_368443


namespace odd_function_negative_domain_l3684_368475

-- Define the function f for x > 0
def f_pos (x : ℝ) : ℝ := x^2 - x - 1

-- Define the property of an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_negative_domain 
  (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_pos : ∀ x > 0, f x = f_pos x) : 
  ∀ x < 0, f x = -x^2 - x + 1 := by
sorry

end odd_function_negative_domain_l3684_368475


namespace constant_ratio_problem_l3684_368427

theorem constant_ratio_problem (x y : ℝ) (k : ℝ) : 
  (∀ x y, k = (5 * x - 3) / (y + 10)) →  -- The ratio is constant for all x and y
  (k = (5 * 3 - 3) / (4 + 10)) →         -- When x = 3, y = 4
  (19 + 10 = (5 * (39 / 7) - 3) / k) →   -- When y = 19, x = 39/7
  x = 39 / 7                             -- Conclusion
  := by sorry

end constant_ratio_problem_l3684_368427


namespace expenditure_increase_l3684_368433

theorem expenditure_increase (income : ℝ) (expenditure : ℝ) (savings : ℝ) 
  (new_income : ℝ) (new_expenditure : ℝ) (new_savings : ℝ) :
  expenditure = 0.75 * income →
  savings = income - expenditure →
  new_income = 1.2 * income →
  new_savings = 1.5 * savings →
  new_savings = new_income - new_expenditure →
  new_expenditure = 1.1 * expenditure :=
by sorry

end expenditure_increase_l3684_368433


namespace count_numbers_with_6_or_7_proof_l3684_368438

/-- The number of integers from 1 to 512 (inclusive) in base 8 that contain at least one digit 6 or 7 -/
def count_numbers_with_6_or_7 : ℕ := 296

/-- The total number of integers we're considering -/
def total_numbers : ℕ := 512

/-- The base we're working in -/
def base : ℕ := 8

/-- The number of digits available in the restricted set (0-5) -/
def restricted_digits : ℕ := 6

/-- The number of digits needed to represent the largest number in our set in base 8 -/
def num_digits : ℕ := 3

theorem count_numbers_with_6_or_7_proof :
  count_numbers_with_6_or_7 = total_numbers - restricted_digits ^ num_digits :=
by sorry

end count_numbers_with_6_or_7_proof_l3684_368438


namespace expected_votes_for_candidate_a_l3684_368488

theorem expected_votes_for_candidate_a (total_voters : ℕ) 
  (democrat_percentage : ℚ) (republican_percentage : ℚ)
  (democrat_support : ℚ) (republican_support : ℚ) :
  democrat_percentage + republican_percentage = 1 →
  democrat_percentage = 3/5 →
  democrat_support = 3/4 →
  republican_support = 1/5 →
  (democrat_percentage * democrat_support + 
   republican_percentage * republican_support) * total_voters = 
  (53/100) * total_voters :=
by sorry

end expected_votes_for_candidate_a_l3684_368488
