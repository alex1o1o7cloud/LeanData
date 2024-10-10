import Mathlib

namespace rectangle_area_proof_l3277_327759

theorem rectangle_area_proof : 
  let card1 : ℝ := 15
  let card2 : ℝ := card1 * 0.9
  let area : ℝ := card1 * card2
  area = 202.5 := by
  sorry

end rectangle_area_proof_l3277_327759


namespace line_intersects_circle_l3277_327729

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

-- Define the line L
def line_L (k x y : ℝ) : Prop := y = k*x - 3*k + 1

-- Theorem statement
theorem line_intersects_circle :
  ∀ (k : ℝ), ∃ (x y : ℝ), circle_C x y ∧ line_L k x y :=
sorry

end line_intersects_circle_l3277_327729


namespace inverse_f_f_condition_l3277_327782

-- Define the function f
def f (x : ℝ) : ℝ := (x - 2)^2

-- State the theorem
theorem inverse_f (x : ℝ) (h : x ≥ -1) : 
  f⁻¹ (x + 1) = 2 - Real.sqrt (x + 1) := by sorry

-- Define the domain of f
def dom_f : Set ℝ := {x : ℝ | x ≤ 2}

-- State the condition given in the problem
theorem f_condition (x : ℝ) (h : x ≤ 1) : 
  f (x + 1) = (x - 1)^2 := by sorry

end inverse_f_f_condition_l3277_327782


namespace rocket_heights_sum_l3277_327738

/-- The height of the first rocket in feet -/
def first_rocket_height : ℝ := 500

/-- The height of the second rocket in feet -/
def second_rocket_height : ℝ := 2 * first_rocket_height

/-- The combined height of both rockets in feet -/
def combined_height : ℝ := first_rocket_height + second_rocket_height

theorem rocket_heights_sum :
  combined_height = 1500 := by
  sorry

end rocket_heights_sum_l3277_327738


namespace triangle_side_length_l3277_327700

theorem triangle_side_length (n : ℕ) : 
  (7 + 11 + n > 35) ∧ 
  (7 + 11 > n) ∧ 
  (7 + n > 11) ∧ 
  (11 + n > 7) →
  n = 18 := by
sorry

end triangle_side_length_l3277_327700


namespace rectangular_prism_sum_l3277_327702

/-- A rectangular prism is a three-dimensional shape with 6 rectangular faces -/
structure RectangularPrism where
  -- We don't need to define any specific properties here

/-- The number of edges in a rectangular prism -/
def num_edges (rp : RectangularPrism) : ℕ := 12

/-- The number of corners in a rectangular prism -/
def num_corners (rp : RectangularPrism) : ℕ := 8

/-- The number of faces in a rectangular prism -/
def num_faces (rp : RectangularPrism) : ℕ := 6

/-- Theorem: The sum of edges, corners, and faces of a rectangular prism is 26 -/
theorem rectangular_prism_sum (rp : RectangularPrism) : 
  num_edges rp + num_corners rp + num_faces rp = 26 := by
  sorry

#check rectangular_prism_sum

end rectangular_prism_sum_l3277_327702


namespace constant_molecular_weight_l3277_327774

/-- The molecular weight of a compound in grams per mole -/
def molecular_weight : ℝ := 3264

/-- The number of moles of the compound -/
def number_of_moles : ℝ := 8

/-- Theorem: The molecular weight of a compound remains constant regardless of the number of moles -/
theorem constant_molecular_weight : 
  molecular_weight = molecular_weight * (number_of_moles / number_of_moles) :=
by sorry

end constant_molecular_weight_l3277_327774


namespace set_equality_and_range_of_a_l3277_327771

-- Define the sets
def M : Set ℝ := {x | (x + 3)^2 ≤ 0}
def N : Set ℝ := {x | x^2 + x - 6 = 0}
def B (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ 5 - a}
def A : Set ℝ := (Set.univ \ M) ∩ N

-- State the theorem
theorem set_equality_and_range_of_a :
  (A = {2}) ∧
  (∀ a : ℝ, (A ∪ B a = A) ↔ a ≥ 3) := by
  sorry

end set_equality_and_range_of_a_l3277_327771


namespace A_not_lose_probability_l3277_327746

/-- The probability of player A winning -/
def prob_A_win : ℝ := 0.30

/-- The probability of a draw between players A and B -/
def prob_draw : ℝ := 0.25

/-- The probability that player A does not lose -/
def prob_A_not_lose : ℝ := prob_A_win + prob_draw

theorem A_not_lose_probability : prob_A_not_lose = 0.55 := by
  sorry

end A_not_lose_probability_l3277_327746


namespace max_primitive_dinosaur_cells_l3277_327749

/-- An animal is a connected figure consisting of equal-sized square cells. -/
structure Animal where
  cells : ℕ
  is_connected : Bool

/-- A dinosaur is an animal with at least 2007 cells. -/
def Dinosaur (a : Animal) : Prop :=
  a.cells ≥ 2007

/-- A primitive dinosaur cannot be partitioned into two or more dinosaurs. -/
def PrimitiveDinosaur (a : Animal) : Prop :=
  Dinosaur a ∧ ¬∃ (b c : Animal), Dinosaur b ∧ Dinosaur c ∧ b.cells + c.cells ≤ a.cells

/-- The maximum number of cells in a primitive dinosaur is 8025. -/
theorem max_primitive_dinosaur_cells :
  ∃ (a : Animal), PrimitiveDinosaur a ∧ a.cells = 8025 ∧
  ∀ (b : Animal), PrimitiveDinosaur b → b.cells ≤ 8025 := by
  sorry


end max_primitive_dinosaur_cells_l3277_327749


namespace simplify_expression_l3277_327783

theorem simplify_expression (w : ℝ) : w - 2*w + 4*w - 5*w + 3 - 5 + 7 - 9 = -2*w - 4 := by
  sorry

end simplify_expression_l3277_327783


namespace smallest_sum_of_perfect_squares_l3277_327740

theorem smallest_sum_of_perfect_squares (x y : ℕ) : 
  x^2 - y^2 = 175 → ∃ (a b : ℕ), a^2 - b^2 = 175 ∧ a^2 + b^2 ≤ x^2 + y^2 ∧ a^2 + b^2 = 625 :=
by sorry

end smallest_sum_of_perfect_squares_l3277_327740


namespace decagon_adjacent_vertices_probability_l3277_327792

/-- A decagon is a polygon with 10 vertices -/
def Decagon : Type := Nat

/-- The number of vertices in a decagon -/
def num_vertices : Nat := 10

/-- The number of adjacent vertices for each vertex in a decagon -/
def num_adjacent : Nat := 2

/-- The probability of choosing two distinct adjacent vertices in a decagon -/
def prob_adjacent_vertices (d : Decagon) : Rat :=
  num_adjacent / (num_vertices - 1)

theorem decagon_adjacent_vertices_probability :
  ∀ d : Decagon, prob_adjacent_vertices d = 2/9 := by
  sorry

end decagon_adjacent_vertices_probability_l3277_327792


namespace quadratic_function_theorem_l3277_327763

def f (a b x : ℝ) : ℝ := 2 * x^2 + a * x + b

theorem quadratic_function_theorem (a b : ℝ) :
  (∀ x, f a b x = f a b (-x)) → -- f is an even function
  f a b 1 = -3 →                -- f(1) = -3
  (∀ x, f a b x = 2 * x^2 - 5) ∧ -- f(x) = 2x² - 5
  {x : ℝ | 2 * x^2 - 5 ≥ 3 * x + 4} = {x : ℝ | x ≤ -3/2 ∨ x ≥ 3} := by
sorry


end quadratic_function_theorem_l3277_327763


namespace system_solution_l3277_327739

theorem system_solution (x y k : ℝ) : 
  x + 3*y = 2*k + 1 → 
  x - y = 1 → 
  x = -y → 
  k = -1 := by
sorry

end system_solution_l3277_327739


namespace distance_to_complex_point_l3277_327725

open Complex

theorem distance_to_complex_point :
  let z : ℂ := 3 / (2 - I)^2
  abs z = 3 / 5 := by sorry

end distance_to_complex_point_l3277_327725


namespace point_D_transformation_l3277_327790

def rotate_90_clockwise (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, -p.1)

def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

def transform_point (p : ℝ × ℝ) : ℝ × ℝ :=
  reflect_y_axis (reflect_x_axis (rotate_90_clockwise p))

theorem point_D_transformation :
  transform_point (4, -3) = (3, 4) := by
  sorry

end point_D_transformation_l3277_327790


namespace reciprocal_and_abs_of_negative_one_sixth_l3277_327754

theorem reciprocal_and_abs_of_negative_one_sixth :
  let x : ℚ := -1/6
  let reciprocal : ℚ := 1/x
  (reciprocal = -6) ∧ (abs reciprocal = 6) := by
  sorry

end reciprocal_and_abs_of_negative_one_sixth_l3277_327754


namespace triangle_area_proof_l3277_327762

noncomputable def f (x : ℝ) := Real.cos (2 * x) + 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem triangle_area_proof (A B C : ℝ) (hA : f A = 1) (ha : Real.sqrt 3 = A) (hbc : B + C = 3) :
  (1 / 2 : ℝ) * B * C * Real.sin A = Real.sqrt 3 / 2 := by
  sorry

end triangle_area_proof_l3277_327762


namespace fraction_equality_l3277_327724

theorem fraction_equality (a b : ℝ) (h1 : a ≠ b) (h2 : b ≠ 0) :
  let x := a / b
  (a^2 + b^2) / (a^2 - b^2) = (x^2 + 1) / (x^2 - 1) :=
by sorry

end fraction_equality_l3277_327724


namespace coefficient_a7_equals_negative_eight_l3277_327786

theorem coefficient_a7_equals_negative_eight :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ),
  (∀ x : ℝ, (x - 2)^8 = a₀ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + 
                        a₅*(x-1)^5 + a₆*(x-1)^6 + a₇*(x-1)^7 + a₈*(x-1)^8) →
  a₇ = -8 := by
sorry

end coefficient_a7_equals_negative_eight_l3277_327786


namespace sin_105_times_sin_15_l3277_327733

theorem sin_105_times_sin_15 : Real.sin (105 * π / 180) * Real.sin (15 * π / 180) = 1 / 4 := by
  sorry

end sin_105_times_sin_15_l3277_327733


namespace pedal_triangle_perimeter_and_area_l3277_327705

/-- Given a triangle with circumradius R and angles α, β, and γ,
    this theorem states the formulas for the perimeter and twice the area of its pedal triangle. -/
theorem pedal_triangle_perimeter_and_area 
  (R : ℝ) (α β γ : ℝ) : 
  ∃ (k t : ℝ),
    k = 4 * R * Real.sin α * Real.sin β * Real.sin γ ∧ 
    2 * t = R^2 * Real.sin (2*α) * Real.sin (2*β) * Real.sin (2*γ) := by
  sorry

end pedal_triangle_perimeter_and_area_l3277_327705


namespace existence_of_a_sequence_l3277_327718

theorem existence_of_a_sequence (n : ℕ) (x : Fin n → Fin n → ℝ) 
  (h : ∀ (i j k : Fin n), x i j + x j k + x k i = 0) :
  ∃ (a : Fin n → ℝ), ∀ (i j : Fin n), x i j = a i - a j := by
  sorry

end existence_of_a_sequence_l3277_327718


namespace num_solutions_eq_1176_l3277_327713

/-- The number of distinct ordered triples (a, b, c) of positive integers satisfying a + b + c = 50 -/
def num_solutions : ℕ :=
  (Finset.range 49).sum (λ k ↦ 49 - k)

/-- Theorem stating that the number of solutions is 1176 -/
theorem num_solutions_eq_1176 : num_solutions = 1176 := by
  sorry

end num_solutions_eq_1176_l3277_327713


namespace theater_queue_arrangements_l3277_327796

theorem theater_queue_arrangements :
  let total_people : ℕ := 7
  let pair_size : ℕ := 2
  let units : ℕ := total_people - pair_size + 1
  units.factorial * pair_size.factorial = 1440 :=
by sorry

end theater_queue_arrangements_l3277_327796


namespace line_equation_through_two_points_l3277_327758

/-- The line passing through points A(-2, 4) and B(-1, 3) has the equation y = -x + 2 -/
theorem line_equation_through_two_points :
  let A : ℝ × ℝ := (-2, 4)
  let B : ℝ × ℝ := (-1, 3)
  let line_eq : ℝ → ℝ := λ x => -x + 2
  (line_eq A.1 = A.2) ∧ (line_eq B.1 = B.2) := by
  sorry

end line_equation_through_two_points_l3277_327758


namespace mercury_radius_scientific_notation_l3277_327765

/-- Given a number in decimal notation, returns its scientific notation as a pair (a, n) where a is the coefficient and n is the exponent. -/
def toScientificNotation (x : ℝ) : ℝ × ℤ :=
  sorry

theorem mercury_radius_scientific_notation :
  toScientificNotation 2440000 = (2.44, 6) :=
sorry

end mercury_radius_scientific_notation_l3277_327765


namespace functional_equation_solution_l3277_327753

/-- A function f from non-negative reals to reals satisfying f(x + y) = f(x) * f(y) -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≥ 0 → y ≥ 0 → f (x + y) = f x * f y

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h1 : FunctionalEquation f)
  (h2 : ∀ x, x ≥ 0 → f x ≥ 0)
  (h3 : f 3 = f 1 ^ 3) :
  ∀ c : ℝ, c ≥ 0 → ∃ g : ℝ → ℝ, FunctionalEquation g ∧ g 1 = c ∧ g 3 = g 1 ^ 3 :=
sorry

end functional_equation_solution_l3277_327753


namespace crayons_lost_l3277_327769

/-- Given that Paul gave away 52 crayons and lost or gave away a total of 587 crayons,
    prove that the number of crayons he lost is 535. -/
theorem crayons_lost (crayons_given_away : ℕ) (total_lost_or_given_away : ℕ)
    (h1 : crayons_given_away = 52)
    (h2 : total_lost_or_given_away = 587) :
    total_lost_or_given_away - crayons_given_away = 535 := by
  sorry

end crayons_lost_l3277_327769


namespace initial_workers_count_l3277_327722

/-- Represents the work done in digging a hole -/
def work (workers : ℕ) (hours : ℕ) (depth : ℕ) : ℕ := workers * hours * depth

theorem initial_workers_count :
  ∀ (W : ℕ),
  (∃ (k : ℕ), k > 0 ∧
    work W 8 30 = k * 30 ∧
    work (W + 35) 6 40 = k * 40) →
  W = 105 :=
by
  sorry

end initial_workers_count_l3277_327722


namespace prob_sum_div_three_is_seven_ninths_l3277_327773

/-- Represents a biased die where even numbers are twice as likely as odd numbers -/
structure BiasedDie where
  /-- Probability of rolling an odd number -/
  odd_prob : ℝ
  /-- Probability of rolling an even number -/
  even_prob : ℝ
  /-- The probabilities sum to 1 -/
  sum_to_one : odd_prob + even_prob = 1
  /-- Even numbers are twice as likely as odd numbers -/
  even_twice_odd : even_prob = 2 * odd_prob

/-- The probability that the sum of three rolls of a biased die is divisible by 3 -/
def prob_sum_div_three (d : BiasedDie) : ℝ :=
  d.even_prob^3 + d.odd_prob^3 + 3 * d.even_prob^2 * d.odd_prob

/-- Theorem: The probability that the sum of three rolls of the biased die is divisible by 3 is 7/9 -/
theorem prob_sum_div_three_is_seven_ninths (d : BiasedDie) :
    prob_sum_div_three d = 7/9 := by
  sorry


end prob_sum_div_three_is_seven_ninths_l3277_327773


namespace arcsin_one_over_sqrt_two_l3277_327703

theorem arcsin_one_over_sqrt_two (π : ℝ) : Real.arcsin (1 / Real.sqrt 2) = π / 4 := by
  sorry

end arcsin_one_over_sqrt_two_l3277_327703


namespace complex_sum_theorem_l3277_327764

theorem complex_sum_theorem (B Q R T : ℂ) : 
  B = 3 - 2*I ∧ Q = -5 + 3*I ∧ R = 2*I ∧ T = -1 + 2*I →
  B - Q + R + T = 7 - I :=
by sorry

end complex_sum_theorem_l3277_327764


namespace intersection_complement_equality_l3277_327732

-- Define the sets
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | ∃ y, y = Real.log (x^2 - 1)}
def N : Set ℝ := {x | 0 < x ∧ x < 2}

-- State the theorem
theorem intersection_complement_equality :
  N ∩ (U \ M) = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end intersection_complement_equality_l3277_327732


namespace exact_sixty_possible_greater_than_sixty_possible_l3277_327726

/-- Represents the number of pieces a single piece of paper can be cut into -/
inductive Cut
  | eight : Cut
  | twelve : Cut

/-- Represents a sequence of cuts applied to the original piece of paper -/
def CutSequence := List Cut

/-- Calculates the number of pieces resulting from applying a sequence of cuts -/
def num_pieces (cuts : CutSequence) : ℕ :=
  cuts.foldl (λ acc cut => match cut with
    | Cut.eight => acc * 8
    | Cut.twelve => acc * 12) 1

/-- Theorem stating that it's possible to obtain exactly 60 pieces -/
theorem exact_sixty_possible : ∃ (cuts : CutSequence), num_pieces cuts = 60 := by
  sorry

/-- Theorem stating that it's possible to obtain any number of pieces greater than 60 -/
theorem greater_than_sixty_possible (n : ℕ) (h : n > 60) : 
  ∃ (cuts : CutSequence), num_pieces cuts = n := by
  sorry

end exact_sixty_possible_greater_than_sixty_possible_l3277_327726


namespace arithmetic_sequence_property_l3277_327785

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum : a 4 + a 7 = 2)
  (h_product : a 5 * a 6 = -3) :
  a 1 * a 10 = -323 :=
sorry

end arithmetic_sequence_property_l3277_327785


namespace ice_cream_flavors_count_l3277_327779

/-- The number of ways to distribute n indistinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of unique ice cream flavors -/
def ice_cream_flavors : ℕ := distribute 5 4

theorem ice_cream_flavors_count : ice_cream_flavors = 56 := by sorry

end ice_cream_flavors_count_l3277_327779


namespace smallest_two_digit_factor_of_5082_l3277_327760

theorem smallest_two_digit_factor_of_5082 (a b : ℕ) 
  (h1 : 10 ≤ a) (h2 : a < b) (h3 : b ≤ 99) (h4 : a * b = 5082) : a = 34 := by
  sorry

end smallest_two_digit_factor_of_5082_l3277_327760


namespace rotation_effect_l3277_327781

-- Define a type for the shapes
inductive Shape
  | Triangle
  | Circle
  | Square
  | Pentagon

-- Define a function to represent the initial arrangement
def initial_position (s : Shape) : ℕ :=
  match s with
  | Shape.Triangle => 0
  | Shape.Circle => 1
  | Shape.Square => 2
  | Shape.Pentagon => 3

-- Define a function to represent the position after rotation
def rotated_position (s : Shape) : ℕ :=
  match s with
  | Shape.Triangle => 1
  | Shape.Circle => 2
  | Shape.Square => 3
  | Shape.Pentagon => 0

-- Theorem stating that each shape moves to the next position after rotation
theorem rotation_effect :
  ∀ s : Shape, (rotated_position s) = ((initial_position s) + 1) % 4 :=
by sorry

end rotation_effect_l3277_327781


namespace max_value_sqrt_function_l3277_327719

theorem max_value_sqrt_function (x : ℝ) (h1 : 2 < x) (h2 : x < 5) :
  ∃ (M : ℝ), M = 4 * Real.sqrt 3 ∧ ∀ y, 2 < y → y < 5 → Real.sqrt (3 * y * (8 - y)) ≤ M :=
sorry

end max_value_sqrt_function_l3277_327719


namespace percent_equivalence_l3277_327766

theorem percent_equivalence : ∃ x : ℚ, (60 / 100 * 500 : ℚ) = x / 100 * 600 ∧ x = 50 := by
  sorry

end percent_equivalence_l3277_327766


namespace pencil_cost_l3277_327715

/-- The cost of a pencil given initial and remaining amounts -/
theorem pencil_cost (initial : ℕ) (remaining : ℕ) (h : initial = 15 ∧ remaining = 4) :
  initial - remaining = 11 := by
  sorry

end pencil_cost_l3277_327715


namespace line_intercept_l3277_327784

/-- Given a line y = ax + b passing through the points (3, -2) and (7, 14), prove that b = -14 -/
theorem line_intercept (a b : ℝ) : 
  (∀ x y : ℝ, y = a * x + b) →   -- Definition of the line
  (-2 : ℝ) = a * 3 + b →         -- Line passes through (3, -2)
  (14 : ℝ) = a * 7 + b →         -- Line passes through (7, 14)
  b = -14 := by sorry

end line_intercept_l3277_327784


namespace intersection_of_A_and_complement_of_B_l3277_327734

def A : Set ℝ := {x | x * (x - 2) < 0}
def B : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}

theorem intersection_of_A_and_complement_of_B :
  A ∩ (Set.univ \ B) = Set.Icc 1 2 ∩ Set.Iio 2 :=
sorry

end intersection_of_A_and_complement_of_B_l3277_327734


namespace larger_number_proof_l3277_327736

theorem larger_number_proof (x y : ℝ) (h1 : x + y = 17) (h2 : x - y = 7) : x = 12 := by
  sorry

end larger_number_proof_l3277_327736


namespace max_value_of_f_l3277_327708

def f (x : ℝ) : ℝ := -(x + 1)^2 + 5

theorem max_value_of_f :
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x₀, f x₀ = M) ∧ M = 5 := by
  sorry

end max_value_of_f_l3277_327708


namespace simplification_equivalence_simplified_is_quadratic_trinomial_l3277_327789

-- Define the original polynomial
def original_polynomial (x : ℝ) : ℝ := 2*x^2 - 5*x + x^2 - 4*x + 5

-- Define the simplified polynomial
def simplified_polynomial (x : ℝ) : ℝ := 3*x^2 - 9*x + 5

-- Theorem stating that the simplified polynomial is equivalent to the original
theorem simplification_equivalence :
  ∀ x, original_polynomial x = simplified_polynomial x :=
by sorry

-- Define what it means for a polynomial to be quadratic
def is_quadratic (p : ℝ → ℝ) : Prop :=
  ∃ a b c, a ≠ 0 ∧ ∀ x, p x = a*x^2 + b*x + c

-- Define what it means for a polynomial to have exactly three terms
def has_three_terms (p : ℝ → ℝ) : Prop :=
  ∃ a b c, a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ ∀ x, p x = a*x^2 + b*x + c

-- Theorem stating that the simplified polynomial is a quadratic trinomial
theorem simplified_is_quadratic_trinomial :
  is_quadratic simplified_polynomial ∧ has_three_terms simplified_polynomial :=
by sorry

end simplification_equivalence_simplified_is_quadratic_trinomial_l3277_327789


namespace pure_imaginary_square_root_l3277_327761

theorem pure_imaginary_square_root (a : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (∃ b : ℝ, (1 + a * Complex.I)^2 = b * Complex.I) →
  (a = 1 ∨ a = -1) :=
sorry

end pure_imaginary_square_root_l3277_327761


namespace max_sqrt_sum_l3277_327788

theorem max_sqrt_sum (x : ℝ) (h : 0 ≤ x ∧ x ≤ 20) :
  Real.sqrt (x + 16) + Real.sqrt (20 - x) + 2 * Real.sqrt x ≤ 8 * Real.sqrt 3 / 3 ∧
  ∃ y : ℝ, 0 ≤ y ∧ y ≤ 20 ∧ Real.sqrt (y + 16) + Real.sqrt (20 - y) + 2 * Real.sqrt y = 8 * Real.sqrt 3 / 3 :=
by sorry

end max_sqrt_sum_l3277_327788


namespace no_valid_covering_exists_l3277_327742

/-- Represents a unit square in the chain --/
structure UnitSquare where
  id : Nat
  left_neighbor : Option Nat
  right_neighbor : Option Nat

/-- Represents the chain of squares --/
def SquareChain := List UnitSquare

/-- Represents a vertex on the cube --/
structure CubeVertex where
  x : Fin 4
  y : Fin 4
  z : Fin 4

/-- Represents the 3x3x3 cube --/
def Cube := Set CubeVertex

/-- A covering is a mapping from squares to positions on the cube surface --/
def Covering := UnitSquare → Option CubeVertex

/-- Checks if a covering is valid according to the problem constraints --/
def is_valid_covering (chain : SquareChain) (cube : Cube) (covering : Covering) : Prop :=
  sorry

/-- The main theorem stating that no valid covering exists --/
theorem no_valid_covering_exists (chain : SquareChain) (cube : Cube) :
  chain.length = 54 → ¬∃ (covering : Covering), is_valid_covering chain cube covering :=
sorry

end no_valid_covering_exists_l3277_327742


namespace stock_price_change_l3277_327704

theorem stock_price_change (total_stocks : ℕ) 
  (h1 : total_stocks = 1980)
  (h2 : ∀ s : Fin total_stocks, ∃ (price_yesterday price_today : ℝ), price_yesterday ≠ price_today)
  (h3 : ∃ (higher lower : ℕ), 
    higher + lower = total_stocks ∧ 
    higher = lower + (lower / 5)) :
  ∃ (higher : ℕ), higher = 1080 ∧ 
    ∃ (lower : ℕ), higher + lower = total_stocks ∧ 
    higher = lower + (lower / 5) := by
sorry

end stock_price_change_l3277_327704


namespace lucas_book_purchase_l3277_327735

theorem lucas_book_purchase (total_money : ℚ) (total_books : ℕ) (book_price : ℚ) 
    (h1 : total_money > 0)
    (h2 : total_books > 0)
    (h3 : book_price > 0)
    (h4 : (1 / 4 : ℚ) * total_money = (1 / 2 : ℚ) * total_books * book_price) : 
  total_money - (total_books * book_price) = (1 / 2 : ℚ) * total_money := by
sorry

end lucas_book_purchase_l3277_327735


namespace quadratic_root_implies_a_value_l3277_327791

theorem quadratic_root_implies_a_value (a : ℝ) : 
  (∃ (z : ℂ), z = 2 + I ∧ z^2 - 4*z + a = 0) → a = 5 := by
  sorry

end quadratic_root_implies_a_value_l3277_327791


namespace perpendicular_parallel_implies_parallel_l3277_327707

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_parallel_implies_parallel
  (a b : Line) (α β : Plane)
  (different_lines : a ≠ b)
  (different_planes : α ≠ β)
  (a_perp_α : perpendicular a α)
  (b_perp_β : perpendicular b β)
  (α_parallel_β : parallel_planes α β) :
  parallel_lines a b :=
sorry

end perpendicular_parallel_implies_parallel_l3277_327707


namespace line_slope_equals_k_l3277_327793

/-- 
Given a line passing through points (-1, -4) and (4, k),
if the slope of the line is equal to k, then k = 1.
-/
theorem line_slope_equals_k (k : ℝ) : 
  (k - (-4)) / (4 - (-1)) = k → k = 1 := by sorry

end line_slope_equals_k_l3277_327793


namespace iron_bar_width_is_48_l3277_327741

-- Define the dimensions of the iron bar
def iron_bar_length : ℝ := 12
def iron_bar_height : ℝ := 6

-- Define the number of iron bars and iron balls
def num_iron_bars : ℕ := 10
def num_iron_balls : ℕ := 720

-- Define the volume of each iron ball
def iron_ball_volume : ℝ := 8

-- Theorem statement
theorem iron_bar_width_is_48 (w : ℝ) :
  (num_iron_bars : ℝ) * (iron_bar_length * w * iron_bar_height) =
  (num_iron_balls : ℝ) * iron_ball_volume →
  w = 48 := by sorry

end iron_bar_width_is_48_l3277_327741


namespace minimum_transport_cost_l3277_327768

theorem minimum_transport_cost
  (total_trees : ℕ)
  (chinese_scholar_trees : ℕ)
  (white_pines : ℕ)
  (type_a_capacity_chinese : ℕ)
  (type_a_capacity_pine : ℕ)
  (type_b_capacity : ℕ)
  (type_a_cost : ℕ)
  (type_b_cost : ℕ)
  (total_trucks : ℕ)
  (h1 : total_trees = 320)
  (h2 : chinese_scholar_trees = white_pines + 80)
  (h3 : chinese_scholar_trees + white_pines = total_trees)
  (h4 : type_a_capacity_chinese = 40)
  (h5 : type_a_capacity_pine = 10)
  (h6 : type_b_capacity = 20)
  (h7 : type_a_cost = 400)
  (h8 : type_b_cost = 360)
  (h9 : total_trucks = 8) :
  ∃ (type_a_trucks : ℕ) (type_b_trucks : ℕ),
    type_a_trucks + type_b_trucks = total_trucks ∧
    type_a_trucks * type_a_capacity_chinese + type_b_trucks * type_b_capacity ≥ chinese_scholar_trees ∧
    type_a_trucks * type_a_capacity_pine + type_b_trucks * type_b_capacity ≥ white_pines ∧
    type_a_trucks * type_a_cost + type_b_trucks * type_b_cost = 2960 ∧
    ∀ (other_a : ℕ) (other_b : ℕ),
      other_a + other_b = total_trucks →
      other_a * type_a_capacity_chinese + other_b * type_b_capacity ≥ chinese_scholar_trees →
      other_a * type_a_capacity_pine + other_b * type_b_capacity ≥ white_pines →
      other_a * type_a_cost + other_b * type_b_cost ≥ 2960 :=
by sorry

end minimum_transport_cost_l3277_327768


namespace track_circumference_l3277_327714

/-- The circumference of a circular track given specific running conditions -/
theorem track_circumference (brenda_first_meeting : ℝ) (sally_second_meeting : ℝ) 
  (h1 : brenda_first_meeting = 120)
  (h2 : sally_second_meeting = 180) :
  let circumference := brenda_first_meeting * 3/2
  circumference = 180 := by sorry

end track_circumference_l3277_327714


namespace arithmetic_sequence_terms_l3277_327721

/-- An arithmetic sequence with given properties has 13 terms -/
theorem arithmetic_sequence_terms (a d : ℝ) (n : ℕ) 
  (h1 : 3 * a + 3 * d = 34)
  (h2 : 3 * a + 3 * d * (n - 1) = 146)
  (h3 : n * (2 * a + (n - 1) * d) / 2 = 390) :
  n = 13 := by
  sorry

end arithmetic_sequence_terms_l3277_327721


namespace binomial_floor_divisibility_l3277_327730

theorem binomial_floor_divisibility (n p : ℕ) (h1 : n ≥ p) (h2 : Nat.Prime (50 * p)) : 
  p ∣ (Nat.choose n p - n / p) :=
by sorry

end binomial_floor_divisibility_l3277_327730


namespace monday_calls_l3277_327776

/-- Represents the number of calls answered on each day of the work week -/
structure WeekCalls where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- The average number of calls per day -/
def averageCalls : ℕ := 40

/-- The number of working days in a week -/
def workDays : ℕ := 5

/-- Jean's call data for the week -/
def jeanCalls : WeekCalls := {
  monday := 0,  -- We don't know this value yet
  tuesday := 46,
  wednesday := 27,
  thursday := 61,
  friday := 31
}

theorem monday_calls : jeanCalls.monday = 35 := by sorry

end monday_calls_l3277_327776


namespace jenny_walking_distance_l3277_327711

theorem jenny_walking_distance (ran_distance : ℝ) (extra_distance : ℝ) :
  ran_distance = 0.6 →
  extra_distance = 0.2 →
  ran_distance = (ran_distance - extra_distance) + extra_distance →
  ran_distance - extra_distance = 0.4 :=
by
  sorry

end jenny_walking_distance_l3277_327711


namespace two_digit_number_subtraction_l3277_327795

theorem two_digit_number_subtraction (b : ℕ) (h1 : b < 9) : 
  (11 * b + 10) - (11 * b + 1) = 9 := by sorry

end two_digit_number_subtraction_l3277_327795


namespace prob_diameter_given_smoothness_correct_prob_smoothness_given_diameter_correct_l3277_327777

/-- Represents the total number of circular parts -/
def total_parts : ℕ := 100

/-- Represents the number of parts with qualified diameters -/
def qualified_diameter : ℕ := 98

/-- Represents the number of parts with qualified smoothness -/
def qualified_smoothness : ℕ := 96

/-- Represents the number of parts with both qualified diameter and smoothness -/
def qualified_both : ℕ := 94

/-- Calculates the probability of qualified diameter given qualified smoothness -/
def prob_diameter_given_smoothness : ℚ :=
  qualified_both / qualified_smoothness

/-- Calculates the probability of qualified smoothness given qualified diameter -/
def prob_smoothness_given_diameter : ℚ :=
  qualified_both / qualified_diameter

theorem prob_diameter_given_smoothness_correct :
  prob_diameter_given_smoothness = 94 / 96 := by sorry

theorem prob_smoothness_given_diameter_correct :
  prob_smoothness_given_diameter = 94 / 98 := by sorry

end prob_diameter_given_smoothness_correct_prob_smoothness_given_diameter_correct_l3277_327777


namespace regular_polygon_150_degrees_has_12_sides_l3277_327775

/-- A regular polygon with interior angles measuring 150° has 12 sides -/
theorem regular_polygon_150_degrees_has_12_sides :
  ∀ n : ℕ,
  n > 2 →
  (∀ θ : ℝ, θ = 150 → n * θ = (n - 2) * 180) →
  n = 12 :=
by sorry

end regular_polygon_150_degrees_has_12_sides_l3277_327775


namespace meeting_participants_ratio_l3277_327794

/-- Given information about participants in a meeting, prove the ratio of female democrats to total female participants -/
theorem meeting_participants_ratio :
  let total_participants : ℕ := 810
  let female_democrats : ℕ := 135
  let male_democrat_ratio : ℚ := 1/4
  let total_democrat_ratio : ℚ := 1/3
  ∃ (female_participants male_participants : ℕ),
    female_participants + male_participants = total_participants ∧
    female_democrats + male_democrat_ratio * male_participants = total_democrat_ratio * total_participants ∧
    female_democrats / female_participants = 1/2 := by
  sorry

end meeting_participants_ratio_l3277_327794


namespace orange_crates_pigeonhole_l3277_327712

theorem orange_crates_pigeonhole (total_crates : ℕ) (min_oranges max_oranges : ℕ) :
  total_crates = 200 →
  min_oranges = 100 →
  max_oranges = 130 →
  ∃ (n : ℕ), n ≥ 7 ∧ 
    ∃ (k : ℕ), min_oranges ≤ k ∧ k ≤ max_oranges ∧
      (∃ (crates : Finset (Fin total_crates)), crates.card = n ∧ 
        ∀ c ∈ crates, ∃ f : Fin total_crates → ℕ, f c = k) :=
by sorry

end orange_crates_pigeonhole_l3277_327712


namespace fiftieth_digit_of_seventh_l3277_327710

/-- The decimal representation of 1/7 as a list of digits -/
def seventhDecimal : List Nat := [1, 4, 2, 8, 5, 7]

/-- The length of the repeating part in the decimal representation of 1/7 -/
def repeatLength : Nat := 6

/-- The 50th digit after the decimal point in the decimal representation of 1/7 -/
def fiftiethDigit : Nat := seventhDecimal[(50 - 1) % repeatLength]

theorem fiftieth_digit_of_seventh :
  fiftiethDigit = 4 := by sorry

end fiftieth_digit_of_seventh_l3277_327710


namespace min_value_of_g_l3277_327797

-- Define the properties of odd and even functions
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def IsEven (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- State the theorem
theorem min_value_of_g (f g : ℝ → ℝ) 
  (h_odd : IsOdd f) (h_even : IsEven g) 
  (h_sum : ∀ x, f x + g x = 2^x) : 
  ∃ m, m = 1 ∧ ∀ x, g x ≥ m :=
sorry

end min_value_of_g_l3277_327797


namespace cone_cylinder_volume_ratio_l3277_327750

/-- The ratio of the volume of a cone to the volume of a cylinder with shared base radius -/
theorem cone_cylinder_volume_ratio 
  (r : ℝ) 
  (h_cyl h_cone : ℝ) 
  (h_r : r = 5)
  (h_h_cyl : h_cyl = 18)
  (h_h_cone : h_cone = 9) :
  (1 / 3 * π * r^2 * h_cone) / (π * r^2 * h_cyl) = 1 / 6 := by
  sorry


end cone_cylinder_volume_ratio_l3277_327750


namespace history_book_cost_l3277_327748

theorem history_book_cost 
  (total_books : ℕ) 
  (math_book_cost : ℕ) 
  (total_price : ℕ) 
  (math_books_bought : ℕ) 
  (h1 : total_books = 80)
  (h2 : math_book_cost = 4)
  (h3 : total_price = 390)
  (h4 : math_books_bought = 10) :
  (total_price - math_books_bought * math_book_cost) / (total_books - math_books_bought) = 5 := by
sorry

end history_book_cost_l3277_327748


namespace regular_hexagon_perimeter_l3277_327747

theorem regular_hexagon_perimeter (s : ℝ) (h : s > 0) : 
  (3 * Real.sqrt 3 / 2) * s^2 = s → 6 * s = 4 * Real.sqrt 3 / 3 := by
  sorry

end regular_hexagon_perimeter_l3277_327747


namespace profit_equation_correct_l3277_327757

/-- Represents the profit calculation for a bicycle sale --/
def profit_equation (x : ℝ) : Prop :=
  0.8 * (1 + 0.45) * x - x = 50

/-- Theorem stating that the profit equation correctly represents the given scenario --/
theorem profit_equation_correct (x : ℝ) : profit_equation x ↔ 
  (∃ (markup discount profit : ℝ),
    markup = 0.45 ∧
    discount = 0.2 ∧
    profit = 50 ∧
    (1 - discount) * (1 + markup) * x - x = profit) :=
by sorry

end profit_equation_correct_l3277_327757


namespace students_neither_football_nor_cricket_l3277_327780

theorem students_neither_football_nor_cricket 
  (total : ℕ) 
  (football : ℕ) 
  (cricket : ℕ) 
  (both : ℕ) 
  (h1 : total = 450) 
  (h2 : football = 325) 
  (h3 : cricket = 175) 
  (h4 : both = 100) : 
  total - (football + cricket - both) = 50 := by
  sorry

end students_neither_football_nor_cricket_l3277_327780


namespace triangle_side_length_l3277_327751

-- Define the triangle ABC
def triangle (A B C : ℝ × ℝ) : Prop :=
  -- Add conditions to ensure it's a valid triangle
  true

-- Define the angle measure in degrees
def angle_measure (A B C : ℝ × ℝ) : ℝ := 
  -- Add definition for angle measure
  0

-- Define the distance between two points
def distance (A B : ℝ × ℝ) : ℝ :=
  -- Add definition for distance
  0

theorem triangle_side_length 
  (A B C : ℝ × ℝ) 
  (h_triangle : triangle A B C)
  (h_angle_B : angle_measure A B C = 45)
  (h_angle_C : angle_measure B C A = 80)
  (h_side_AC : distance A C = 5) :
  distance B C = (10 * Real.sin (55 * π / 180)) / Real.sqrt 2 :=
by
  sorry

end triangle_side_length_l3277_327751


namespace marias_stamps_l3277_327717

theorem marias_stamps (S : ℕ) : 
  S > 1 ∧ 
  S % 9 = 1 ∧ 
  S % 10 = 1 ∧ 
  S % 11 = 1 ∧
  (∀ T : ℕ, T > 1 ∧ T % 9 = 1 ∧ T % 10 = 1 ∧ T % 11 = 1 → S ≤ T) → 
  S = 991 := by
sorry

end marias_stamps_l3277_327717


namespace teacher_age_l3277_327778

theorem teacher_age (n : ℕ) (initial_avg : ℚ) (transfer_age : ℕ) (new_avg : ℚ) :
  n = 45 →
  initial_avg = 14 →
  transfer_age = 15 →
  new_avg = 14.66 →
  let remaining_students := n - 1
  let total_age := n * initial_avg
  let remaining_age := total_age - transfer_age
  let teacher_age := (remaining_students + 1) * new_avg - remaining_age
  (∀ p : ℕ, Prime p → p > teacher_age → p ≥ 17) ∧
  Prime 17 ∧
  17 > teacher_age :=
by sorry

end teacher_age_l3277_327778


namespace profit_percentage_calculation_l3277_327787

/-- Calculate the profit percentage given the selling price and cost price -/
theorem profit_percentage_calculation (selling_price cost_price : ℚ) :
  selling_price = 1800 ∧ cost_price = 1500 →
  (selling_price - cost_price) / cost_price * 100 = 20 := by
  sorry

end profit_percentage_calculation_l3277_327787


namespace sequence_2019th_term_l3277_327706

theorem sequence_2019th_term (a : ℕ → ℤ) 
  (h1 : a 1 = 2) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n - 2) : 
  a 2019 = -4034 := by
  sorry

end sequence_2019th_term_l3277_327706


namespace min_value_of_a_l3277_327799

theorem min_value_of_a (a : ℝ) : 
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 1 → 1/x + a/y ≥ 4) → 
  a ≥ 1 := by
sorry

end min_value_of_a_l3277_327799


namespace license_plate_count_l3277_327737

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The total number of characters (letters + digits) -/
def num_chars : ℕ := num_letters + num_digits

/-- The number of possible license plates -/
def num_license_plates : ℕ := num_letters * num_chars * 1 * num_digits

theorem license_plate_count :
  num_license_plates = 9360 :=
by sorry

end license_plate_count_l3277_327737


namespace boat_speed_ratio_l3277_327752

/-- The ratio of average speed to still water speed for a boat trip --/
theorem boat_speed_ratio 
  (still_water_speed : ℝ) 
  (current_speed : ℝ) 
  (downstream_distance : ℝ) 
  (upstream_distance : ℝ) 
  (h1 : still_water_speed = 20)
  (h2 : current_speed = 4)
  (h3 : downstream_distance = 5)
  (h4 : upstream_distance = 3) :
  let downstream_speed := still_water_speed + current_speed
  let upstream_speed := still_water_speed - current_speed
  let total_distance := downstream_distance + upstream_distance
  let total_time := downstream_distance / downstream_speed + upstream_distance / upstream_speed
  let average_speed := total_distance / total_time
  average_speed / still_water_speed = 96 / 95 :=
by sorry

end boat_speed_ratio_l3277_327752


namespace school_bus_problem_l3277_327728

theorem school_bus_problem (total_distance bus_speed walking_speed rest_time : ℝ) 
  (h_total : total_distance = 21)
  (h_bus : bus_speed = 60)
  (h_walk : walking_speed = 4)
  (h_rest : rest_time = 1/6) :
  ∃ (x : ℝ), 
    (x = 19 ∧ total_distance - x = 2) ∧ 
    ((total_distance - x) / walking_speed + rest_time = (total_distance + x) / bus_speed) :=
by sorry

end school_bus_problem_l3277_327728


namespace tank_capacity_l3277_327798

theorem tank_capacity : ∀ (T : ℚ),
  (3/4 : ℚ) * T + 7 = (7/8 : ℚ) * T →
  T = 56 := by sorry

end tank_capacity_l3277_327798


namespace max_sum_surrounding_45_l3277_327727

theorem max_sum_surrounding_45 (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℕ) : 
  a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₁ ≠ a₅ ∧ a₁ ≠ a₆ ∧ a₁ ≠ a₇ ∧ a₁ ≠ a₈ ∧
  a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₂ ≠ a₅ ∧ a₂ ≠ a₆ ∧ a₂ ≠ a₇ ∧ a₂ ≠ a₈ ∧
  a₃ ≠ a₄ ∧ a₃ ≠ a₅ ∧ a₃ ≠ a₆ ∧ a₃ ≠ a₇ ∧ a₃ ≠ a₈ ∧
  a₄ ≠ a₅ ∧ a₄ ≠ a₆ ∧ a₄ ≠ a₇ ∧ a₄ ≠ a₈ ∧
  a₅ ≠ a₆ ∧ a₅ ≠ a₇ ∧ a₅ ≠ a₈ ∧
  a₆ ≠ a₇ ∧ a₆ ≠ a₈ ∧
  a₇ ≠ a₈ ∧
  0 < a₁ ∧ 0 < a₂ ∧ 0 < a₃ ∧ 0 < a₄ ∧ 0 < a₅ ∧ 0 < a₆ ∧ 0 < a₇ ∧ 0 < a₈ ∧
  a₁ * 45 * a₅ = 3240 ∧
  a₂ * 45 * a₆ = 3240 ∧
  a₃ * 45 * a₇ = 3240 ∧
  a₄ * 45 * a₈ = 3240 →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ ≤ 160 :=
by sorry

end max_sum_surrounding_45_l3277_327727


namespace seven_students_distribution_l3277_327716

/-- The number of ways to distribute n students into two dormitories,
    with each dormitory having at least m students -/
def distribution_count (n : ℕ) (m : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 112 ways to distribute 7 students into two dormitories,
    with each dormitory having at least 2 students -/
theorem seven_students_distribution :
  distribution_count 7 2 = 112 := by
  sorry

end seven_students_distribution_l3277_327716


namespace rational_function_value_l3277_327756

-- Define the function types
def linear_function (α : Type*) [Ring α] := α → α
def quadratic_function (α : Type*) [Ring α] := α → α
def rational_function (α : Type*) [Field α] := α → α

-- Define the properties of the rational function
def has_vertical_asymptotes (f : rational_function ℝ) (a b : ℝ) : Prop :=
  ∀ (x : ℝ), x ≠ a ∧ x ≠ b → f x ≠ 0

def passes_through (f : rational_function ℝ) (x y : ℝ) : Prop :=
  f x = y

-- Main theorem
theorem rational_function_value
  (p : linear_function ℝ)
  (q : quadratic_function ℝ)
  (f : rational_function ℝ)
  (h1 : ∀ (x : ℝ), f x = p x / q x)
  (h2 : has_vertical_asymptotes f (-1) 4)
  (h3 : passes_through f 0 0)
  (h4 : passes_through f 1 (-3)) :
  p (-2) / q (-2) = -6 :=
sorry

end rational_function_value_l3277_327756


namespace absolute_value_inequality_l3277_327744

-- Define the solution set
def solution_set : Set ℝ := {x | x > 3 ∨ x < -1}

-- State the theorem
theorem absolute_value_inequality :
  {x : ℝ | |x - 1| > 2} = solution_set := by sorry

end absolute_value_inequality_l3277_327744


namespace canoe_weight_problem_l3277_327745

theorem canoe_weight_problem (canoe_capacity : ℕ) (dog_weight_ratio : ℚ) (total_weight : ℕ) :
  canoe_capacity = 6 →
  dog_weight_ratio = 1/4 →
  total_weight = 595 →
  ∃ (person_weight : ℕ),
    person_weight = 140 ∧
    (↑(2 * canoe_capacity) / 3 : ℚ).floor * person_weight + 
    (dog_weight_ratio * person_weight).num / (dog_weight_ratio * person_weight).den = total_weight :=
by sorry

end canoe_weight_problem_l3277_327745


namespace clara_cookie_sales_clara_total_cookies_l3277_327720

theorem clara_cookie_sales : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ
  | cookies_per_box1, cookies_per_box2, cookies_per_box3,
    boxes_sold1, boxes_sold2, boxes_sold3 =>
  cookies_per_box1 * boxes_sold1 +
  cookies_per_box2 * boxes_sold2 +
  cookies_per_box3 * boxes_sold3

theorem clara_total_cookies :
  clara_cookie_sales 12 20 16 50 80 70 = 3320 := by
  sorry

end clara_cookie_sales_clara_total_cookies_l3277_327720


namespace complex_fraction_simplification_l3277_327772

theorem complex_fraction_simplification :
  (5 + 7 * Complex.I) / (2 + 3 * Complex.I) = 31/13 - 1/13 * Complex.I := by
  sorry

end complex_fraction_simplification_l3277_327772


namespace swimming_speed_in_still_water_l3277_327731

/-- The swimming speed of a person in still water, given their performance against a current. -/
theorem swimming_speed_in_still_water (water_speed : ℝ) (distance : ℝ) (time : ℝ) 
  (h1 : water_speed = 4)
  (h2 : distance = 8)
  (h3 : time = 2)
  (h4 : (swimming_speed - water_speed) * time = distance) :
  swimming_speed = 8 :=
sorry

end swimming_speed_in_still_water_l3277_327731


namespace egg_transfer_proof_l3277_327770

/-- Proves that transferring 24 eggs from basket B to basket A will make the number of eggs in basket A twice the number of eggs in basket B -/
theorem egg_transfer_proof (initial_A initial_B transferred : ℕ) 
  (h1 : initial_A = 54)
  (h2 : initial_B = 63)
  (h3 : transferred = 24) :
  initial_A + transferred = 2 * (initial_B - transferred) := by
  sorry

end egg_transfer_proof_l3277_327770


namespace y_value_when_x_is_zero_l3277_327767

theorem y_value_when_x_is_zero (t : ℚ) (x y : ℚ) 
  (h1 : x = 3 - 2*t) 
  (h2 : y = 3*t + 6) 
  (h3 : x = 0) : 
  y = 21/2 := by
  sorry

end y_value_when_x_is_zero_l3277_327767


namespace marble_count_l3277_327709

theorem marble_count (total : ℕ) (red blue white : ℕ) : 
  total = 108 →
  blue = red / 3 →
  white = blue / 2 →
  red + blue + white = total →
  white < red ∧ white < blue :=
by sorry

end marble_count_l3277_327709


namespace line_equation_proof_l3277_327701

/-- Given a line described by the vector equation (3, -4) · ((x, y) - (-2, 8)) = 0,
    prove that its slope-intercept form y = mx + b has m = 3/4 and b = 19/2. -/
theorem line_equation_proof :
  let vector_eq := fun (x y : ℝ) => 3 * (x + 2) + (-4) * (y - 8) = 0
  ∃ m b : ℝ, (∀ x y : ℝ, vector_eq x y ↔ y = m * x + b) ∧ m = 3/4 ∧ b = 19/2 := by
  sorry

end line_equation_proof_l3277_327701


namespace stating_retirement_benefit_formula_l3277_327743

/-- Represents the retirement benefit calculation for a teacher. -/
structure TeacherBenefit where
  /-- The number of years the teacher has taught. -/
  y : ℝ
  /-- The proportionality constant for the benefit calculation. -/
  k : ℝ
  /-- The additional years in the first scenario. -/
  c : ℝ
  /-- The additional years in the second scenario. -/
  d : ℝ
  /-- The benefit increase in the first scenario. -/
  r : ℝ
  /-- The benefit increase in the second scenario. -/
  s : ℝ
  /-- Ensures that c and d are different. -/
  h_c_neq_d : c ≠ d
  /-- The benefit is proportional to the square root of years taught. -/
  h_benefit : k * Real.sqrt y > 0
  /-- The equation for the first scenario. -/
  h_eq1 : k * Real.sqrt (y + c) = k * Real.sqrt y + r
  /-- The equation for the second scenario. -/
  h_eq2 : k * Real.sqrt (y + d) = k * Real.sqrt y + s

/-- 
Theorem stating that the original annual retirement benefit 
is equal to (s² - r²) / (2(s - r)) given the conditions.
-/
theorem retirement_benefit_formula (tb : TeacherBenefit) : 
  tb.k * Real.sqrt tb.y = (tb.s^2 - tb.r^2) / (2 * (tb.s - tb.r)) := by
  sorry


end stating_retirement_benefit_formula_l3277_327743


namespace carbon_atoms_count_l3277_327723

/-- Represents the number of atoms of each element in the compound -/
structure CompoundComposition where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ

/-- Atomic weights of elements in atomic mass units (amu) -/
def atomic_weight : CompoundComposition → ℕ
  | ⟨c, h, o⟩ => 12 * c + 1 * h + 16 * o

/-- The compound has 1 Hydrogen and 1 Oxygen atom -/
def compound_constraints (comp : CompoundComposition) : Prop :=
  comp.hydrogen = 1 ∧ comp.oxygen = 1

/-- The molecular weight of the compound is 65 amu -/
def molecular_weight_constraint (comp : CompoundComposition) : Prop :=
  atomic_weight comp = 65

theorem carbon_atoms_count :
  ∀ comp : CompoundComposition,
    compound_constraints comp →
    molecular_weight_constraint comp →
    comp.carbon = 4 :=
by sorry

end carbon_atoms_count_l3277_327723


namespace x_cubed_coefficient_in_binomial_difference_x_cubed_coefficient_is_negative_ten_l3277_327755

theorem x_cubed_coefficient_in_binomial_difference : ℤ :=
  let n₁ : ℕ := 5
  let n₂ : ℕ := 6
  let k : ℕ := 3
  let coeff₁ : ℤ := (Nat.choose n₁ k : ℤ)
  let coeff₂ : ℤ := (Nat.choose n₂ k : ℤ)
  coeff₁ - coeff₂

theorem x_cubed_coefficient_is_negative_ten :
  x_cubed_coefficient_in_binomial_difference = -10 := by
  sorry

end x_cubed_coefficient_in_binomial_difference_x_cubed_coefficient_is_negative_ten_l3277_327755
