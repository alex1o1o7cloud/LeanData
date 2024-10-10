import Mathlib

namespace hyperbola_equation_l2389_238950

/-- Proves that given a parabola y^2 = 8x whose latus rectum passes through a focus of a hyperbola
    x^2/a^2 - y^2/b^2 = 1 (a > 0, b > 0), and one asymptote of the hyperbola is x + √3y = 0,
    the equation of the hyperbola is x^2/3 - y^2 = 1. -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), y^2 = 8*x ∧ x = -2) →  -- Latus rectum of parabola
  (∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1) →  -- Hyperbola equation
  (∃ (x y : ℝ), x + Real.sqrt 3 * y = 0) →  -- Asymptote equation
  (∀ (x y : ℝ), x^2/3 - y^2 = 1) :=  -- Resulting hyperbola equation
by sorry


end hyperbola_equation_l2389_238950


namespace partial_fraction_decomposition_l2389_238993

theorem partial_fraction_decomposition :
  ∃ (A B : ℝ),
    (∀ x : ℝ, x ≠ 12 ∧ x ≠ -3 →
      (6 * x + 3) / (x^2 - 9 * x - 36) = A / (x - 12) + B / (x + 3)) ∧
    A = 5 ∧
    B = 1 := by
  sorry

end partial_fraction_decomposition_l2389_238993


namespace lines_parallel_iff_m_eq_one_l2389_238990

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel_lines (m : ℝ) : Prop :=
  (-(1 : ℝ) / (1 + m) = -(m / 2))

/-- The first line equation -/
def line1 (m x y : ℝ) : Prop :=
  x + (1 + m) * y = 2 - m

/-- The second line equation -/
def line2 (m x y : ℝ) : Prop :=
  m * x + 2 * y + 8 = 0

/-- The theorem stating that the lines are parallel if and only if m = 1 -/
theorem lines_parallel_iff_m_eq_one :
  ∀ m : ℝ, (∃ x y : ℝ, line1 m x y ∧ line2 m x y) →
    (parallel_lines m ↔ m = 1) :=
by sorry

end lines_parallel_iff_m_eq_one_l2389_238990


namespace quadratic_inequality_range_l2389_238972

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + (a - 1)*x + 1 < 0) → (a > 3 ∨ a < -1) := by
sorry

end quadratic_inequality_range_l2389_238972


namespace expression_evaluation_l2389_238994

theorem expression_evaluation (a b : ℝ) (h1 : a = 2) (h2 : b = -1) :
  ((2*a + 3*b) * (2*a - 3*b) - (2*a - b)^2 - 2*a*b) / (-2*b) = -7 := by
  sorry

end expression_evaluation_l2389_238994


namespace equation_solutions_l2389_238902

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 16 = 0 ↔ x = 4 ∨ x = -4) ∧
  (∀ x : ℝ, (x + 10)^3 + 27 = 0 ↔ x = -13) :=
by sorry

end equation_solutions_l2389_238902


namespace fraction_simplification_l2389_238945

theorem fraction_simplification (y : ℝ) (h : y = 5) : 
  (y^4 - 8*y^2 + 16) / (y^2 - 4) = 21 := by
  sorry

end fraction_simplification_l2389_238945


namespace function_identities_equivalence_l2389_238924

theorem function_identities_equivalence (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y) = f x + f y) ↔
  (∀ x y : ℝ, f (x * y + x + y) = f (x * y) + f x + f y) :=
by sorry

end function_identities_equivalence_l2389_238924


namespace exam_maximum_marks_l2389_238958

theorem exam_maximum_marks :
  let pass_percentage : ℚ := 90 / 100
  let student_marks : ℕ := 250
  let failing_margin : ℕ := 300
  let maximum_marks : ℕ := 612
  (pass_percentage * maximum_marks : ℚ) = (student_marks + failing_margin : ℚ) ∧
  maximum_marks = (student_marks + failing_margin : ℚ) / pass_percentage :=
by sorry

end exam_maximum_marks_l2389_238958


namespace second_number_proof_l2389_238987

theorem second_number_proof (a b : ℝ) (h1 : a = 50) (h2 : 0.6 * a - 0.3 * b = 27) : b = 10 := by
  sorry

end second_number_proof_l2389_238987


namespace slope_range_for_given_inclination_l2389_238999

theorem slope_range_for_given_inclination (α : Real) (h : α ∈ Set.Icc (π / 4) (3 * π / 4)) :
  let k := Real.tan α
  k ∈ Set.Iic (-1) ∪ Set.Ici 1 :=
sorry

end slope_range_for_given_inclination_l2389_238999


namespace negative_expression_l2389_238944

theorem negative_expression : 
  (-(-3) > 0) ∧ (-3^2 < 0) ∧ ((-3)^2 > 0) ∧ (|(-3)| > 0) :=
by sorry


end negative_expression_l2389_238944


namespace karen_took_one_sixth_l2389_238997

/-- 
Given:
- Sasha added 48 cards to a box
- There were originally 43 cards in the box
- There are now 83 cards in the box

Prove that the fraction of cards Karen took out is 1/6
-/
theorem karen_took_one_sixth (cards_added : ℕ) (original_cards : ℕ) (final_cards : ℕ) 
  (h1 : cards_added = 48)
  (h2 : original_cards = 43)
  (h3 : final_cards = 83) :
  (cards_added + original_cards - final_cards : ℚ) / cards_added = 1 / 6 := by
  sorry

end karen_took_one_sixth_l2389_238997


namespace sum_f_2016_2017_2018_l2389_238908

/-- An odd periodic function with period 4 and f(1) = 1 -/
def f : ℝ → ℝ :=
  sorry

/-- f is an odd function -/
axiom f_odd : ∀ x, f (-x) = -f x

/-- f has period 4 -/
axiom f_periodic : ∀ x, f (x + 4) = f x

/-- f(1) = 1 -/
axiom f_one : f 1 = 1

theorem sum_f_2016_2017_2018 : f 2016 + f 2017 + f 2018 = 1 := by
  sorry

end sum_f_2016_2017_2018_l2389_238908


namespace range_of_a_l2389_238992

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x > a^2 - a - 3) → 
  a > -1 ∧ a < 2 :=
by sorry

end range_of_a_l2389_238992


namespace exists_uncovered_cell_l2389_238935

/-- Represents a grid cell --/
structure Cell :=
  (x : Nat) (y : Nat)

/-- Represents a rectangle --/
structure Rectangle :=
  (width : Nat) (height : Nat)

/-- The dimensions of the grid --/
def gridWidth : Nat := 11
def gridHeight : Nat := 1117

/-- The dimensions of the cutting rectangle --/
def cuttingRectangle : Rectangle := { width := 6, height := 1 }

/-- A function to check if a cell is covered by a rectangle --/
def isCovered (c : Cell) (r : Rectangle) (position : Cell) : Prop :=
  c.x ≥ position.x ∧ c.x < position.x + r.width ∧
  c.y ≥ position.y ∧ c.y < position.y + r.height

/-- The main theorem --/
theorem exists_uncovered_cell :
  ∃ (c : Cell), c.x < gridWidth ∧ c.y < gridHeight ∧
  ∀ (arrangements : List Cell),
    ∃ (p : Cell), p ∈ arrangements →
      ¬(isCovered c cuttingRectangle p) :=
sorry

end exists_uncovered_cell_l2389_238935


namespace complement_angle_l2389_238970

theorem complement_angle (A : ℝ) (h : A = 25) : 90 - A = 65 := by
  sorry

end complement_angle_l2389_238970


namespace number_puzzle_l2389_238978

theorem number_puzzle : ∃ x : ℤ, x - 29 + 64 = 76 ∧ x = 41 := by sorry

end number_puzzle_l2389_238978


namespace triangle_area_similarity_l2389_238957

-- Define the triangles
variable (A B C D E F : ℝ × ℝ)

-- Define the similarity relation
def similar (t1 t2 : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : Prop := sorry

-- Define the area function
def area (t : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : ℝ := sorry

-- Define the side length function
def side_length (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_area_similarity :
  similar (A, B, C) (D, E, F) →
  side_length A B / side_length D E = 2 →
  area (A, B, C) = 8 →
  area (D, E, F) = 2 := by
sorry

end triangle_area_similarity_l2389_238957


namespace shipment_composition_l2389_238933

/-- Represents a shipment of boxes with two possible weights -/
structure Shipment where
  total_boxes : ℕ
  weight1 : ℕ
  weight2 : ℕ
  count1 : ℕ
  count2 : ℕ
  initial_avg : ℚ

/-- Theorem about the composition of a specific shipment -/
theorem shipment_composition (s : Shipment) 
  (h1 : s.total_boxes = 30)
  (h2 : s.weight1 = 10)
  (h3 : s.weight2 = 20)
  (h4 : s.initial_avg = 18)
  (h5 : s.count1 + s.count2 = s.total_boxes)
  (h6 : s.weight1 * s.count1 + s.weight2 * s.count2 = s.initial_avg * s.total_boxes) :
  s.count1 = 6 ∧ s.count2 = 24 := by
  sorry

/-- Function to calculate the number of heavy boxes to remove to reach a target average -/
def boxes_to_remove (s : Shipment) (target_avg : ℚ) : ℕ := by
  sorry

end shipment_composition_l2389_238933


namespace cinnamon_swirls_distribution_l2389_238996

theorem cinnamon_swirls_distribution (total_pieces : ℕ) (num_people : ℕ) (pieces_per_person : ℕ) : 
  total_pieces = 12 → 
  num_people = 3 → 
  total_pieces = num_people * pieces_per_person →
  pieces_per_person = 4 := by
  sorry

end cinnamon_swirls_distribution_l2389_238996


namespace triangle_side_length_l2389_238981

theorem triangle_side_length (a b c : ℝ) (C : ℝ) : 
  a = 2 → b = 3 → C = Real.pi / 3 → c = Real.sqrt 7 := by sorry

end triangle_side_length_l2389_238981


namespace inequalities_given_sum_positive_l2389_238982

theorem inequalities_given_sum_positive (a b : ℝ) (h : a + b > 0) : 
  (a^5 * b^2 + a^4 * b^3 ≥ 0) ∧ 
  (a^21 + b^21 > 0) ∧ 
  ((a+2)*(b+2) > a*b) := by
  sorry

end inequalities_given_sum_positive_l2389_238982


namespace londolozi_lion_population_l2389_238991

/-- Calculates the lion population after a given number of months -/
def lionPopulation (initialPopulation birthRate deathRate months : ℕ) : ℕ :=
  initialPopulation + birthRate * months - deathRate * months

/-- Theorem: The lion population in Londolozi after 12 months -/
theorem londolozi_lion_population :
  lionPopulation 100 5 1 12 = 148 := by
  sorry

#eval lionPopulation 100 5 1 12

end londolozi_lion_population_l2389_238991


namespace money_sum_l2389_238922

theorem money_sum (a b : ℝ) (h1 : (3/10) * a = (1/5) * b) (h2 : b = 60) : a + b = 100 := by
  sorry

end money_sum_l2389_238922


namespace monotone_decreasing_implies_a_leq_3_l2389_238964

/-- The function f(x) = x^2 + 4ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 4*a*x + 2

/-- The theorem stating that if f(x) is monotonically decreasing in (-∞, 6), then a ≤ 3 -/
theorem monotone_decreasing_implies_a_leq_3 (a : ℝ) :
  (∀ x y, x < y → y < 6 → f a x > f a y) → a ≤ 3 := by
  sorry

end monotone_decreasing_implies_a_leq_3_l2389_238964


namespace distance_X_to_CD_l2389_238966

/-- Square with side length 2s and quarter-circle arcs -/
structure SquareWithArcs (s : ℝ) :=
  (A B C D : ℝ × ℝ)
  (X : ℝ × ℝ)
  (h_square : A = (0, 0) ∧ B = (2*s, 0) ∧ C = (2*s, 2*s) ∧ D = (0, 2*s))
  (h_arc_A : (X.1 - A.1)^2 + (X.2 - A.2)^2 = (2*s)^2)
  (h_arc_B : (X.1 - B.1)^2 + (X.2 - B.2)^2 = (2*s)^2)
  (h_X_inside : 0 < X.1 ∧ X.1 < 2*s ∧ 0 < X.2 ∧ X.2 < 2*s)

/-- The distance from X to side CD in a SquareWithArcs is 2s(2 - √3) -/
theorem distance_X_to_CD (s : ℝ) (sq : SquareWithArcs s) :
  2*s - sq.X.2 = 2*s*(2 - Real.sqrt 3) :=
sorry

end distance_X_to_CD_l2389_238966


namespace binomial_1000_1000_l2389_238984

theorem binomial_1000_1000 : Nat.choose 1000 1000 = 1 := by
  sorry

end binomial_1000_1000_l2389_238984


namespace rotation_180_transforms_rectangle_l2389_238977

-- Define the points of rectangle ABCD
def A : ℝ × ℝ := (-3, 2)
def B : ℝ × ℝ := (-1, 2)
def C : ℝ × ℝ := (-1, 5)
def D : ℝ × ℝ := (-3, 5)

-- Define the points of rectangle A'B'C'D'
def A' : ℝ × ℝ := (3, -2)
def B' : ℝ × ℝ := (1, -2)
def C' : ℝ × ℝ := (1, -5)
def D' : ℝ × ℝ := (3, -5)

-- Define the 180° rotation transformation
def rotate180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

-- Theorem statement
theorem rotation_180_transforms_rectangle :
  rotate180 A = A' ∧
  rotate180 B = B' ∧
  rotate180 C = C' ∧
  rotate180 D = D' := by
  sorry


end rotation_180_transforms_rectangle_l2389_238977


namespace geometric_sequence_sum_constant_l2389_238928

/-- A geometric sequence with a specific sum formula -/
structure GeometricSequence where
  a : ℕ → ℝ
  sum : ℕ → ℝ
  sum_formula : ∀ n, sum n = 3^(n + 1) + a 1
  is_geometric : ∀ n, a (n + 2) * a n = (a (n + 1))^2

/-- The value of 'a' in the sum formula is -3 -/
theorem geometric_sequence_sum_constant (seq : GeometricSequence) : seq.a 1 - 9 = -3 := by
  sorry

end geometric_sequence_sum_constant_l2389_238928


namespace unique_solution_l2389_238989

/-- Represents the arithmetic operations and equality --/
inductive Operation
| Add
| Sub
| Mul
| Div
| Eq

/-- The set of equations given in the problem --/
def Equations (A B C D E : Operation) : Prop :=
  (4 / 2 = 2) ∧
  (8 = 4 * 2) ∧
  (2 + 3 = 5) ∧
  (4 = 5 - 1) ∧
  (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧ (A ≠ E) ∧
  (B ≠ C) ∧ (B ≠ D) ∧ (B ≠ E) ∧
  (C ≠ D) ∧ (C ≠ E) ∧
  (D ≠ E)

/-- The theorem stating the unique solution to the problem --/
theorem unique_solution :
  ∃! (A B C D E : Operation),
    Equations A B C D E ∧
    A = Operation.Div ∧
    B = Operation.Eq ∧
    C = Operation.Mul ∧
    D = Operation.Add ∧
    E = Operation.Sub := by sorry

end unique_solution_l2389_238989


namespace quadrangular_pyramid_edge_length_l2389_238936

/-- A quadrangular pyramid with equal edge lengths -/
structure QuadrangularPyramid where
  edge_length : ℝ
  sum_of_edges : ℝ
  edge_sum_eq : sum_of_edges = 8 * edge_length

/-- Theorem: In a quadrangular pyramid with equal edge lengths, 
    if the sum of edge lengths is 14.8 meters, then each edge is 1.85 meters long -/
theorem quadrangular_pyramid_edge_length 
  (pyramid : QuadrangularPyramid) 
  (h : pyramid.sum_of_edges = 14.8) : 
  pyramid.edge_length = 1.85 := by
  sorry

#check quadrangular_pyramid_edge_length

end quadrangular_pyramid_edge_length_l2389_238936


namespace geometric_sequence_ratio_sum_l2389_238962

theorem geometric_sequence_ratio_sum (k p r : ℝ) (h1 : p ≠ 1) (h2 : r ≠ 1) (h3 : p ≠ r) 
  (h4 : k ≠ 0) (h5 : k * p^3 - k * r^3 = 3 * (k * p - k * r)) :
  p + r = Real.sqrt 3 ∨ p + r = -Real.sqrt 3 := by
  sorry

end geometric_sequence_ratio_sum_l2389_238962


namespace max_black_cells_l2389_238947

/-- Represents a board with black and white cells -/
def Board (n : ℕ) := Fin (2*n+1) → Fin (2*n+1) → Bool

/-- Checks if a 2x2 sub-board has at most 2 black cells -/
def ValidSubBoard (b : Board n) (i j : Fin (2*n)) : Prop :=
  (b i j).toNat + (b i (j+1)).toNat + (b (i+1) j).toNat + (b (i+1) (j+1)).toNat ≤ 2

/-- A board is valid if all its 2x2 sub-boards have at most 2 black cells -/
def ValidBoard (b : Board n) : Prop :=
  ∀ i j, ValidSubBoard b i j

/-- Counts the number of black cells in a board -/
def CountBlackCells (b : Board n) : ℕ :=
  (Finset.univ.sum fun i => (Finset.univ.sum fun j => (b i j).toNat))

/-- The maximum number of black cells in a valid (2n+1) × (2n+1) board is (2n+1)(n+1) -/
theorem max_black_cells (n : ℕ) :
  (∃ b : Board n, ValidBoard b ∧ CountBlackCells b = (2*n+1)*(n+1)) ∧
  (∀ b : Board n, ValidBoard b → CountBlackCells b ≤ (2*n+1)*(n+1)) := by
  sorry

end max_black_cells_l2389_238947


namespace squirrel_acorns_l2389_238985

theorem squirrel_acorns (total_acorns : ℕ) (winter_months : ℕ) (spring_acorns : ℕ) 
  (h1 : total_acorns = 210)
  (h2 : winter_months = 3)
  (h3 : spring_acorns = 30) :
  (total_acorns / winter_months) - (spring_acorns / winter_months) = 60 := by
  sorry

end squirrel_acorns_l2389_238985


namespace harry_lost_nineteen_pencils_l2389_238960

/-- The number of pencils Anna has -/
def anna_pencils : ℕ := 50

/-- The number of pencils Harry initially had -/
def harry_initial_pencils : ℕ := 2 * anna_pencils

/-- The number of pencils Harry has left -/
def harry_remaining_pencils : ℕ := 81

/-- The number of pencils Harry lost -/
def harry_lost_pencils : ℕ := harry_initial_pencils - harry_remaining_pencils

theorem harry_lost_nineteen_pencils : harry_lost_pencils = 19 := by
  sorry

end harry_lost_nineteen_pencils_l2389_238960


namespace wheel_probability_l2389_238905

theorem wheel_probability :
  let total_ratio : ℕ := 6 + 2 + 1 + 4
  let red_ratio : ℕ := 6
  let blue_ratio : ℕ := 1
  let target_ratio : ℕ := red_ratio + blue_ratio
  (target_ratio : ℚ) / total_ratio = 7 / 13 :=
by sorry

end wheel_probability_l2389_238905


namespace semicircle_in_rectangle_radius_l2389_238927

/-- Given a rectangle with a semi-circle inscribed, prove that the radius is 27 cm -/
theorem semicircle_in_rectangle_radius (L W r : ℝ) : 
  L > 0 → W > 0 → r > 0 →
  2 * L + 2 * W = 216 → -- Perimeter of rectangle is 216 cm
  W = 2 * r → -- Width is twice the radius
  L = 2 * r → -- Length is diameter (twice the radius)
  r = 27 := by
sorry

end semicircle_in_rectangle_radius_l2389_238927


namespace a_range_l2389_238921

theorem a_range (a : ℝ) (h : a^(3/2) < a^(Real.sqrt 2)) : 0 < a ∧ a < 1 := by
  sorry

end a_range_l2389_238921


namespace cement_price_per_bag_l2389_238998

theorem cement_price_per_bag 
  (cement_bags : ℕ) 
  (sand_lorries : ℕ) 
  (sand_tons_per_lorry : ℕ) 
  (sand_price_per_ton : ℕ) 
  (total_payment : ℕ) 
  (h1 : cement_bags = 500)
  (h2 : sand_lorries = 20)
  (h3 : sand_tons_per_lorry = 10)
  (h4 : sand_price_per_ton = 40)
  (h5 : total_payment = 13000) :
  (total_payment - sand_lorries * sand_tons_per_lorry * sand_price_per_ton) / cement_bags = 10 :=
by sorry

end cement_price_per_bag_l2389_238998


namespace product_magnitude_l2389_238934

theorem product_magnitude (z₁ z₂ : ℂ) (h1 : Complex.abs z₁ = 3) (h2 : z₂ = Complex.mk 2 1) :
  Complex.abs (z₁ * z₂) = 3 * Real.sqrt 5 := by
  sorry

end product_magnitude_l2389_238934


namespace lottery_probabilities_l2389_238937

/-- Represents the probability of winning a single lottery event -/
def p : ℝ := 0.05

/-- The probability of winning both lotteries -/
def win_both : ℝ := p * p

/-- The probability of winning exactly one lottery -/
def win_one : ℝ := p * (1 - p) + (1 - p) * p

/-- The probability of winning at least one lottery -/
def win_at_least_one : ℝ := win_both + win_one

theorem lottery_probabilities :
  win_both = 0.0025 ∧ win_one = 0.095 ∧ win_at_least_one = 0.0975 := by
  sorry


end lottery_probabilities_l2389_238937


namespace box_volume_increase_l2389_238976

/-- Theorem about the volume of a rectangular box after increasing dimensions --/
theorem box_volume_increase (l w h : ℝ) 
  (volume : l * w * h = 4500)
  (surface_area : 2 * (l * w + l * h + w * h) = 1800)
  (edge_sum : 4 * (l + w + h) = 216) :
  (l + 1) * (w + 1) * (h + 1) = 5455 := by
  sorry

end box_volume_increase_l2389_238976


namespace trigonometric_simplification_l2389_238920

theorem trigonometric_simplification :
  let sin15 := Real.sin (15 * π / 180)
  let sin30 := Real.sin (30 * π / 180)
  let sin45 := Real.sin (45 * π / 180)
  let sin60 := Real.sin (60 * π / 180)
  let sin75 := Real.sin (75 * π / 180)
  let cos15 := Real.cos (15 * π / 180)
  let cos30 := Real.cos (30 * π / 180)
  (sin15 + sin30 + sin45 + sin60 + sin75) / (sin15 * cos15 * cos30) = 4 * Real.sqrt 3 := by
  sorry

end trigonometric_simplification_l2389_238920


namespace admission_cutoff_score_admission_cutoff_score_is_96_l2389_238941

theorem admission_cutoff_score (total_average : ℝ) (admitted_fraction : ℝ) 
  (admitted_score_diff : ℝ) (non_admitted_score_diff : ℝ) : ℝ :=
  let cutoff := total_average + (admitted_fraction * admitted_score_diff - 
    (1 - admitted_fraction) * non_admitted_score_diff)
  cutoff

theorem admission_cutoff_score_is_96 :
  admission_cutoff_score 90 (2/5) 15 20 = 96 := by
  sorry

end admission_cutoff_score_admission_cutoff_score_is_96_l2389_238941


namespace emma_last_page_l2389_238914

/-- Represents a reader with their reading speed in seconds per page -/
structure Reader where
  name : String
  speed : ℕ

/-- Represents the novel reading scenario -/
structure NovelReading where
  totalPages : ℕ
  emma : Reader
  liam : Reader
  noah : Reader
  noahPages : ℕ

/-- Calculates the last page Emma should read -/
def lastPageForEmma (scenario : NovelReading) : ℕ :=
  sorry

/-- Theorem stating that the last page Emma should read is 525 -/
theorem emma_last_page (scenario : NovelReading) 
  (h1 : scenario.totalPages = 900)
  (h2 : scenario.emma = ⟨"Emma", 15⟩)
  (h3 : scenario.liam = ⟨"Liam", 45⟩)
  (h4 : scenario.noah = ⟨"Noah", 30⟩)
  (h5 : scenario.noahPages = 200)
  : lastPageForEmma scenario = 525 := by
  sorry

end emma_last_page_l2389_238914


namespace system_solution_l2389_238968

theorem system_solution (x y : ℝ) : x = 1 ∧ y = -2 → x + y = -1 ∧ x - y = 3 := by
  sorry

end system_solution_l2389_238968


namespace shopping_cost_calculation_l2389_238954

/-- Calculates the total cost of a shopping trip and determines the additional amount needed --/
theorem shopping_cost_calculation (shirts_count sunglasses_count skirts_count sandals_count hats_count bags_count earrings_count : ℕ)
  (shirt_price sunglasses_price skirt_price sandal_price hat_price bag_price earring_price : ℚ)
  (discount_rate tax_rate : ℚ) (payment : ℚ) :
  let subtotal := shirts_count * shirt_price + sunglasses_count * sunglasses_price + 
                  skirts_count * skirt_price + sandals_count * sandal_price + 
                  hats_count * hat_price + bags_count * bag_price + 
                  earrings_count * earring_price
  let discounted_total := subtotal * (1 - discount_rate)
  let final_total := discounted_total * (1 + tax_rate)
  let change_needed := final_total - payment
  shirts_count = 10 ∧ sunglasses_count = 2 ∧ skirts_count = 4 ∧
  sandals_count = 3 ∧ hats_count = 5 ∧ bags_count = 7 ∧ earrings_count = 6 ∧
  shirt_price = 5 ∧ sunglasses_price = 12 ∧ skirt_price = 18 ∧
  sandal_price = 3 ∧ hat_price = 8 ∧ bag_price = 14 ∧ earring_price = 6 ∧
  discount_rate = 1/10 ∧ tax_rate = 13/200 ∧ payment = 300 →
  change_needed = 307/20 := by
  sorry

end shopping_cost_calculation_l2389_238954


namespace andrew_kept_stickers_l2389_238931

def total_stickers : ℕ := 2000

def daniel_stickers : ℕ := (total_stickers * 5) / 100

def fred_stickers : ℕ := daniel_stickers + 120

def emily_stickers : ℕ := ((daniel_stickers + fred_stickers) * 50) / 100

def gina_stickers : ℕ := 80

def hannah_stickers : ℕ := ((emily_stickers + gina_stickers) * 20) / 100

def total_given_away : ℕ := daniel_stickers + fred_stickers + emily_stickers + gina_stickers + hannah_stickers

theorem andrew_kept_stickers : total_stickers - total_given_away = 1392 := by
  sorry

end andrew_kept_stickers_l2389_238931


namespace quadratic_roots_uniqueness_l2389_238915

/-- Given two quadratic polynomials with specific root relationships, 
    prove that there is only one set of values for the roots and coefficients. -/
theorem quadratic_roots_uniqueness (p q u v : ℝ) : 
  p ≠ 0 ∧ q ≠ 0 ∧ u ≠ 0 ∧ v ≠ 0 ∧  -- non-zero roots
  p ≠ q ∧ u ≠ v ∧  -- distinct roots
  (∀ x, x^2 + u*x - v = (x - p)*(x - q)) ∧  -- first polynomial
  (∀ x, x^2 + p*x - q = (x - u)*(x - v)) →  -- second polynomial
  p = -1 ∧ q = 2 ∧ u = -1 ∧ v = 2 := by
sorry

end quadratic_roots_uniqueness_l2389_238915


namespace lucy_fish_total_l2389_238930

theorem lucy_fish_total (initial : ℕ) (additional : ℕ) (total : ℕ) : 
  initial = 212 → additional = 68 → total = initial + additional → total = 280 := by
sorry

end lucy_fish_total_l2389_238930


namespace marathon_remainder_l2389_238913

/-- Represents the length of a marathon in miles and yards -/
structure Marathon where
  miles : ℕ
  yards : ℕ

/-- Represents a distance in miles and yards -/
structure Distance where
  miles : ℕ
  yards : ℕ

def marathon : Marathon :=
  { miles := 26, yards := 385 }

def yardsPerMile : ℕ := 1760

def numMarathons : ℕ := 15

theorem marathon_remainder (m : ℕ) (y : ℕ) 
  (h : Distance.mk m y = 
    { miles := numMarathons * marathon.miles + (numMarathons * marathon.yards) / yardsPerMile,
      yards := (numMarathons * marathon.yards) % yardsPerMile }) :
  y = 495 := by
  sorry

end marathon_remainder_l2389_238913


namespace smallest_three_digit_multiple_of_13_l2389_238900

theorem smallest_three_digit_multiple_of_13 : 
  ∀ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 13 ∣ n → n ≥ 104 :=
by
  sorry

end smallest_three_digit_multiple_of_13_l2389_238900


namespace subcommittee_count_l2389_238946

def committee_size : ℕ := 7
def subcommittee_size : ℕ := 3

theorem subcommittee_count : 
  Nat.choose committee_size subcommittee_size = 35 := by
  sorry

end subcommittee_count_l2389_238946


namespace chemical_solution_mixing_l2389_238952

theorem chemical_solution_mixing (initial_concentration : ℝ) 
  (replacement_concentration : ℝ) (replaced_portion : ℝ) 
  (resulting_concentration : ℝ) : 
  initial_concentration = 0.85 →
  replacement_concentration = 0.20 →
  replaced_portion = 0.6923076923076923 →
  resulting_concentration = 
    (initial_concentration * (1 - replaced_portion) + 
     replacement_concentration * replaced_portion) →
  resulting_concentration = 0.40 := by
sorry

end chemical_solution_mixing_l2389_238952


namespace smallest_four_digit_multiple_of_18_l2389_238979

theorem smallest_four_digit_multiple_of_18 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 18 ∣ n → 1008 ≤ n :=
by sorry

end smallest_four_digit_multiple_of_18_l2389_238979


namespace not_both_divisible_by_seven_l2389_238949

theorem not_both_divisible_by_seven (a b : ℝ) : 
  (¬ ∃ k : ℤ, a * b = 7 * k) → (¬ ∃ m : ℤ, a = 7 * m) ∧ (¬ ∃ n : ℤ, b = 7 * n) := by
  sorry

end not_both_divisible_by_seven_l2389_238949


namespace orange_bin_count_l2389_238961

/-- Given an initial quantity of oranges, a number of oranges removed, and a number of oranges added,
    calculate the final quantity of oranges. -/
def final_orange_count (initial : ℕ) (removed : ℕ) (added : ℕ) : ℕ :=
  initial - removed + added

/-- Theorem stating that given the specific values from the problem,
    the final orange count is 31. -/
theorem orange_bin_count : final_orange_count 5 2 28 = 31 := by
  sorry

end orange_bin_count_l2389_238961


namespace fraction_equality_expression_equality_l2389_238969

-- Problem 1
theorem fraction_equality : (2021 * 2023) / (2022^2 - 1) = 1 := by sorry

-- Problem 2
theorem expression_equality : 2 * 101^2 + 2 * 101 * 98 + 2 * 49^2 = 45000 := by sorry

end fraction_equality_expression_equality_l2389_238969


namespace coefficient_of_x4_l2389_238918

theorem coefficient_of_x4 (x : ℝ) : 
  let expression := 2*(x^2 - x^4 + 2*x^3) + 4*(x^4 - x^3 + x^2 + 2*x^5 - x^6) + 3*(2*x^3 + x^4 - 4*x^2)
  ∃ (a b c d e f : ℝ), expression = a*x^6 + b*x^5 + 5*x^4 + c*x^3 + d*x^2 + e*x + f :=
by
  sorry

end coefficient_of_x4_l2389_238918


namespace fraction_of_girls_l2389_238938

theorem fraction_of_girls (total_students : ℕ) (boys : ℕ) (h1 : total_students = 160) (h2 : boys = 60) :
  (total_students - boys : ℚ) / total_students = 5 / 8 := by
  sorry

#check fraction_of_girls

end fraction_of_girls_l2389_238938


namespace correct_num_clowns_l2389_238912

/-- The number of clowns attending a carousel --/
def num_clowns : ℕ := 4

/-- The number of children attending the carousel --/
def num_children : ℕ := 30

/-- The total number of candies initially --/
def total_candies : ℕ := 700

/-- The number of candies given to each person --/
def candies_per_person : ℕ := 20

/-- The number of candies left after distribution --/
def candies_left : ℕ := 20

/-- Theorem stating that the number of clowns is correct given the conditions --/
theorem correct_num_clowns :
  num_clowns * candies_per_person + num_children * candies_per_person + candies_left = total_candies :=
by sorry

end correct_num_clowns_l2389_238912


namespace no_a_in_either_subject_l2389_238940

theorem no_a_in_either_subject (total_students : ℕ) (physics_a : ℕ) (chemistry_a : ℕ) (both_a : ℕ)
  (h1 : total_students = 40)
  (h2 : physics_a = 10)
  (h3 : chemistry_a = 18)
  (h4 : both_a = 6) :
  total_students - (physics_a + chemistry_a - both_a) = 18 :=
by sorry

end no_a_in_either_subject_l2389_238940


namespace parabola_points_relationship_l2389_238925

/-- Proves that for points A(2, y₁), B(3, y₂), and C(-1, y₃) lying on the parabola 
    y = ax² - 4ax + c where a > 0, the relationship y₁ < y₂ < y₃ holds. -/
theorem parabola_points_relationship (a c y₁ y₂ y₃ : ℝ) 
  (ha : a > 0)
  (h1 : y₁ = a * 2^2 - 4 * a * 2 + c)
  (h2 : y₂ = a * 3^2 - 4 * a * 3 + c)
  (h3 : y₃ = a * (-1)^2 - 4 * a * (-1) + c) :
  y₁ < y₂ ∧ y₂ < y₃ := by
  sorry

#check parabola_points_relationship

end parabola_points_relationship_l2389_238925


namespace max_value_of_expression_l2389_238909

theorem max_value_of_expression (x y z : ℝ) (h : x + y + 2*z = 5) :
  ∃ (max : ℝ), max = 25/6 ∧ ∀ (a b c : ℝ), a + b + 2*c = 5 → a*b + a*c + b*c ≤ max :=
by sorry

end max_value_of_expression_l2389_238909


namespace charity_event_revenue_l2389_238911

theorem charity_event_revenue (total_tickets : Nat) (total_revenue : Nat) 
  (full_price_tickets : Nat) (discount_tickets : Nat) (full_price : Nat) :
  total_tickets = 190 →
  total_revenue = 2871 →
  full_price_tickets + discount_tickets = total_tickets →
  full_price_tickets * full_price + discount_tickets * (full_price / 3) = total_revenue →
  full_price_tickets * full_price = 1900 :=
by sorry

end charity_event_revenue_l2389_238911


namespace committee_formation_theorem_l2389_238959

/-- The number of ways to form a committee with leaders --/
def committee_formation_ways (n m k : ℕ) : ℕ :=
  (Nat.choose n m) * (2^m - 2)

/-- Theorem stating the number of ways to form the committee --/
theorem committee_formation_theorem :
  committee_formation_ways 10 5 4 = 7560 := by
  sorry

end committee_formation_theorem_l2389_238959


namespace rectangle_area_l2389_238971

theorem rectangle_area (y : ℝ) (h : y > 0) :
  ∃ (w l : ℝ), w > 0 ∧ l > 0 ∧ l = 3 * w ∧ y^2 = l^2 + w^2 ∧ w * l = (3 * y^2) / 10 := by
  sorry

end rectangle_area_l2389_238971


namespace work_scaling_l2389_238986

/-- Given that 3 people can do 3 times of a particular work in 3 days,
    prove that 6 people can do 6 times of that work in the same number of days. -/
theorem work_scaling (work : ℕ → ℕ → ℕ → Prop) : 
  work 3 3 3 → work 6 6 3 :=
by sorry

end work_scaling_l2389_238986


namespace quadratic_equation_result_l2389_238951

theorem quadratic_equation_result (x : ℝ) (h : x^2 - 2*x + 1 = 0) : 2*x^2 - 4*x = -2 := by
  sorry

end quadratic_equation_result_l2389_238951


namespace net_cash_change_l2389_238904

/-- Represents the financial state of a person -/
structure FinancialState where
  cash : Int
  ownsHouse : Bool

/-- Represents a house transaction -/
inductive Transaction
  | Rent : Transaction
  | BuyHouse : Int → Transaction
  | SellHouse : Int → Transaction

def initialValueA : Int := 15000
def initialValueB : Int := 20000
def initialHouseValue : Int := 15000
def rentAmount : Int := 2000

def applyTransaction (state : FinancialState) (transaction : Transaction) : FinancialState :=
  match transaction with
  | Transaction.Rent => 
      if state.ownsHouse then 
        { cash := state.cash + rentAmount, ownsHouse := state.ownsHouse }
      else 
        { cash := state.cash - rentAmount, ownsHouse := state.ownsHouse }
  | Transaction.BuyHouse price => 
      { cash := state.cash - price, ownsHouse := true }
  | Transaction.SellHouse price => 
      { cash := state.cash + price, ownsHouse := false }

def transactions : List Transaction := [
  Transaction.Rent,
  Transaction.SellHouse 18000,
  Transaction.BuyHouse 17000
]

theorem net_cash_change 
  (initialA : FinancialState) 
  (initialB : FinancialState) 
  (finalA : FinancialState) 
  (finalB : FinancialState) :
  initialA = { cash := initialValueA, ownsHouse := true } →
  initialB = { cash := initialValueB, ownsHouse := false } →
  finalA = transactions.foldl applyTransaction initialA →
  finalB = transactions.foldl applyTransaction initialB →
  finalA.cash - initialA.cash = 3000 ∧ 
  finalB.cash - initialB.cash = -3000 :=
sorry

end net_cash_change_l2389_238904


namespace cow_husk_consumption_l2389_238901

theorem cow_husk_consumption 
  (cows bags days : ℕ) 
  (h : cows = 45 ∧ bags = 45 ∧ days = 45) : 
  (1 : ℕ) * days = 45 := by
  sorry

end cow_husk_consumption_l2389_238901


namespace chess_tournament_players_l2389_238932

theorem chess_tournament_players (total_games : ℕ) (h1 : total_games = 30) : ∃ n : ℕ, n > 0 ∧ total_games = n * (n - 1) := by
  sorry

end chess_tournament_players_l2389_238932


namespace no_solution_equation_l2389_238923

theorem no_solution_equation : ¬∃ (x : ℝ), x - 7 / (x - 3) = 3 - 7 / (x - 3) := by
  sorry

end no_solution_equation_l2389_238923


namespace fifth_term_of_sequence_l2389_238965

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

theorem fifth_term_of_sequence (a₁ d : ℤ) :
  arithmetic_sequence a₁ d 20 = 12 →
  arithmetic_sequence a₁ d 21 = 16 →
  arithmetic_sequence a₁ d 5 = -48 := by
sorry

end fifth_term_of_sequence_l2389_238965


namespace exact_fourth_power_implies_zero_coefficients_exact_square_implies_perfect_square_l2389_238917

-- Part (a)
theorem exact_fourth_power_implies_zero_coefficients 
  (a b c : ℤ) 
  (h : ∀ x : ℤ, ∃ y : ℤ, a * x^2 + b * x + c = y^4) :
  a = 0 ∧ b = 0 := by sorry

-- Part (b)
theorem exact_square_implies_perfect_square 
  (a b c : ℤ) 
  (h : ∀ x : ℤ, ∃ z : ℤ, a * x^2 + b * x + c = z^2) :
  ∃ d e : ℤ, ∀ x : ℤ, a * x^2 + b * x + c = (d * x + e)^2 := by sorry

end exact_fourth_power_implies_zero_coefficients_exact_square_implies_perfect_square_l2389_238917


namespace sara_quarters_l2389_238907

def cents : ℕ := 275
def cents_per_quarter : ℕ := 25

theorem sara_quarters : cents / cents_per_quarter = 11 := by
  sorry

end sara_quarters_l2389_238907


namespace max_value_xyz_l2389_238956

theorem max_value_xyz (x y z : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : 3 * x + 2 * y + 6 * z = 1) :
  x^4 * y^3 * z^2 ≤ 1 / 372008 :=
by sorry

end max_value_xyz_l2389_238956


namespace inequality_proof_l2389_238906

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 - b*c) / (2*a^2 + b*c) + (b^2 - c*a) / (2*b^2 + c*a) + (c^2 - a*b) / (2*c^2 + a*b) ≤ 0 :=
by sorry

end inequality_proof_l2389_238906


namespace max_value_of_f_l2389_238910

def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 4*x + a

theorem max_value_of_f (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f a x ≥ -2) ∧
  (∃ x ∈ Set.Icc 0 1, f a x = -2) →
  (∃ x ∈ Set.Icc 0 1, f a x = 1) ∧
  (∀ x ∈ Set.Icc 0 1, f a x ≤ 1) :=
by sorry

end max_value_of_f_l2389_238910


namespace reflection_across_y_axis_l2389_238955

-- Define a function f from real numbers to real numbers
variable (f : ℝ → ℝ)

-- Define a point (x, y) on the graph of y = f(x)
variable (x y : ℝ)

-- Define the reflection transformation across the y-axis
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

-- State the theorem
theorem reflection_across_y_axis :
  y = f x ↔ y = f (-(-x)) :=
sorry

end reflection_across_y_axis_l2389_238955


namespace derek_savings_l2389_238953

theorem derek_savings (P : ℚ) : P * 2^11 = 4096 → P = 2 := by
  sorry

end derek_savings_l2389_238953


namespace town_population_increase_l2389_238903

/-- Calculates the average percent increase of population per year given initial and final populations over a decade. -/
def average_percent_increase (initial_population final_population : ℕ) : ℚ :=
  let total_increase : ℕ := final_population - initial_population
  let average_annual_increase : ℚ := total_increase / 10
  (average_annual_increase / initial_population) * 100

/-- Theorem stating that the average percent increase of population per year is 7% for the given conditions. -/
theorem town_population_increase :
  average_percent_increase 175000 297500 = 7 := by
  sorry

#eval average_percent_increase 175000 297500

end town_population_increase_l2389_238903


namespace evaluate_expression_l2389_238967

theorem evaluate_expression : (8^3 / 8^2) * 2^10 = 8192 := by
  sorry

end evaluate_expression_l2389_238967


namespace parallel_vectors_imply_x_equals_one_l2389_238983

-- Define the points
def A : ℝ × ℝ := (0, -3)
def B : ℝ × ℝ := (3, 3)
def C : ℝ → ℝ × ℝ := λ x ↦ (x, -1)

-- Define the vectors
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def AC (x : ℝ) : ℝ × ℝ := ((C x).1 - A.1, (C x).2 - A.2)

-- Theorem statement
theorem parallel_vectors_imply_x_equals_one :
  ∀ x : ℝ, (∃ k : ℝ, AB = k • (AC x)) → x = 1 := by
  sorry

end parallel_vectors_imply_x_equals_one_l2389_238983


namespace area_of_AGKIJEFB_l2389_238963

-- Define the hexagons and point K
structure Hexagon where
  vertices : Fin 6 → ℝ × ℝ
  area : ℝ

def Point := ℝ × ℝ

-- Define the problem setup
axiom hexagon1 : Hexagon
axiom hexagon2 : Hexagon
axiom K : Point

-- State the conditions
axiom shared_side : hexagon1.vertices 4 = hexagon2.vertices 4 ∧ hexagon1.vertices 5 = hexagon2.vertices 5
axiom equal_areas : hexagon1.area = 36 ∧ hexagon2.area = 36
axiom K_on_AB : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ K = (1 - t) • hexagon1.vertices 0 + t • hexagon1.vertices 1
axiom AK_KB_ratio : ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a / b = 1 / 2 ∧
  K = (b / (a + b)) • hexagon1.vertices 0 + (a / (a + b)) • hexagon1.vertices 1
axiom K_midpoint_GH : K = (1 / 2) • hexagon2.vertices 0 + (1 / 2) • hexagon2.vertices 1

-- Define the polygon AGKIJEFB
def polygon_AGKIJEFB_area : ℝ := sorry

-- State the theorem to be proved
theorem area_of_AGKIJEFB : polygon_AGKIJEFB_area = 36 + Real.sqrt 6 := by sorry

end area_of_AGKIJEFB_l2389_238963


namespace congruence_solution_l2389_238948

theorem congruence_solution (n : ℕ) : n ∈ Finset.range 47 ∧ 13 * n ≡ 9 [MOD 47] ↔ n = 20 := by
  sorry

end congruence_solution_l2389_238948


namespace sock_order_ratio_l2389_238973

/-- Represents the number of pairs of socks --/
structure SockOrder where
  green : ℕ
  red : ℕ

/-- Represents the price of socks --/
structure SockPrice where
  red : ℝ
  green : ℝ

/-- Calculates the total cost of a sock order --/
def totalCost (order : SockOrder) (price : SockPrice) : ℝ :=
  order.green * price.green + order.red * price.red

theorem sock_order_ratio (original : SockOrder) (price : SockPrice) :
  original.green = 6 →
  price.green = 3 * price.red →
  let interchanged : SockOrder := ⟨original.red, original.green⟩
  totalCost interchanged price = 1.2 * totalCost original price →
  2 * original.red = 3 * original.green := by
  sorry

end sock_order_ratio_l2389_238973


namespace min_value_of_f_l2389_238942

theorem min_value_of_f (x : ℝ) (h : x > 0) : 
  let f := fun x => 1 / x^2 + 2 * x
  (∀ y > 0, f y ≥ 3) ∧ (∃ z > 0, f z = 3) := by
  sorry

end min_value_of_f_l2389_238942


namespace three_digit_divisible_by_nine_l2389_238939

theorem three_digit_divisible_by_nine :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 
  n % 10 = 3 ∧ 
  (n / 100) % 10 = 5 ∧ 
  n % 9 = 0 ∧
  n = 513 :=
sorry

end three_digit_divisible_by_nine_l2389_238939


namespace ellipse_other_intersection_l2389_238980

/-- Define an ellipse with foci at (0,0) and (4,0) that intersects the x-axis at (1,0) -/
def ellipse (x : ℝ) : Prop :=
  (|x| + |x - 4|) = 4

/-- The other point of intersection of the ellipse with the x-axis -/
def other_intersection : ℝ := 4

/-- Theorem stating that the other point of intersection is (4,0) -/
theorem ellipse_other_intersection :
  ellipse other_intersection ∧ other_intersection ≠ 1 :=
sorry

end ellipse_other_intersection_l2389_238980


namespace least_number_with_divisibility_property_l2389_238975

def is_divisible (n m : ℕ) : Prop := m ≠ 0 ∧ n % m = 0

def is_least_with_property (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a + 1 = b ∧ b ≤ 20 ∧
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 20 ∧ k ≠ a ∧ k ≠ b → is_divisible n k) ∧
  (∀ m : ℕ, m < n → ¬∃ (c d : ℕ), c + 1 = d ∧ d ≤ 20 ∧
    (∀ k : ℕ, 1 ≤ k ∧ k ≤ 20 ∧ k ≠ c ∧ k ≠ d → is_divisible m k))

theorem least_number_with_divisibility_property :
  is_least_with_property 12252240 := by sorry

end least_number_with_divisibility_property_l2389_238975


namespace simplify_and_evaluate_l2389_238943

theorem simplify_and_evaluate (x : ℝ) (h : x ≠ 2) :
  (x + 2 + 3 / (x - 2)) / ((1 + 2*x + x^2) / (x - 2)) = (x - 1) / (x + 1) ∧
  (4 + 2 + 3 / (4 - 2)) / ((1 + 2*4 + 4^2) / (4 - 2)) = 3 / 5 :=
by sorry

end simplify_and_evaluate_l2389_238943


namespace difference_is_10q_minus_10_l2389_238974

/-- The difference in dimes between two people's money, given their quarter amounts -/
def difference_in_dimes (charles_quarters richard_quarters : ℤ) : ℚ :=
  2.5 * (charles_quarters - richard_quarters)

/-- Proof that the difference in dimes between Charles and Richard's money is 10(q - 1) -/
theorem difference_is_10q_minus_10 (q : ℤ) :
  difference_in_dimes (5 * q + 1) (q + 5) = 10 * (q - 1) := by
  sorry

#check difference_is_10q_minus_10

end difference_is_10q_minus_10_l2389_238974


namespace intersection_complement_equality_l2389_238916

def U : Finset ℕ := {0, 1, 2, 3, 4}
def M : Finset ℕ := {0, 1, 2}
def N : Finset ℕ := {2, 3}

theorem intersection_complement_equality :
  M ∩ (U \ N) = {0, 1} := by sorry

end intersection_complement_equality_l2389_238916


namespace parallel_line_y_intercept_l2389_238919

/-- A line parallel to y = -3x - 6 passing through (3, -1) has y-intercept 8 -/
theorem parallel_line_y_intercept :
  ∀ (b : ℝ → ℝ),
  (∀ x y, b x = y ↔ ∃ k, y = -3 * x + k) →  -- b is parallel to y = -3x - 6
  b 3 = -1 →                               -- b passes through (3, -1)
  ∃ k, b 0 = k ∧ k = 8 :=                  -- y-intercept of b is 8
by sorry

end parallel_line_y_intercept_l2389_238919


namespace y_intercept_for_specific_line_l2389_238988

/-- A line in a 2D plane. -/
structure Line where
  slope : ℝ
  x_intercept : ℝ

/-- The y-intercept of a line. -/
def y_intercept (l : Line) : ℝ × ℝ :=
  (0, l.slope * (-l.x_intercept) + 0)

/-- Theorem: For a line with slope -3 and x-intercept (7,0), the y-intercept is (0,21). -/
theorem y_intercept_for_specific_line :
  let l := Line.mk (-3) 7
  y_intercept l = (0, 21) := by
  sorry

end y_intercept_for_specific_line_l2389_238988


namespace greatest_three_digit_multiple_of_17_l2389_238995

theorem greatest_three_digit_multiple_of_17 :
  ∀ n : ℕ, n < 1000 → n % 17 = 0 → n ≤ 986 :=
by sorry

end greatest_three_digit_multiple_of_17_l2389_238995


namespace v2_equals_22_at_neg_4_l2389_238929

/-- Horner's Rule for a specific polynomial -/
def horner_polynomial (x : ℝ) : ℝ := 
  ((((x * x + 6) * x + 9) * x + 0) * x + 0) * x + 208

/-- v2 calculation in Horner's Rule -/
def v2 (x : ℝ) : ℝ := 
  (1 * x * x) + 6

/-- Theorem: v2 equals 22 when x = -4 for the given polynomial -/
theorem v2_equals_22_at_neg_4 : 
  v2 (-4) = 22 := by sorry

end v2_equals_22_at_neg_4_l2389_238929


namespace diagonals_of_120_degree_polygon_l2389_238926

/-- The number of diagonals in a regular polygon with 120° interior angles is 9. -/
theorem diagonals_of_120_degree_polygon : ∃ (n : ℕ), 
  (∀ (i : ℕ), i < n → (180 * (n - 2) : ℝ) / n = 120) → 
  (n * (n - 3) : ℝ) / 2 = 9 := by
  sorry

end diagonals_of_120_degree_polygon_l2389_238926
