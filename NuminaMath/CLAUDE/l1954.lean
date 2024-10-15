import Mathlib

namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l1954_195450

/-- Two lines are parallel if and only if their slopes are equal. -/
def parallel_lines (m1 a1 b1 m2 a2 b2 : ℝ) : Prop :=
  m1 * a2 = m2 * a1

/-- Given that the line 2x + ay + 1 = 0 is parallel to x - 4y - 1 = 0, prove that a = -8 -/
theorem parallel_lines_a_value (a : ℝ) :
  parallel_lines 2 a 1 1 (-4) (-1) → a = -8 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l1954_195450


namespace NUMINAMATH_CALUDE_arithmetic_and_geometric_is_constant_l1954_195493

def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

def is_geometric_progression (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a 1 * q^n

def is_constant_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n = a 1

theorem arithmetic_and_geometric_is_constant (a : ℕ → ℝ) :
  is_arithmetic_progression a → is_geometric_progression a → is_constant_sequence a :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_and_geometric_is_constant_l1954_195493


namespace NUMINAMATH_CALUDE_weight_of_raisins_l1954_195463

/-- Given the total weight of snacks and the weight of peanuts, 
    prove that the weight of raisins is 0.4 pounds. -/
theorem weight_of_raisins (total_weight peanuts_weight : ℝ) 
  (h1 : total_weight = 0.5)
  (h2 : peanuts_weight = 0.1) : 
  total_weight - peanuts_weight = 0.4 := by
sorry

end NUMINAMATH_CALUDE_weight_of_raisins_l1954_195463


namespace NUMINAMATH_CALUDE_expression_evaluation_l1954_195458

theorem expression_evaluation :
  3 + Real.sqrt 3 + (1 / (3 + Real.sqrt 3)) + (1 / (Real.sqrt 3 - 3)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1954_195458


namespace NUMINAMATH_CALUDE_range_of_f_l1954_195404

-- Define the function f(x) = |x| - 4
def f (x : ℝ) : ℝ := |x| - 4

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = {y : ℝ | y ≥ -4} :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l1954_195404


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l1954_195453

theorem imaginary_part_of_complex_division (i : ℂ) : 
  i * i = -1 → Complex.im ((4 - 3 * i) / i) = -4 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l1954_195453


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1954_195485

/-- Two vectors are parallel if and only if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (4, x)
  parallel a b → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1954_195485


namespace NUMINAMATH_CALUDE_race_outcomes_count_l1954_195431

/-- Represents the number of participants in the race -/
def num_participants : ℕ := 6

/-- Represents the number of top positions we're considering -/
def num_top_positions : ℕ := 4

/-- Calculates the number of permutations of k items chosen from n items -/
def permutations (n k : ℕ) : ℕ := 
  if k > n then 0
  else (List.range n).foldr (λ i acc => (i + 1) * acc) 1

/-- Theorem stating the number of possible race outcomes -/
theorem race_outcomes_count : 
  (permutations (num_participants - 1) (num_top_positions - 1)) * num_participants - 
  (permutations (num_participants - 1) (num_top_positions - 1)) = 300 := by
  sorry


end NUMINAMATH_CALUDE_race_outcomes_count_l1954_195431


namespace NUMINAMATH_CALUDE_toy_shipment_calculation_l1954_195420

theorem toy_shipment_calculation (displayed_percentage : ℚ) (stored_toys : ℕ) : 
  displayed_percentage = 30 / 100 →
  stored_toys = 140 →
  (1 - displayed_percentage) * 200 = stored_toys := by
  sorry

end NUMINAMATH_CALUDE_toy_shipment_calculation_l1954_195420


namespace NUMINAMATH_CALUDE_parabola_vertex_l1954_195443

/-- The vertex of the parabola y = 2(x-3)^2 + 1 is at the point (3, 1). -/
theorem parabola_vertex (x y : ℝ) : 
  y = 2 * (x - 3)^2 + 1 → (3, 1) = (x, y) := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1954_195443


namespace NUMINAMATH_CALUDE_Z_three_seven_l1954_195428

def Z (a b : ℝ) : ℝ := b + 15 * a - a^2

theorem Z_three_seven : Z 3 7 = 43 := by sorry

end NUMINAMATH_CALUDE_Z_three_seven_l1954_195428


namespace NUMINAMATH_CALUDE_certain_number_is_even_l1954_195478

theorem certain_number_is_even (z : ℕ) (h1 : z > 0) (h2 : 4 ∣ z) :
  ∀ x : ℤ, (z * (2 + x + z) + 3) % 2 = 1 ↔ Even x :=
by sorry

end NUMINAMATH_CALUDE_certain_number_is_even_l1954_195478


namespace NUMINAMATH_CALUDE_retailer_profit_percent_l1954_195421

/-- Calculates the profit percent for a retailer given the purchase price, overhead expenses, and selling price. -/
def profit_percent (purchase_price overhead_expenses selling_price : ℚ) : ℚ :=
  let cost_price := purchase_price + overhead_expenses
  let profit := selling_price - cost_price
  (profit / cost_price) * 100

/-- Theorem stating that the profit percent for the given scenario is approximately 21.46% -/
theorem retailer_profit_percent :
  let ε := 0.01
  let result := profit_percent 232 15 300
  (result > 21.46 - ε) ∧ (result < 21.46 + ε) :=
by
  sorry

end NUMINAMATH_CALUDE_retailer_profit_percent_l1954_195421


namespace NUMINAMATH_CALUDE_multiplier_problem_l1954_195492

theorem multiplier_problem (m : ℝ) : m * 5.0 - 7 = 13 → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_multiplier_problem_l1954_195492


namespace NUMINAMATH_CALUDE_base_conversion_435_to_base_3_l1954_195484

theorem base_conversion_435_to_base_3 :
  (1 * 3^5 + 2 * 3^4 + 1 * 3^3 + 0 * 3^2 + 1 * 3^1 + 1 * 3^0) = 435 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_435_to_base_3_l1954_195484


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1954_195436

/-- The eccentricity of a hyperbola with equation x^2/2 - y^2 = 1 is √6/2 -/
theorem hyperbola_eccentricity :
  let a : ℝ := Real.sqrt 2
  let b : ℝ := 1
  let c : ℝ := Real.sqrt 3
  let e : ℝ := c / a
  e = Real.sqrt 6 / 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1954_195436


namespace NUMINAMATH_CALUDE_jim_investment_approx_l1954_195426

/-- Represents the investment ratios of John, James, Jim, and Jordan respectively -/
def investment_ratio : Fin 4 → ℕ
  | 0 => 8   -- John
  | 1 => 11  -- James
  | 2 => 15  -- Jim
  | 3 => 19  -- Jordan

/-- The total investment amount in dollars -/
def total_investment : ℚ := 127000

/-- Jim's index in the investment ratio -/
def jim_index : Fin 4 := 2

/-- Calculate Jim's investment amount -/
def jim_investment : ℚ :=
  (total_investment * investment_ratio jim_index) /
  (Finset.sum Finset.univ investment_ratio)

theorem jim_investment_approx :
  ∃ ε > 0, |jim_investment - 35943.40| < ε := by sorry

end NUMINAMATH_CALUDE_jim_investment_approx_l1954_195426


namespace NUMINAMATH_CALUDE_lcm_gcd_product_l1954_195425

theorem lcm_gcd_product (a b : ℕ) (h1 : a = 24) (h2 : b = 54) :
  (Nat.lcm a b) * (Nat.gcd a b) = 1296 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_l1954_195425


namespace NUMINAMATH_CALUDE_no_real_solutions_to_inequality_l1954_195459

theorem no_real_solutions_to_inequality :
  ¬∃ x : ℝ, x ≠ 5 ∧ (x^3 - 125) / (x - 5) < 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_to_inequality_l1954_195459


namespace NUMINAMATH_CALUDE_min_moves_to_equalize_l1954_195407

/-- Represents a casket with coins -/
structure Casket :=
  (coins : ℕ)

/-- Represents the circular arrangement of caskets -/
def CasketCircle := Vector Casket 7

/-- A move transfers one coin between neighboring caskets -/
def Move := Fin 7 → Fin 7

/-- Checks if a move is valid (transfers to a neighboring casket) -/
def isValidMove (m : Move) : Prop :=
  ∀ i, m i = (i + 1) % 7 ∨ m i = (i + 6) % 7 ∨ m i = i

/-- Applies a move to a casket circle -/
def applyMove (circle : CasketCircle) (m : Move) : CasketCircle :=
  sorry

/-- Checks if all caskets have the same number of coins -/
def isEqualized (circle : CasketCircle) : Prop :=
  ∀ i j, (circle.get i).coins = (circle.get j).coins

/-- The initial arrangement of caskets -/
def initialCircle : CasketCircle :=
  Vector.ofFn (λ i => match i with
    | 0 => ⟨9⟩
    | 1 => ⟨17⟩
    | 2 => ⟨12⟩
    | 3 => ⟨5⟩
    | 4 => ⟨18⟩
    | 5 => ⟨10⟩
    | 6 => ⟨20⟩)

/-- The main theorem to be proved -/
theorem min_moves_to_equalize :
  ∃ (moves : List Move),
    moves.length = 22 ∧
    (∀ m ∈ moves, isValidMove m) ∧
    isEqualized (moves.foldl applyMove initialCircle) ∧
    (∀ (otherMoves : List Move),
      otherMoves.length < 22 →
      ¬isEqualized (otherMoves.foldl applyMove initialCircle)) :=
  sorry

end NUMINAMATH_CALUDE_min_moves_to_equalize_l1954_195407


namespace NUMINAMATH_CALUDE_interview_panel_seating_l1954_195419

/-- Represents the number of players in each team --/
structure TeamSizes :=
  (team1 : Nat)
  (team2 : Nat)
  (team3 : Nat)

/-- Calculates the number of seating arrangements for players from different teams
    where teammates must sit together --/
def seatingArrangements (sizes : TeamSizes) : Nat :=
  Nat.factorial 3 * Nat.factorial sizes.team1 * Nat.factorial sizes.team2 * Nat.factorial sizes.team3

/-- Theorem stating that for the given team sizes, there are 1728 seating arrangements --/
theorem interview_panel_seating :
  seatingArrangements ⟨4, 3, 2⟩ = 1728 := by
  sorry

#eval seatingArrangements ⟨4, 3, 2⟩

end NUMINAMATH_CALUDE_interview_panel_seating_l1954_195419


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l1954_195481

theorem complex_number_in_second_quadrant : 
  let z : ℂ := (Complex.I / (1 + Complex.I)) + (1 + Complex.I * Real.sqrt 3) ^ 2
  (z.re < 0) ∧ (z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l1954_195481


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l1954_195451

-- Define the universe of discourse
variable (U : Type)

-- Define the predicate for being a domestic mobile phone
variable (D : U → Prop)

-- Define the predicate for having trap consumption
variable (T : U → Prop)

-- State the theorem
theorem negation_of_universal_statement :
  (¬ ∀ x, D x → T x) ↔ (∃ x, D x ∧ ¬ T x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l1954_195451


namespace NUMINAMATH_CALUDE_smallest_reciprocal_l1954_195455

theorem smallest_reciprocal (a b c : ℕ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (h_order : a > b ∧ b > c) :
  (1 : ℚ) / a < (1 : ℚ) / b ∧ (1 : ℚ) / b < (1 : ℚ) / c :=
by sorry

end NUMINAMATH_CALUDE_smallest_reciprocal_l1954_195455


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1954_195456

theorem inequality_system_solution (x : ℝ) :
  (3 * (x + 2) - x > 4 ∧ (1 + 2*x) / 3 ≥ x - 1) ↔ (-1 < x ∧ x ≤ 4) := by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1954_195456


namespace NUMINAMATH_CALUDE_jeff_shelter_cats_l1954_195479

def cat_shelter_problem (initial_cats : ℕ) 
  (monday_added : ℕ) (tuesday_added : ℕ) 
  (wednesday_adopted wednesday_added : ℕ) 
  (thursday_adopted thursday_added : ℕ)
  (friday_adopted friday_added : ℕ) : Prop :=
  let after_monday := initial_cats + monday_added
  let after_tuesday := after_monday + tuesday_added
  let after_wednesday := after_tuesday + wednesday_added - wednesday_adopted
  let after_thursday := after_wednesday + thursday_added - thursday_adopted
  let final_count := after_thursday + friday_added - friday_adopted
  final_count = 30

theorem jeff_shelter_cats : 
  cat_shelter_problem 20 9 6 8 2 3 3 2 3 :=
by sorry

end NUMINAMATH_CALUDE_jeff_shelter_cats_l1954_195479


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1954_195494

-- Problem 1
theorem simplify_expression_1 (a : ℝ) : 2 * a^2 - 5 * a + a^2 + 4 * a - 3 * a^2 = -a := by
  sorry

-- Problem 2
theorem simplify_expression_2 (a b : ℝ) : 2 * (a^2 + 3 * b^3) - (1/3) * (9 * a^2 - 12 * b^3) = -a^2 + 10 * b^3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1954_195494


namespace NUMINAMATH_CALUDE_circle_alignment_exists_l1954_195448

-- Define the circle type
structure Circle where
  circumference : ℝ
  marked_points : ℕ
  arc_length : ℝ

-- Define the theorem
theorem circle_alignment_exists (c1 c2 : Circle)
  (h1 : c1.circumference = 100)
  (h2 : c2.circumference = 100)
  (h3 : c1.marked_points = 100)
  (h4 : c2.arc_length < 1) :
  ∃ (alignment : ℝ), ∀ (point : ℕ) (arc : ℝ),
    point < c1.marked_points →
    arc < c2.arc_length →
    (point : ℝ) * c1.circumference / c1.marked_points + alignment ≠ arc :=
sorry

end NUMINAMATH_CALUDE_circle_alignment_exists_l1954_195448


namespace NUMINAMATH_CALUDE_inequality_proof_l1954_195477

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h1 : a^2 < 16*b*c) (h2 : b^2 < 16*c*a) (h3 : c^2 < 16*a*b) :
  a^2 + b^2 + c^2 < 2*(a*b + b*c + c*a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1954_195477


namespace NUMINAMATH_CALUDE_perfect_square_3_4_4_6_5_6_l1954_195400

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

theorem perfect_square_3_4_4_6_5_6 :
  is_perfect_square (3^4 * 4^6 * 5^6) :=
by
  sorry

end NUMINAMATH_CALUDE_perfect_square_3_4_4_6_5_6_l1954_195400


namespace NUMINAMATH_CALUDE_geometric_series_sum_l1954_195464

/-- The sum of the infinite geometric series 4/3 - 5/12 + 25/144 - 125/1728 + ... -/
theorem geometric_series_sum : 
  let a : ℚ := 4/3
  let r : ℚ := -5/16
  let series_sum : ℚ := a / (1 - r)
  series_sum = 64/63 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l1954_195464


namespace NUMINAMATH_CALUDE_alison_lollipops_l1954_195401

theorem alison_lollipops :
  ∀ (alison henry diane : ℕ),
  henry = alison + 30 →
  alison = diane / 2 →
  alison + henry + diane = 45 * 6 →
  alison = 60 := by
sorry

end NUMINAMATH_CALUDE_alison_lollipops_l1954_195401


namespace NUMINAMATH_CALUDE_nth_monomial_form_l1954_195480

/-- A sequence of monomials is defined as follows:
    1st term: a
    2nd term: 3a²
    3rd term: 5a³
    4th term: 7a⁴
    5th term: 9a⁵
    ...
    This function represents the coefficient of the nth term in this sequence. -/
def monomial_coefficient (n : ℕ) : ℕ := 2 * n - 1

/-- This function represents the exponent of 'a' in the nth term of the sequence. -/
def monomial_exponent (n : ℕ) : ℕ := n

/-- This theorem states that the nth term of the sequence is (2n - 1)aⁿ -/
theorem nth_monomial_form (n : ℕ) (a : ℝ) :
  monomial_coefficient n * a ^ monomial_exponent n = (2 * n - 1) * a ^ n :=
sorry

end NUMINAMATH_CALUDE_nth_monomial_form_l1954_195480


namespace NUMINAMATH_CALUDE_reflection_composition_l1954_195427

theorem reflection_composition :
  let x_reflection : Matrix (Fin 2) (Fin 2) ℝ := !![1, 0; 0, -1]
  let y_reflection : Matrix (Fin 2) (Fin 2) ℝ := !![-1, 0; 0, 1]
  x_reflection * y_reflection = !![-1, 0; 0, -1] := by
sorry

end NUMINAMATH_CALUDE_reflection_composition_l1954_195427


namespace NUMINAMATH_CALUDE_marbleChoices_eq_56_l1954_195439

/-- A function that returns the number of ways to choose one marble from a set of 15 
    and two ordered marbles from a set of 8 such that the sum of the two chosen marbles 
    equals the number on the single chosen marble -/
def marbleChoices : ℕ :=
  let jessicaMarbles := Finset.range 15
  let myMarbles := Finset.range 8
  Finset.sum jessicaMarbles (λ j => 
    Finset.sum myMarbles (λ m1 => 
      Finset.sum myMarbles (λ m2 => 
        if m1 + m2 + 2 = j + 1 then 1 else 0)))

theorem marbleChoices_eq_56 : marbleChoices = 56 := by
  sorry

end NUMINAMATH_CALUDE_marbleChoices_eq_56_l1954_195439


namespace NUMINAMATH_CALUDE_prob_same_gender_is_one_third_l1954_195403

/-- Represents the gender of a student -/
inductive Gender
| Male
| Female

/-- Represents a group of students -/
structure StudentGroup where
  males : Finset Gender
  females : Finset Gender
  male_count : males.card = 2
  female_count : females.card = 2

/-- Represents a selection of two students -/
structure Selection where
  first : Gender
  second : Gender

/-- The probability of selecting two students of the same gender -/
def prob_same_gender (group : StudentGroup) : ℚ :=
  (2 : ℚ) / 6

theorem prob_same_gender_is_one_third (group : StudentGroup) :
  prob_same_gender group = (1 : ℚ) / 3 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_gender_is_one_third_l1954_195403


namespace NUMINAMATH_CALUDE_unique_mod_10_solution_l1954_195470

theorem unique_mod_10_solution : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -4229 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_unique_mod_10_solution_l1954_195470


namespace NUMINAMATH_CALUDE_P_equals_Q_l1954_195487

-- Define the sets P and Q
def P : Set ℕ := {2, 3}
def Q : Set ℕ := {3, 2}

-- Theorem stating that P and Q are equal
theorem P_equals_Q : P = Q := by sorry

end NUMINAMATH_CALUDE_P_equals_Q_l1954_195487


namespace NUMINAMATH_CALUDE_boys_in_class_l1954_195449

theorem boys_in_class (total_students : ℕ) (total_cost : ℕ) (boys_cost : ℕ) (girls_cost : ℕ)
  (h1 : total_students = 43)
  (h2 : total_cost = 1101)
  (h3 : boys_cost = 24)
  (h4 : girls_cost = 27) :
  ∃ (boys girls : ℕ),
    boys + girls = total_students ∧
    boys * boys_cost + girls * girls_cost = total_cost ∧
    boys = 20 := by
  sorry

end NUMINAMATH_CALUDE_boys_in_class_l1954_195449


namespace NUMINAMATH_CALUDE_multiply_squared_equation_l1954_195488

/-- Given that a^2 * b = 3 * (4a + 2) and a = 1 is a possible solution, prove that b = 18 -/
theorem multiply_squared_equation (a b : ℝ) : 
  a^2 * b = 3 * (4*a + 2) → a = 1 → b = 18 := by
  sorry

end NUMINAMATH_CALUDE_multiply_squared_equation_l1954_195488


namespace NUMINAMATH_CALUDE_max_value_of_y_l1954_195441

open Complex

theorem max_value_of_y (θ : Real) (h : 0 < θ ∧ θ < Real.pi / 2) :
  let z : ℂ := 3 * cos θ + 2 * I * sin θ
  let y : Real := θ - arg z
  ∃ (max_y : Real), ∀ (θ' : Real), 0 < θ' ∧ θ' < Real.pi / 2 →
    let z' : ℂ := 3 * cos θ' + 2 * I * sin θ'
    let y' : Real := θ' - arg z'
    y' ≤ max_y ∧ max_y = Real.arctan (Real.sqrt 6 / 12) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_y_l1954_195441


namespace NUMINAMATH_CALUDE_point_coordinates_l1954_195429

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is in the third quadrant -/
def in_third_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Distance from a point to the x-axis -/
def distance_to_x_axis (p : Point) : ℝ :=
  |p.y|

/-- Distance from a point to the y-axis -/
def distance_to_y_axis (p : Point) : ℝ :=
  |p.x|

theorem point_coordinates (p : Point) 
  (h1 : in_third_quadrant p)
  (h2 : distance_to_x_axis p = 8)
  (h3 : distance_to_y_axis p = 5) :
  p = Point.mk (-5) (-8) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l1954_195429


namespace NUMINAMATH_CALUDE_tomato_count_l1954_195413

/-- Represents a rectangular garden with tomatoes -/
structure TomatoGarden where
  rows : ℕ
  columns : ℕ
  tomato_position : ℕ × ℕ

/-- Calculates the total number of tomatoes in the garden -/
def total_tomatoes (garden : TomatoGarden) : ℕ :=
  garden.rows * garden.columns

/-- Theorem stating the total number of tomatoes in the garden -/
theorem tomato_count (garden : TomatoGarden) 
  (h1 : garden.tomato_position.1 = 8)  -- 8th row from front
  (h2 : garden.rows - garden.tomato_position.1 + 1 = 14)  -- 14th row from back
  (h3 : garden.tomato_position.2 = 7)  -- 7th row from left
  (h4 : garden.columns - garden.tomato_position.2 + 1 = 13)  -- 13th row from right
  : total_tomatoes garden = 399 := by
  sorry

#eval total_tomatoes { rows := 21, columns := 19, tomato_position := (8, 7) }

end NUMINAMATH_CALUDE_tomato_count_l1954_195413


namespace NUMINAMATH_CALUDE_external_tangent_intersections_collinear_l1954_195495

-- Define a circle in a plane
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point in a plane
def Point := ℝ × ℝ

-- Define a function to get the common external tangent intersection point of two circles
def commonExternalTangentIntersection (c1 c2 : Circle) : Point :=
  sorry

-- Theorem statement
theorem external_tangent_intersections_collinear (c1 c2 c3 : Circle) :
  let A := commonExternalTangentIntersection c1 c2
  let B := commonExternalTangentIntersection c2 c3
  let C := commonExternalTangentIntersection c3 c1
  ∃ (m b : ℝ), (A.1 = m * A.2 + b) ∧ (B.1 = m * B.2 + b) ∧ (C.1 = m * C.2 + b) :=
by sorry

end NUMINAMATH_CALUDE_external_tangent_intersections_collinear_l1954_195495


namespace NUMINAMATH_CALUDE_yard_area_l1954_195430

/-- Given a rectangular yard where one side is 40 feet and the sum of the other three sides is 56 feet,
    the area of the yard is 320 square feet. -/
theorem yard_area (length width : ℝ) : 
  length = 40 →
  2 * width + length = 56 →
  length * width = 320 := by
  sorry

end NUMINAMATH_CALUDE_yard_area_l1954_195430


namespace NUMINAMATH_CALUDE_career_preference_circle_graph_l1954_195482

theorem career_preference_circle_graph (total_students : ℝ) (h_positive : total_students > 0) :
  let male_ratio : ℝ := 2
  let female_ratio : ℝ := 3
  let male_preference_ratio : ℝ := 1/4
  let female_preference_ratio : ℝ := 3/4
  let total_ratio : ℝ := male_ratio + female_ratio
  let male_students : ℝ := (male_ratio / total_ratio) * total_students
  let female_students : ℝ := (female_ratio / total_ratio) * total_students
  let preference_students : ℝ := male_preference_ratio * male_students + female_preference_ratio * female_students
  let preference_ratio : ℝ := preference_students / total_students
  let circle_degrees : ℝ := 360
  
  preference_ratio * circle_degrees = 198 := by sorry

end NUMINAMATH_CALUDE_career_preference_circle_graph_l1954_195482


namespace NUMINAMATH_CALUDE_max_true_statements_l1954_195467

theorem max_true_statements (a b : ℝ) : 
  (∃ (s : Finset (Prop)), s.card = 2 ∧ 
    (∀ (p : Prop), p ∈ s → p) ∧
    s ⊆ {1/a > 1/b, a^3 > b^3, a > b, a < 0, b > 0}) ∧
  (∀ (s : Finset (Prop)), s.card > 2 → 
    s ⊆ {1/a > 1/b, a^3 > b^3, a > b, a < 0, b > 0} → 
    ∃ (p : Prop), p ∈ s ∧ ¬p) :=
sorry

end NUMINAMATH_CALUDE_max_true_statements_l1954_195467


namespace NUMINAMATH_CALUDE_most_likely_car_count_l1954_195408

/-- Represents the number of cars counted in a given time interval -/
structure CarCount where
  cars : ℕ
  seconds : ℕ

/-- Represents the total time taken by the train to pass -/
structure TotalTime where
  minutes : ℕ
  seconds : ℕ

/-- Calculates the most likely number of cars in the train -/
def calculateTotalCars (initial_count : CarCount) (total_time : TotalTime) : ℕ :=
  let total_seconds := total_time.minutes * 60 + total_time.seconds
  let rate := initial_count.cars / initial_count.seconds
  rate * total_seconds

/-- Theorem stating that given the conditions, the most likely number of cars is 70 -/
theorem most_likely_car_count 
  (initial_count : CarCount)
  (total_time : TotalTime)
  (h1 : initial_count = ⟨5, 15⟩)
  (h2 : total_time = ⟨3, 30⟩) :
  calculateTotalCars initial_count total_time = 70 := by
  sorry

#eval calculateTotalCars ⟨5, 15⟩ ⟨3, 30⟩

end NUMINAMATH_CALUDE_most_likely_car_count_l1954_195408


namespace NUMINAMATH_CALUDE_binomial_odd_iff_power_of_two_minus_one_l1954_195469

theorem binomial_odd_iff_power_of_two_minus_one (n : ℕ) :
  (∀ k : ℕ, k ≤ n → Odd (Nat.choose n k)) ↔
  ∃ m : ℕ, m ≥ 1 ∧ n = 2^m - 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_odd_iff_power_of_two_minus_one_l1954_195469


namespace NUMINAMATH_CALUDE_distance_driven_margies_car_distance_l1954_195466

/-- Proves that given a car's fuel efficiency and gas price, 
    we can calculate the distance driven with a certain amount of money. -/
theorem distance_driven (efficiency : ℝ) (gas_price : ℝ) (money : ℝ) :
  efficiency > 0 → gas_price > 0 → money > 0 →
  (efficiency * (money / gas_price) = 200) ↔ 
  (efficiency = 40 ∧ gas_price = 5 ∧ money = 25) :=
sorry

/-- Specific instance of the theorem for Margie's car -/
theorem margies_car_distance : 
  ∃ (efficiency gas_price money : ℝ),
    efficiency > 0 ∧ gas_price > 0 ∧ money > 0 ∧
    efficiency = 40 ∧ gas_price = 5 ∧ money = 25 ∧
    efficiency * (money / gas_price) = 200 :=
sorry

end NUMINAMATH_CALUDE_distance_driven_margies_car_distance_l1954_195466


namespace NUMINAMATH_CALUDE_smallest_multiple_of_eleven_l1954_195465

theorem smallest_multiple_of_eleven (x y : ℤ) 
  (h1 : ∃ k : ℤ, x + 2 = 11 * k) 
  (h2 : ∃ m : ℤ, y - 1 = 11 * m) : 
  (∃ n : ℕ+, ∃ p : ℤ, x^2 + x*y + y^2 + n = 11 * p) ∧ 
  (∀ n : ℕ+, n < 8 → ¬∃ p : ℤ, x^2 + x*y + y^2 + n = 11 * p) :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_eleven_l1954_195465


namespace NUMINAMATH_CALUDE_right_triangle_leg_sum_equals_circle_diameters_sum_l1954_195418

/-- 
A right triangle with inscribed and circumscribed circles.
-/
structure RightTriangle where
  -- Side lengths
  a : ℝ
  b : ℝ
  c : ℝ
  -- Radii of inscribed and circumscribed circles
  r : ℝ
  R : ℝ
  -- Conditions
  right_angle : c^2 = a^2 + b^2
  c_is_diameter : c = 2 * R
  nonneg : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < r ∧ 0 < R

/-- 
In a right triangle, the sum of the legs is equal to the sum of 
the diameters of the inscribed and circumscribed circles.
-/
theorem right_triangle_leg_sum_equals_circle_diameters_sum 
  (t : RightTriangle) : t.a + t.b = 2 * t.R + 2 * t.r := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_leg_sum_equals_circle_diameters_sum_l1954_195418


namespace NUMINAMATH_CALUDE_sum_of_squares_l1954_195402

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 400) : x^2 + y^2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1954_195402


namespace NUMINAMATH_CALUDE_exists_A_for_monomial_l1954_195497

-- Define what a monomial is
def is_monomial (e : ℝ → ℝ) : Prop :=
  ∃ (c : ℝ) (n : ℕ), ∀ x, e x = c * x^n

-- Define the expression -3x + A
def expr (A : ℝ → ℝ) (x : ℝ) : ℝ := -3*x + A x

-- Theorem statement
theorem exists_A_for_monomial :
  ∃ (A : ℝ → ℝ), is_monomial (expr A) :=
sorry

end NUMINAMATH_CALUDE_exists_A_for_monomial_l1954_195497


namespace NUMINAMATH_CALUDE_car_distance_proof_l1954_195409

/-- Proves that the distance covered by a car is 144 km given specific conditions -/
theorem car_distance_proof (initial_time : ℝ) (speed : ℝ) (time_factor : ℝ) :
  initial_time = 6 →
  speed = 16 →
  time_factor = 3/2 →
  initial_time * time_factor * speed = 144 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_proof_l1954_195409


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l1954_195422

open Real

theorem trigonometric_equation_solution (x : ℝ) :
  sin x + sin (2*x) + sin (3*x) = 1 + cos x + cos (2*x) ↔
  (∃ k : ℤ, x = π/2 + k * π) ∨
  (∃ k : ℤ, x = 2*π/3 + k * 2*π) ∨
  (∃ k : ℤ, x = 4*π/3 + k * 2*π) ∨
  (∃ k : ℤ, x = π/6 + k * 2*π) ∨
  (∃ k : ℤ, x = 5*π/6 + k * 2*π) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l1954_195422


namespace NUMINAMATH_CALUDE_mixed_fractions_sum_product_l1954_195438

theorem mixed_fractions_sum_product : 
  (9 + 1/2 + 7 + 1/6 + 5 + 1/12 + 3 + 1/20 + 1 + 1/30) * 12 = 310 := by
  sorry

end NUMINAMATH_CALUDE_mixed_fractions_sum_product_l1954_195438


namespace NUMINAMATH_CALUDE_certain_fraction_proof_l1954_195414

theorem certain_fraction_proof :
  ∀ (x y : ℚ),
  (3 : ℚ) / 5 / ((6 : ℚ) / 7) = (7 : ℚ) / 15 / (x / y) →
  x / y = (2 : ℚ) / 3 := by
  sorry

end NUMINAMATH_CALUDE_certain_fraction_proof_l1954_195414


namespace NUMINAMATH_CALUDE_cosine_graph_transformation_l1954_195432

theorem cosine_graph_transformation (x : ℝ) :
  let f (x : ℝ) := 2 * Real.cos (x + π / 3)
  let g (x : ℝ) := 2 * Real.cos (2 * x + π / 6)
  let h (x : ℝ) := f (2 * x)
  h (x - π / 12) = g x :=
by sorry

end NUMINAMATH_CALUDE_cosine_graph_transformation_l1954_195432


namespace NUMINAMATH_CALUDE_product_selection_probabilities_l1954_195473

/-- A box containing products -/
structure Box where
  total : ℕ
  good : ℕ
  defective : ℕ
  h_total : total = good + defective

/-- The probability of an event when selecting two products from a box -/
def probability (box : Box) (favorable : ℕ) : ℚ :=
  favorable / (box.total.choose 2)

theorem product_selection_probabilities (box : Box) 
  (h_total : box.total = 6)
  (h_good : box.good = 4)
  (h_defective : box.defective = 2) :
  probability box (box.good * box.defective) = 8 / 15 ∧
  probability box (box.good.choose 2) = 2 / 5 ∧
  1 - probability box (box.good.choose 2) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_product_selection_probabilities_l1954_195473


namespace NUMINAMATH_CALUDE_right_prism_cut_count_l1954_195447

theorem right_prism_cut_count : 
  let b : ℕ := 2023
  let count := (Finset.filter 
    (fun p : ℕ × ℕ => 
      let (a, c) := p
      a ≤ b ∧ b ≤ c ∧ a * c = b * b ∧ a < c)
    (Finset.product (Finset.range (b + 1)) (Finset.range (b * b + 1)))).card
  count = 13 := by
sorry

end NUMINAMATH_CALUDE_right_prism_cut_count_l1954_195447


namespace NUMINAMATH_CALUDE_bicycle_cost_proof_l1954_195442

def bicycle_cost (car_wash_income : ℕ) (lawn_mow_income : ℕ) (additional_needed : ℕ) : ℕ :=
  car_wash_income + lawn_mow_income + additional_needed

theorem bicycle_cost_proof :
  let car_wash_income := 3 * 10
  let lawn_mow_income := 2 * 13
  let additional_needed := 24
  bicycle_cost car_wash_income lawn_mow_income additional_needed = 80 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_cost_proof_l1954_195442


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l1954_195410

theorem arithmetic_mean_of_fractions :
  let a := 5 / 8
  let b := 9 / 16
  let c := 11 / 16
  a = (b + c) / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l1954_195410


namespace NUMINAMATH_CALUDE_expression_evaluation_l1954_195411

theorem expression_evaluation (y : ℝ) : 
  (1 : ℝ)^(4*y - 1) / (2 * ((7 : ℝ)⁻¹ + (4 : ℝ)⁻¹)) = 14/11 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1954_195411


namespace NUMINAMATH_CALUDE_exists_x_squared_minus_two_x_plus_one_nonpositive_l1954_195434

theorem exists_x_squared_minus_two_x_plus_one_nonpositive :
  ∃ x : ℝ, x^2 - 2*x + 1 ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_x_squared_minus_two_x_plus_one_nonpositive_l1954_195434


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1954_195462

/-- If the solution set of (1-m^2)x^2-(1+m)x-1<0 with respect to x is ℝ,
    then m satisfies m ≤ -1 or m > 5/3 -/
theorem inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, (1 - m^2) * x^2 - (1 + m) * x - 1 < 0) →
  (m ≤ -1 ∨ m > 5/3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1954_195462


namespace NUMINAMATH_CALUDE_sin_180_degrees_l1954_195437

theorem sin_180_degrees : Real.sin (π) = 0 := by sorry

end NUMINAMATH_CALUDE_sin_180_degrees_l1954_195437


namespace NUMINAMATH_CALUDE_calculation_proof_l1954_195489

theorem calculation_proof : (-3 : ℚ) * 6 / (-2) * (1/2) = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1954_195489


namespace NUMINAMATH_CALUDE_probability_5_heads_in_7_flips_l1954_195491

def fair_coin_probability (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) / (2 ^ n : ℚ)

theorem probability_5_heads_in_7_flips :
  fair_coin_probability 7 5 = 21 / 128 := by
  sorry

end NUMINAMATH_CALUDE_probability_5_heads_in_7_flips_l1954_195491


namespace NUMINAMATH_CALUDE_inequality_proof_l1954_195472

theorem inequality_proof (x y : ℝ) (hx : x > Real.sqrt 2) (hy : y > Real.sqrt 2) :
  x^4 - x^3*y + x^2*y^2 - x*y^3 + y^4 > x^2 + y^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1954_195472


namespace NUMINAMATH_CALUDE_chips_count_proof_l1954_195405

/-- The total number of chips Viviana and Susana have together -/
def total_chips (viviana_chocolate viviana_vanilla susana_chocolate susana_vanilla : ℕ) : ℕ :=
  viviana_chocolate + viviana_vanilla + susana_chocolate + susana_vanilla

theorem chips_count_proof :
  ∀ (viviana_vanilla susana_chocolate : ℕ),
    viviana_vanilla = 20 →
    susana_chocolate = 25 →
    ∃ (viviana_chocolate susana_vanilla : ℕ),
      viviana_chocolate = susana_chocolate + 5 ∧
      susana_vanilla = (3 * viviana_vanilla) / 4 ∧
      total_chips viviana_chocolate viviana_vanilla susana_chocolate susana_vanilla = 90 := by
  sorry

end NUMINAMATH_CALUDE_chips_count_proof_l1954_195405


namespace NUMINAMATH_CALUDE_tenth_term_is_negative_eight_l1954_195433

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  first_positive : a 1 > 0
  sum_condition : a 1 + a 7 = 2
  product_condition : a 5 * a 6 = -8
  is_geometric : ∀ n : ℕ, n > 0 → ∃ q : ℝ, a (n + 1) = a n * q

/-- The 10th term of the geometric sequence is -8 -/
theorem tenth_term_is_negative_eight (seq : GeometricSequence) : seq.a 10 = -8 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_is_negative_eight_l1954_195433


namespace NUMINAMATH_CALUDE_complex_equation_solutions_l1954_195435

theorem complex_equation_solutions :
  ∃! (s : Finset ℂ), (∀ z ∈ s, Complex.abs z < 20 ∧ Complex.exp z = (z - 1) / (z + 1)) ∧ Finset.card s = 8 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solutions_l1954_195435


namespace NUMINAMATH_CALUDE_midpoint_locus_is_annulus_l1954_195475

/-- Two non-intersecting circles in a plane --/
structure TwoCircles where
  center1 : ℝ × ℝ
  center2 : ℝ × ℝ
  radius1 : ℝ
  radius2 : ℝ
  h1 : radius1 > 0
  h2 : radius2 > 0
  h3 : radius1 > radius2
  h4 : dist center1 center2 > radius1 + radius2

/-- The locus of midpoints of segments with endpoints on two non-intersecting circles --/
def midpointLocus (c : TwoCircles) : Set (ℝ × ℝ) :=
  {p | ∃ (a b : ℝ × ℝ), 
    dist a c.center1 = c.radius1 ∧ 
    dist b c.center2 = c.radius2 ∧ 
    p = ((a.1 + b.1) / 2, (a.2 + b.2) / 2)}

/-- An annulus (ring) in a plane --/
def annulus (center : ℝ × ℝ) (inner_radius outer_radius : ℝ) : Set (ℝ × ℝ) :=
  {p | inner_radius ≤ dist p center ∧ dist p center ≤ outer_radius}

/-- The main theorem: the locus of midpoints is an annulus --/
theorem midpoint_locus_is_annulus (c : TwoCircles) :
  ∃ (center : ℝ × ℝ),
    midpointLocus c = annulus center ((c.radius1 - c.radius2) / 2) ((c.radius1 + c.radius2) / 2) :=
by sorry

end NUMINAMATH_CALUDE_midpoint_locus_is_annulus_l1954_195475


namespace NUMINAMATH_CALUDE_initial_ratio_is_four_to_one_l1954_195498

/-- Represents a mixture of milk and water -/
structure Mixture where
  milk : ℝ
  water : ℝ

/-- The initial mixture before adding water -/
def initial_mixture : Mixture :=
  { milk := 48, water := 12 }

/-- The final mixture after adding 60 litres of water -/
def final_mixture : Mixture :=
  { milk := initial_mixture.milk, water := initial_mixture.water + 60 }

/-- The total volume of the initial mixture -/
def initial_volume : ℝ := 60

theorem initial_ratio_is_four_to_one :
  initial_mixture.milk / initial_mixture.water = 4 ∧
  initial_mixture.milk + initial_mixture.water = initial_volume ∧
  final_mixture.milk / final_mixture.water = 1 / 2 := by
  sorry

#check initial_ratio_is_four_to_one

end NUMINAMATH_CALUDE_initial_ratio_is_four_to_one_l1954_195498


namespace NUMINAMATH_CALUDE_trig_identity_l1954_195452

theorem trig_identity (α : ℝ) : 
  (Real.sin (α - π/6))^2 + (Real.sin (α + π/6))^2 - (Real.sin α)^2 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1954_195452


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_two_l1954_195416

theorem sqrt_expression_equals_two :
  Real.sqrt 12 + Real.sqrt 4 * (Real.sqrt 5 - Real.pi) ^ 0 - |(-2 * Real.sqrt 3)| = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_two_l1954_195416


namespace NUMINAMATH_CALUDE_line_slope_problem_l1954_195417

theorem line_slope_problem (k : ℝ) (h1 : k > 0) 
  (h2 : (k + 1) * (2 - k) = k - 5) : k = (1 + Real.sqrt 29) / 2 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_problem_l1954_195417


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_diff_l1954_195415

-- Define the repeating decimals
def repeating_234 : ℚ := 234 / 999
def repeating_567 : ℚ := 567 / 999
def repeating_891 : ℚ := 891 / 999

-- State the theorem
theorem repeating_decimal_sum_diff : 
  repeating_234 + repeating_567 - repeating_891 = -10 / 111 := by
sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_diff_l1954_195415


namespace NUMINAMATH_CALUDE_simplify_expression_l1954_195483

theorem simplify_expression (a b : ℝ) : 105*a - 38*a + 27*b - 12*b = 67*a + 15*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1954_195483


namespace NUMINAMATH_CALUDE_cos_arcsin_three_fifths_l1954_195460

theorem cos_arcsin_three_fifths : 
  Real.cos (Real.arcsin (3/5)) = 4/5 := by sorry

end NUMINAMATH_CALUDE_cos_arcsin_three_fifths_l1954_195460


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1954_195490

theorem quadratic_inequality_range (a : ℝ) :
  (∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) ↔ a < -1 ∨ a > 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1954_195490


namespace NUMINAMATH_CALUDE_fraction_equality_implies_value_l1954_195424

theorem fraction_equality_implies_value (b : ℝ) :
  b / (b + 30) = 0.92 → b = 345 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_value_l1954_195424


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1954_195440

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (a 20 + a 21 = 10) →
  (a 22 + a 23 = 20) →
  (a 24 + a 25 = 40) :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1954_195440


namespace NUMINAMATH_CALUDE_infinite_series_sum_l1954_195446

open Real

noncomputable def series_sum (n : ℕ) : ℝ :=
  3^n / (1 + 3^n + 3^(n+2) + 3^(2*n+2))

theorem infinite_series_sum :
  (∑' n, series_sum n) = (1 : ℝ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l1954_195446


namespace NUMINAMATH_CALUDE_cultural_festival_talents_l1954_195476

theorem cultural_festival_talents (total_students : ℕ) 
  (cannot_sing cannot_dance cannot_act no_talents : ℕ) : ℕ :=
by
  -- Define the conditions
  have h1 : total_students = 150 := by sorry
  have h2 : cannot_sing = 75 := by sorry
  have h3 : cannot_dance = 95 := by sorry
  have h4 : cannot_act = 40 := by sorry
  have h5 : no_talents = 20 := by sorry
  
  -- Define the number of students with each talent
  let can_sing := total_students - cannot_sing
  let can_dance := total_students - cannot_dance
  let can_act := total_students - cannot_act
  
  -- Define the sum of students with at least one talent
  let with_talents := total_students - no_talents
  
  -- Define the sum of all talents (ignoring overlaps)
  let sum_talents := can_sing + can_dance + can_act
  
  -- Calculate the number of students with exactly two talents
  let two_talents := sum_talents - with_talents
  
  -- Prove that two_talents equals 90
  have h6 : two_talents = 90 := by sorry
  
  -- Return the result
  exact two_talents

-- The theorem states that given the conditions, 
-- the number of students with exactly two talents is 90

end NUMINAMATH_CALUDE_cultural_festival_talents_l1954_195476


namespace NUMINAMATH_CALUDE_no_right_prism_with_diagonals_4_5_7_l1954_195468

theorem no_right_prism_with_diagonals_4_5_7 :
  ¬∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
    x^2 + y^2 = 16 ∧ x^2 + z^2 = 25 ∧ y^2 + z^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_no_right_prism_with_diagonals_4_5_7_l1954_195468


namespace NUMINAMATH_CALUDE_fraction_evaluation_l1954_195412

theorem fraction_evaluation (a b : ℝ) (h1 : a = 5) (h2 : b = 3) :
  3 / (a + b) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l1954_195412


namespace NUMINAMATH_CALUDE_sqrt_sum_reciprocal_l1954_195444

theorem sqrt_sum_reciprocal (x : ℝ) (hx_pos : x > 0) (hx_sum : x + 1/x = 50) :
  Real.sqrt x + 1 / Real.sqrt x = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_reciprocal_l1954_195444


namespace NUMINAMATH_CALUDE_cookie_ratio_l1954_195406

/-- Proves that the ratio of Glenn's cookies to Kenny's cookies is 4:1 given the problem conditions --/
theorem cookie_ratio (kenny : ℕ) (glenn : ℕ) (chris : ℕ) : 
  chris = kenny / 2 → 
  glenn = 24 → 
  chris + kenny + glenn = 33 → 
  glenn / kenny = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_cookie_ratio_l1954_195406


namespace NUMINAMATH_CALUDE_delta_fourth_order_zero_l1954_195499

def u (n : ℕ) : ℕ := n^3 + 2*n

def Δ : (ℕ → ℕ) → (ℕ → ℕ)
| f => λ n => f (n + 1) - f n

def iteratedΔ : ℕ → (ℕ → ℕ) → (ℕ → ℕ)
| 0 => id
| k + 1 => Δ ∘ iteratedΔ k

theorem delta_fourth_order_zero (n : ℕ) : 
  ∀ k : ℕ, (∀ n : ℕ, iteratedΔ k u n = 0) ↔ k ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_delta_fourth_order_zero_l1954_195499


namespace NUMINAMATH_CALUDE_ball_drawing_theorem_l1954_195445

def num_red_balls : ℕ := 5
def num_black_balls : ℕ := 7
def red_ball_score : ℕ := 2
def black_ball_score : ℕ := 1
def total_balls_drawn : ℕ := 6
def max_score : ℕ := 8

def ways_to_draw_balls : ℕ :=
  (Nat.choose num_black_balls total_balls_drawn) +
  (Nat.choose num_red_balls 1 * Nat.choose num_black_balls (total_balls_drawn - 1))

theorem ball_drawing_theorem :
  ways_to_draw_balls = 112 :=
sorry

end NUMINAMATH_CALUDE_ball_drawing_theorem_l1954_195445


namespace NUMINAMATH_CALUDE_existence_of_n_for_prime_divisibility_l1954_195454

theorem existence_of_n_for_prime_divisibility (p : ℕ) (hp : Nat.Prime p) :
  ∃ n : ℕ, p ∣ (2^n + 3^n + 6^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_n_for_prime_divisibility_l1954_195454


namespace NUMINAMATH_CALUDE_factorization_proof_l1954_195423

/-- Prove the factorization of two polynomial expressions -/
theorem factorization_proof (x y : ℝ) : 
  (2 * x^2 * y - 4 * x * y + 2 * y = 2 * y * (x - 1)^2) ∧ 
  (x^4 - 9 * x^2 = x^2 * (x + 3) * (x - 3)) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l1954_195423


namespace NUMINAMATH_CALUDE_cos_negative_1830_degrees_l1954_195471

theorem cos_negative_1830_degrees : Real.cos ((-1830 : ℝ) * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_1830_degrees_l1954_195471


namespace NUMINAMATH_CALUDE_parabola_equation_l1954_195461

/-- A parabola in the Cartesian coordinate system -/
structure Parabola where
  -- The equation of the parabola in the form y^2 = ax
  a : ℝ
  -- Condition that the parabola is symmetric with respect to the x-axis
  x_axis_symmetry : True
  -- Condition that the vertex is at the origin
  vertex_at_origin : True
  -- Condition that the parabola passes through the point (2, 4)
  passes_through_point : a * 2 = 4^2

/-- Theorem stating that the parabola y^2 = 8x satisfies the given conditions -/
theorem parabola_equation : ∃ (p : Parabola), p.a = 8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l1954_195461


namespace NUMINAMATH_CALUDE_infinite_solutions_equation_l1954_195457

theorem infinite_solutions_equation :
  ∃ (S : Set (ℕ × ℕ × ℕ)), 
    (∀ (x y z : ℕ), (x, y, z) ∈ S → 
      x > 2008 ∧ y > 2008 ∧ z > 2008 ∧ 
      x^2 + y^2 + z^2 - x*y*z + 10 = 0) ∧
    Set.Infinite S :=
sorry

end NUMINAMATH_CALUDE_infinite_solutions_equation_l1954_195457


namespace NUMINAMATH_CALUDE_hares_per_rabbit_l1954_195474

theorem hares_per_rabbit (dog : Nat) (cats : Nat) (rabbits_per_cat : Nat) (total_animals : Nat) :
  dog = 1 →
  cats = 4 →
  rabbits_per_cat = 2 →
  total_animals = 37 →
  ∃ hares_per_rabbit : Nat, 
    total_animals = dog + cats + (cats * rabbits_per_cat) + (cats * rabbits_per_cat * hares_per_rabbit) ∧
    hares_per_rabbit = 3 := by
  sorry

end NUMINAMATH_CALUDE_hares_per_rabbit_l1954_195474


namespace NUMINAMATH_CALUDE_job_completion_days_l1954_195486

/-- Represents the job scenario with given parameters -/
structure JobScenario where
  total_days : ℕ
  initial_workers : ℕ
  days_worked : ℕ
  work_done_fraction : ℚ
  fired_workers : ℕ

/-- Calculates the remaining days to complete the job -/
def remaining_days (job : JobScenario) : ℕ :=
  sorry

/-- Theorem statement for the job completion problem -/
theorem job_completion_days (job : JobScenario) 
  (h1 : job.total_days = 100)
  (h2 : job.initial_workers = 10)
  (h3 : job.days_worked = 20)
  (h4 : job.work_done_fraction = 1/4)
  (h5 : job.fired_workers = 2) :
  remaining_days job = 75 := by sorry

end NUMINAMATH_CALUDE_job_completion_days_l1954_195486


namespace NUMINAMATH_CALUDE_photo_arrangements_l1954_195496

def num_students : ℕ := 7

def arrangements (n : ℕ) (pair_together : Bool) (avoid_adjacent : Bool) : ℕ := 
  sorry

theorem photo_arrangements : 
  arrangements num_students true true = 1200 := by sorry

end NUMINAMATH_CALUDE_photo_arrangements_l1954_195496
