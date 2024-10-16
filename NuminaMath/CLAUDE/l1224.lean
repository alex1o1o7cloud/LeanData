import Mathlib

namespace NUMINAMATH_CALUDE_compound_molecular_weight_l1224_122425

/-- Atomic weight of Copper in g/mol -/
def copper_weight : ℝ := 63.546

/-- Atomic weight of Carbon in g/mol -/
def carbon_weight : ℝ := 12.011

/-- Atomic weight of Oxygen in g/mol -/
def oxygen_weight : ℝ := 15.999

/-- Number of Copper atoms in the compound -/
def copper_count : ℕ := 1

/-- Number of Carbon atoms in the compound -/
def carbon_count : ℕ := 1

/-- Number of Oxygen atoms in the compound -/
def oxygen_count : ℕ := 3

/-- Molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := 
  copper_count * copper_weight + 
  carbon_count * carbon_weight + 
  oxygen_count * oxygen_weight

/-- Theorem stating that the molecular weight of the compound is 123.554 g/mol -/
theorem compound_molecular_weight : 
  molecular_weight = 123.554 := by sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l1224_122425


namespace NUMINAMATH_CALUDE_probability_of_non_intersection_l1224_122478

-- Define the circles and their properties
def CircleA : Type := Unit
def CircleB : Type := Unit

-- Define the probability space
def Ω : Type := CircleA × CircleB

-- Define the center distributions
def centerA_distribution : Set ℝ := Set.Icc 0 2
def centerB_distribution : Set ℝ := Set.Icc 0 3

-- Define the radius of each circle
def radiusA : ℝ := 2
def radiusB : ℝ := 1

-- Define the probability measure
def P : Set Ω → ℝ := sorry

-- Define the event of non-intersection
def non_intersection : Set Ω := sorry

-- Theorem statement
theorem probability_of_non_intersection :
  P non_intersection = (4 * Real.sqrt 5 - 5) / 3 := by sorry

end NUMINAMATH_CALUDE_probability_of_non_intersection_l1224_122478


namespace NUMINAMATH_CALUDE_smallest_abs_z_l1224_122416

theorem smallest_abs_z (z : ℂ) (h : Complex.abs (z - 9) + Complex.abs (z - 6*I) = 15) : 
  ∃ (min_abs_z : ℝ), min_abs_z = 3.6 ∧ ∀ w : ℂ, Complex.abs (w - 9) + Complex.abs (w - 6*I) = 15 → Complex.abs w ≥ min_abs_z :=
sorry

end NUMINAMATH_CALUDE_smallest_abs_z_l1224_122416


namespace NUMINAMATH_CALUDE_three_heads_probability_l1224_122461

/-- The probability of getting heads on a single flip of a biased coin. -/
def p_heads : ℚ := 1 / 3

/-- The number of consecutive flips we're considering. -/
def n_flips : ℕ := 3

/-- The probability of getting n_flips consecutive heads. -/
def p_all_heads : ℚ := p_heads ^ n_flips

theorem three_heads_probability :
  p_all_heads = 1 / 27 := by sorry

end NUMINAMATH_CALUDE_three_heads_probability_l1224_122461


namespace NUMINAMATH_CALUDE_function_properties_l1224_122408

-- Define the function f and its derivative
def f (a b m x : ℝ) : ℝ := 2 * x^3 + a * x^2 + b * x + m
def f' (a b x : ℝ) : ℝ := 6 * x^2 + 2 * a * x + b

-- Main theorem
theorem function_properties (a b m : ℝ) :
  (∀ x : ℝ, f' a b x = f' a b (-1 - x)) →  -- f' is symmetric about x = -1/2
  (f' a b 1 = 0) →                         -- f'(1) = 0
  (a = 3 ∧ b = -12) ∧                      -- Part 1: values of a and b
  (∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧     -- f has exactly three zeros
    f 3 (-12) m x₁ = 0 ∧
    f 3 (-12) m x₂ = 0 ∧
    f 3 (-12) m x₃ = 0 ∧
    ∀ x : ℝ, f 3 (-12) m x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃) →
  -20 < m ∧ m < 7                          -- Part 2: range of m
  := by sorry


end NUMINAMATH_CALUDE_function_properties_l1224_122408


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l1224_122411

theorem quadratic_expression_value (x y : ℝ) 
  (h1 : 4 * x + y = 12) 
  (h2 : x + 4 * y = 18) : 
  20 * x^2 + 24 * x * y + 20 * y^2 = 468 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l1224_122411


namespace NUMINAMATH_CALUDE_orange_box_problem_l1224_122433

theorem orange_box_problem (box1_capacity box2_capacity : ℕ) 
  (box1_fill_fraction : ℚ) (total_oranges : ℕ) :
  box1_capacity = 80 →
  box2_capacity = 50 →
  box1_fill_fraction = 3/4 →
  total_oranges = 90 →
  ∃ (box2_fill_fraction : ℚ),
    box2_fill_fraction = 3/5 ∧
    (box1_capacity : ℚ) * box1_fill_fraction + (box2_capacity : ℚ) * box2_fill_fraction = total_oranges := by
  sorry

end NUMINAMATH_CALUDE_orange_box_problem_l1224_122433


namespace NUMINAMATH_CALUDE_average_of_three_numbers_l1224_122463

theorem average_of_three_numbers (M : ℝ) (h1 : 12 < M) (h2 : M < 25) : 
  ∃ k : ℝ, k = 5 ∧ (8 + 15 + (M + k)) / 3 = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_average_of_three_numbers_l1224_122463


namespace NUMINAMATH_CALUDE_three_digit_automorphic_numbers_l1224_122432

theorem three_digit_automorphic_numbers : 
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ n^2 % 1000 = n} = {625, 376} := by sorry

end NUMINAMATH_CALUDE_three_digit_automorphic_numbers_l1224_122432


namespace NUMINAMATH_CALUDE_percentage_calculation_l1224_122450

theorem percentage_calculation (number : ℝ) (p : ℝ) 
  (h1 : (4/5) * (3/8) * number = 24) 
  (h2 : p * number / 100 = 199.99999999999997) : 
  p = 250 := by sorry

end NUMINAMATH_CALUDE_percentage_calculation_l1224_122450


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_l1224_122441

theorem negation_of_forall_positive (R : Type) [OrderedRing R] :
  (¬ (∀ x : R, x^2 + x + 1 > 0)) ↔ (∃ x : R, x^2 + x + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_l1224_122441


namespace NUMINAMATH_CALUDE_divisor_count_theorem_l1224_122494

/-- The set of possible values for k -/
def possible_k : Set Nat := {45, 53, 253, 280}

/-- Definition of a valid divisor sequence -/
def is_valid_divisor_sequence (d : Nat → Nat) (k : Nat) : Prop :=
  d 1 = 1 ∧ d k > 1 ∧ 
  (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ k → d i < d j) ∧
  (∀ i, 1 ≤ i ∧ i ≤ k → (∃ m, d m = i ∨ d m = (d k) / i))

/-- Main theorem -/
theorem divisor_count_theorem (n : Nat) (d : Nat → Nat) (k : Nat) :
  n > 0 →
  k ≥ 5 →
  k ≤ 1000 →
  is_valid_divisor_sequence d k →
  n = (d 2) ^ (d 3) * (d 4) ^ (d 5) →
  k ∈ possible_k :=
sorry

end NUMINAMATH_CALUDE_divisor_count_theorem_l1224_122494


namespace NUMINAMATH_CALUDE_gcd_7163_209_l1224_122436

theorem gcd_7163_209 :
  let a := 7163
  let b := 209
  let c := 57
  let d := 38
  let e := 19
  a = b * 34 + c →
  b = c * 3 + d →
  c = d * 1 + e →
  d = e * 2 →
  Nat.gcd a b = e :=
by sorry

end NUMINAMATH_CALUDE_gcd_7163_209_l1224_122436


namespace NUMINAMATH_CALUDE_inverse_of_100_mod_101_l1224_122451

theorem inverse_of_100_mod_101 : ∃ x : ℕ, 0 ≤ x ∧ x ≤ 100 ∧ (100 * x) % 101 = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_100_mod_101_l1224_122451


namespace NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l1224_122474

theorem square_sum_zero_implies_both_zero (a b : ℝ) (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l1224_122474


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l1224_122410

theorem max_sum_of_factors (A B : ℕ) (h : A * B = 2005) :
  A + B ≤ 2006 ∧ ∃ (X Y : ℕ), X * Y = 2005 ∧ X + Y = 2006 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l1224_122410


namespace NUMINAMATH_CALUDE_monochromatic_triangle_exists_l1224_122498

/-- A coloring of the edges of a complete graph on 10 vertices using two colors -/
def TwoColoring : Type := Fin 10 → Fin 10 → Bool

/-- A triangle in a graph is represented by three distinct vertices -/
structure Triangle (n : Nat) where
  v1 : Fin n
  v2 : Fin n
  v3 : Fin n
  distinct : v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3

/-- A triangle is monochromatic if all its edges have the same color -/
def isMonochromatic (c : TwoColoring) (t : Triangle 10) : Prop :=
  c t.v1 t.v2 = c t.v2 t.v3 ∧ c t.v2 t.v3 = c t.v3 t.v1

/-- The main theorem: every two-coloring of K_10 contains a monochromatic triangle -/
theorem monochromatic_triangle_exists (c : TwoColoring) : 
  ∃ t : Triangle 10, isMonochromatic c t := by
  sorry


end NUMINAMATH_CALUDE_monochromatic_triangle_exists_l1224_122498


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1224_122483

theorem geometric_sequence_problem (a : ℕ → ℝ) :
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) →  -- geometric sequence condition
  a 3 = 4 →
  a 7 = 12 →
  a 11 = 36 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1224_122483


namespace NUMINAMATH_CALUDE_ratio_to_percent_l1224_122447

theorem ratio_to_percent (a b : ℕ) (h : a = 2 ∧ b = 3) : (a : ℚ) / (a + b : ℚ) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_ratio_to_percent_l1224_122447


namespace NUMINAMATH_CALUDE_log_equation_solution_l1224_122430

theorem log_equation_solution (x : ℝ) :
  x > 0 →
  (Real.log x / Real.log 4) * (Real.log 8 / Real.log x) = Real.log 8 / Real.log 4 →
  x = 4 ∨ x = 8 :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1224_122430


namespace NUMINAMATH_CALUDE_count_divisible_sum_l1224_122409

theorem count_divisible_sum : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, n > 0 ∧ (n * (n + 1) / 2) ∣ (8 * n)) ∧ 
  (∀ n : ℕ, n > 0 ∧ (n * (n + 1) / 2) ∣ (8 * n) → n ∈ S) ∧ 
  Finset.card S = 4 := by
  sorry

end NUMINAMATH_CALUDE_count_divisible_sum_l1224_122409


namespace NUMINAMATH_CALUDE_mason_savings_l1224_122491

theorem mason_savings (total_savings : ℚ) (days : ℕ) (dime_value : ℚ) : 
  total_savings = 3 → days = 30 → dime_value = 0.1 → 
  (total_savings / days) * dime_value = 0.01 := by
sorry

end NUMINAMATH_CALUDE_mason_savings_l1224_122491


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1224_122468

/-- Two vectors in ℝ² -/
def a : ℝ × ℝ := (4, -2)
def b : ℝ → ℝ × ℝ := λ x ↦ (x, 5)

/-- Parallel vectors in ℝ² -/
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_x_value :
  parallel a (b x) → x = -10 := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1224_122468


namespace NUMINAMATH_CALUDE_train_length_l1224_122485

/-- The length of a train given its speed and time to pass a stationary observer -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 108 → time_s = 6 → speed_kmh * (5/18) * time_s = 180 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l1224_122485


namespace NUMINAMATH_CALUDE_garden_plants_l1224_122479

/-- The total number of plants in a rectangular garden -/
def total_plants (rows : ℕ) (columns : ℕ) : ℕ := rows * columns

/-- Theorem: A garden with 52 rows and 15 columns has 780 plants in total -/
theorem garden_plants : total_plants 52 15 = 780 := by
  sorry

end NUMINAMATH_CALUDE_garden_plants_l1224_122479


namespace NUMINAMATH_CALUDE_elevatorProblem_l1224_122449

-- Define the type of elevator move
inductive Move
| Up7 : Move
| Up10 : Move
| Down7 : Move
| Down10 : Move

-- Function to apply a move to a floor number
def applyMove (floor : ℕ) (move : Move) : ℕ :=
  match move with
  | Move.Up7 => floor + 7
  | Move.Up10 => floor + 10
  | Move.Down7 => if floor ≥ 7 then floor - 7 else floor
  | Move.Down10 => if floor ≥ 10 then floor - 10 else floor

-- Function to check if a floor is visited in a sequence of moves
def isVisited (startFloor : ℕ) (moves : List Move) (targetFloor : ℕ) : Prop :=
  targetFloor ∈ List.scanl applyMove startFloor moves

-- Theorem stating the existence of a valid sequence of moves
theorem elevatorProblem : 
  ∃ (moves : List Move), 
    moves.length ≤ 10 ∧ 
    isVisited 1 moves 13 ∧ 
    isVisited 1 moves 16 ∧ 
    isVisited 1 moves 24 :=
by
  sorry


end NUMINAMATH_CALUDE_elevatorProblem_l1224_122449


namespace NUMINAMATH_CALUDE_no_four_digit_numbers_with_one_eighth_property_l1224_122404

theorem no_four_digit_numbers_with_one_eighth_property : 
  ¬∃ (N : ℕ), 
    (1000 ≤ N ∧ N < 10000) ∧ 
    (∃ (a x : ℕ), 
      1 ≤ a ∧ a ≤ 9 ∧
      100 ≤ x ∧ x < 1000 ∧
      N = 1000 * a + x ∧
      x = N / 8) := by
sorry

end NUMINAMATH_CALUDE_no_four_digit_numbers_with_one_eighth_property_l1224_122404


namespace NUMINAMATH_CALUDE_smallest_integer_fraction_eleven_satisfies_smallest_integer_is_eleven_l1224_122469

theorem smallest_integer_fraction (y : ℤ) : (5 : ℚ) / 8 < (y : ℚ) / 17 → y ≥ 11 :=
by sorry

theorem eleven_satisfies (y : ℤ) : (5 : ℚ) / 8 < (11 : ℚ) / 17 :=
by sorry

theorem smallest_integer_is_eleven : 
  ∃ y : ℤ, ((5 : ℚ) / 8 < (y : ℚ) / 17) ∧ (∀ z : ℤ, (5 : ℚ) / 8 < (z : ℚ) / 17 → z ≥ y) ∧ y = 11 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_fraction_eleven_satisfies_smallest_integer_is_eleven_l1224_122469


namespace NUMINAMATH_CALUDE_molar_mass_X1_l1224_122481

-- Define the substances
def X1 : String := "CuO"
def X2 : String := "Cu"
def X3 : String := "CuSO4"
def X4 : String := "Cu(OH)2"

-- Define the molar masses
def molar_mass_Cu : Float := 63.5
def molar_mass_O : Float := 16.0

-- Define the chemical reactions
def reaction1 : String := "X1 + H2 → X2 + H2O"
def reaction2 : String := "X2 + H2SO4 → X3 + H2"
def reaction3 : String := "X3 + 2KOH → X4 + K2SO4"
def reaction4 : String := "X4 → X1 + H2O"

-- Define the properties of the substances
def X1_properties : String := "black powder"
def X2_properties : String := "red-colored substance"
def X3_properties : String := "blue-colored solution"
def X4_properties : String := "blue precipitate"

-- Theorem to prove
theorem molar_mass_X1 : 
  molar_mass_Cu + molar_mass_O = 79.5 := by sorry

end NUMINAMATH_CALUDE_molar_mass_X1_l1224_122481


namespace NUMINAMATH_CALUDE_unique_solution_modulo_l1224_122475

theorem unique_solution_modulo : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 14 ∧ n ≡ 16427 [ZMOD 15] := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_modulo_l1224_122475


namespace NUMINAMATH_CALUDE_senior_tickets_sold_l1224_122455

/-- Proves the number of senior citizen tickets sold given the total tickets,
    ticket prices, and total receipts -/
theorem senior_tickets_sold
  (total_tickets : ℕ)
  (adult_price senior_price : ℕ)
  (total_receipts : ℕ)
  (h1 : total_tickets = 510)
  (h2 : adult_price = 21)
  (h3 : senior_price = 15)
  (h4 : total_receipts = 8748) :
  ∃ (adult_tickets senior_tickets : ℕ),
    adult_tickets + senior_tickets = total_tickets ∧
    adult_price * adult_tickets + senior_price * senior_tickets = total_receipts ∧
    senior_tickets = 327 :=
by sorry

end NUMINAMATH_CALUDE_senior_tickets_sold_l1224_122455


namespace NUMINAMATH_CALUDE_distance_between_tangent_circles_l1224_122492

/-- The distance between the centers of two externally tangent circles -/
def distance_between_centers (r1 r2 : ℝ) : ℝ := r1 + r2

/-- Two circles are externally tangent -/
axiom externally_tangent (O O' : Set ℝ) : Prop

theorem distance_between_tangent_circles 
  (O O' : Set ℝ) (r1 r2 : ℝ) 
  (h1 : externally_tangent O O')
  (h2 : r1 = 8)
  (h3 : r2 = 3) : 
  distance_between_centers r1 r2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_tangent_circles_l1224_122492


namespace NUMINAMATH_CALUDE_object_height_l1224_122406

/-- The height function of an object thrown upward -/
def h (k : ℝ) (t : ℝ) : ℝ := -k * (t - 3)^2 + 150

/-- The value of k for which the object is at 94 feet after 5 seconds -/
theorem object_height (k : ℝ) : h k 5 = 94 → k = 14 := by
  sorry

end NUMINAMATH_CALUDE_object_height_l1224_122406


namespace NUMINAMATH_CALUDE_monotonic_f_implies_a_in_range_l1224_122422

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - 4) * (x - a)

-- State the theorem
theorem monotonic_f_implies_a_in_range (a : ℝ) :
  (∀ x y, x ≤ y ∧ y ≤ -2 → f a x ≤ f a y) ∧
  (∀ x y, 2 ≤ x ∧ x ≤ y → f a x ≤ f a y) →
  -2 ≤ a ∧ a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_monotonic_f_implies_a_in_range_l1224_122422


namespace NUMINAMATH_CALUDE_perp_planes_necessary_not_sufficient_l1224_122489

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes
variable (perp_planes : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the containment relation between a line and a plane
variable (contained_in : Line → Plane → Prop)

-- Main theorem
theorem perp_planes_necessary_not_sufficient 
  (α β : Plane) (m : Line) 
  (h_different : α ≠ β)
  (h_contained : contained_in m α) :
  (∀ m, contained_in m α → perp_line_plane m β → perp_planes α β) ∧ 
  (∃ m, contained_in m α ∧ perp_planes α β ∧ ¬perp_line_plane m β) :=
sorry

end NUMINAMATH_CALUDE_perp_planes_necessary_not_sufficient_l1224_122489


namespace NUMINAMATH_CALUDE_uncle_welly_roses_l1224_122493

def roses_problem (day1 : ℕ) (extra : ℕ) : Prop :=
  let day2 := day1 + extra
  let day3 := 2 * day1
  day1 + day2 + day3 = 220

theorem uncle_welly_roses : roses_problem 50 20 := by
  sorry

end NUMINAMATH_CALUDE_uncle_welly_roses_l1224_122493


namespace NUMINAMATH_CALUDE_set_equality_l1224_122465

def M : Set ℝ := {x | ∃ n : ℤ, x = n}
def N : Set ℝ := {x | ∃ n : ℤ, x = n / 2}
def P : Set ℝ := {x | ∃ n : ℤ, x = n + 1 / 2}

theorem set_equality : N = M ∪ P := by sorry

end NUMINAMATH_CALUDE_set_equality_l1224_122465


namespace NUMINAMATH_CALUDE_expression_value_l1224_122419

theorem expression_value (x y : ℝ) (h : |x + 1| + (2 * y - 4)^2 = 0) :
  (2 * x^2 * y - 3 * x * y) - 2 * (x^2 * y - x * y + 1/2 * x * y^2) + x * y = 4 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l1224_122419


namespace NUMINAMATH_CALUDE_min_groups_for_children_l1224_122496

/-- Given a total of 30 children and a maximum of 7 children per group,
    prove that the minimum number of equal-sized groups needed is 5. -/
theorem min_groups_for_children (total_children : Nat) (max_per_group : Nat) 
    (h1 : total_children = 30) (h2 : max_per_group = 7) : 
    (∃ (group_size : Nat), group_size ≤ max_per_group ∧ 
    total_children % group_size = 0 ∧ 
    total_children / group_size = 5 ∧
    ∀ (other_size : Nat), other_size ≤ max_per_group ∧ 
    total_children % other_size = 0 → 
    total_children / other_size ≥ 5) := by
  sorry

end NUMINAMATH_CALUDE_min_groups_for_children_l1224_122496


namespace NUMINAMATH_CALUDE_min_value_of_f_f_is_even_f_monotone_increasing_l1224_122497

noncomputable section

-- Define the operation *
def star (a b : ℝ) : ℝ := a * b + a + b

-- Define the function f
def f (x : ℝ) : ℝ := 1 + Real.exp x + 1 / Real.exp x

-- Theorem statements
theorem min_value_of_f : ∀ x : ℝ, f x ≥ 3 ∧ ∃ x₀ : ℝ, f x₀ = 3 := by sorry

theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by sorry

theorem f_monotone_increasing : 
  ∀ x y : ℝ, x ≥ 0 → y ≥ x → f y ≥ f x := by sorry

end

end NUMINAMATH_CALUDE_min_value_of_f_f_is_even_f_monotone_increasing_l1224_122497


namespace NUMINAMATH_CALUDE_complex_number_problem_l1224_122426

theorem complex_number_problem (a : ℝ) (z : ℂ) : 
  z = a + Complex.I * Real.sqrt 3 → z * z = 4 → a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l1224_122426


namespace NUMINAMATH_CALUDE_pet_store_ratio_l1224_122464

theorem pet_store_ratio (dogs : ℕ) (total : ℕ) : 
  dogs = 6 → 
  total = 39 → 
  (dogs + dogs / 2 + 2 * dogs + (total - (dogs + dogs / 2 + 2 * dogs))) / dogs = 3 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_ratio_l1224_122464


namespace NUMINAMATH_CALUDE_win_sector_area_l1224_122445

theorem win_sector_area (r : ℝ) (p : ℝ) (h1 : r = 12) (h2 : p = 1/4) :
  p * (π * r^2) = 36 * π := by
  sorry

end NUMINAMATH_CALUDE_win_sector_area_l1224_122445


namespace NUMINAMATH_CALUDE_fish_count_difference_l1224_122415

theorem fish_count_difference (n G S R : ℕ) : 
  n > 0 → 
  n = G + S + R → 
  n - G = (2 * n) / 3 - 1 → 
  n - R = (2 * n) / 3 + 4 → 
  S = G + 2 :=
by sorry

end NUMINAMATH_CALUDE_fish_count_difference_l1224_122415


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1224_122413

theorem solution_set_inequality (x : ℝ) : 
  (x - 1) * (2 - x) ≥ 0 ↔ 1 ≤ x ∧ x ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1224_122413


namespace NUMINAMATH_CALUDE_toothpick_grid_count_l1224_122453

/-- Calculates the number of toothpicks in a grid with missing toothpicks in regular intervals -/
def toothpick_count (length width : ℕ) (row_interval column_interval : ℕ) : ℕ :=
  let vertical_lines := length + 1
  let horizontal_lines := width + 1
  let vertical_missing := (vertical_lines / column_interval) * width
  let horizontal_missing := (horizontal_lines / row_interval) * length
  let vertical_count := vertical_lines * width - vertical_missing
  let horizontal_count := horizontal_lines * length - horizontal_missing
  vertical_count + horizontal_count

/-- The total number of toothpicks in the specified grid -/
theorem toothpick_grid_count :
  toothpick_count 45 25 5 4 = 2304 := by sorry

end NUMINAMATH_CALUDE_toothpick_grid_count_l1224_122453


namespace NUMINAMATH_CALUDE_expression_equality_l1224_122486

theorem expression_equality : 150 * (150 - 8) - (150 * 150 - 8) = -1192 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1224_122486


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_1000_l1224_122454

theorem consecutive_integers_sum_1000 :
  ∃ (m k : ℕ), m > 0 ∧ k ≥ 0 ∧ (k + 1) * (2 * m + k) = 2000 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_1000_l1224_122454


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1224_122476

-- Define the inequality
def inequality (x : ℝ) : Prop := (x - 2) / (x + 1) ≤ 0

-- Define the solution set
def solution_set : Set ℝ := {x | x > -1 ∧ x ≤ 2}

-- Theorem statement
theorem inequality_solution_set :
  {x : ℝ | inequality x ∧ x + 1 ≠ 0} = solution_set := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1224_122476


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l1224_122473

theorem sum_of_reciprocals_of_roots (x : ℝ) : 
  x^2 - 15*x + 36 = 0 → 
  ∃ s₁ s₂ : ℝ, s₁ ≠ s₂ ∧ x^2 - 15*x + 36 = (x - s₁) * (x - s₂) ∧ 
  (1 / s₁ + 1 / s₂ = 5 / 12) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l1224_122473


namespace NUMINAMATH_CALUDE_reflection_theorem_l1224_122421

/-- Represents a line in 2D space --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Represents a point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on a given line --/
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Checks if two points are symmetric with respect to the line x + y = 0 --/
def symmetricPoints (p1 p2 : Point) : Prop :=
  p2.x = -p1.y ∧ p2.y = -p1.x

/-- The main theorem to prove --/
theorem reflection_theorem :
  ∀ (b : ℝ),
  let incident_ray : Line := { slope := -3, intercept := b }
  let reflected_ray : Line := { slope := -1/3, intercept := 2 }
  let incident_point : Point := { x := 1, y := b - 3 }
  let reflected_point : Point := { x := -b + 3, y := -1 }
  pointOnLine incident_point incident_ray ∧
  pointOnLine reflected_point reflected_ray ∧
  symmetricPoints incident_point reflected_point →
  b = -6 := by
sorry

end NUMINAMATH_CALUDE_reflection_theorem_l1224_122421


namespace NUMINAMATH_CALUDE_mixed_fraction_product_l1224_122439

theorem mixed_fraction_product (X Y : ℕ) : 
  (5 + 1 / X) * (Y + 1 / 2) = 43 → X = 17 ∧ Y = 8 :=
sorry

end NUMINAMATH_CALUDE_mixed_fraction_product_l1224_122439


namespace NUMINAMATH_CALUDE_point_distance_product_l1224_122472

theorem point_distance_product (y₁ y₂ : ℝ) : 
  ((-4 - 3)^2 + (y₁ - (-1))^2 = 13^2) →
  ((-4 - 3)^2 + (y₂ - (-1))^2 = 13^2) →
  y₁ ≠ y₂ →
  y₁ * y₂ = -119 := by
sorry

end NUMINAMATH_CALUDE_point_distance_product_l1224_122472


namespace NUMINAMATH_CALUDE_james_has_more_balloons_l1224_122400

/-- The number of balloons James has -/
def james_balloons : ℕ := 232

/-- The number of balloons Amy has -/
def amy_balloons : ℕ := 101

/-- The difference in the number of balloons between James and Amy -/
def balloon_difference : ℕ := james_balloons - amy_balloons

theorem james_has_more_balloons : balloon_difference = 131 := by
  sorry

end NUMINAMATH_CALUDE_james_has_more_balloons_l1224_122400


namespace NUMINAMATH_CALUDE_age_comparison_l1224_122456

theorem age_comparison (present_age : ℕ) (years_ago : ℕ) : 
  (present_age = 50) →
  (present_age = (125 * (present_age - years_ago)) / 100) →
  (present_age = (250 * present_age) / (250 + 50)) →
  (years_ago = 10) := by
sorry


end NUMINAMATH_CALUDE_age_comparison_l1224_122456


namespace NUMINAMATH_CALUDE_triangle_perpendicular_bisector_distance_l1224_122444

/-- Given a triangle ABC with sides a, b, c where b > c, if a line HK perpendicular to BC
    divides the triangle into two equal areas, then the distance CK (from C to the foot of
    the perpendicular) is equal to (1/2) * sqrt(a^2 + b^2 - c^2). -/
theorem triangle_perpendicular_bisector_distance
  (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_order : b > c) (h_equal_areas : ∃ (k : ℝ), k > 0 ∧ k < b ∧ k * (a * k / (2 * b)) = a * b / 4) :
  ∃ (k : ℝ), k = (1/2) * Real.sqrt (a^2 + b^2 - c^2) ∧
              k > 0 ∧ k < b ∧ k * (a * k / (2 * b)) = a * b / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perpendicular_bisector_distance_l1224_122444


namespace NUMINAMATH_CALUDE_equation_solution_l1224_122446

noncomputable def f (x a : ℝ) : ℝ := Real.sqrt ((x + 2)^2 + 4 * a^2 - 4) + Real.sqrt ((x - 2)^2 + 4 * a^2 - 4)

theorem equation_solution (a b : ℝ) (h : b ≥ 0) :
  (∀ x, f x a = 4 * b → (b ∈ Set.Icc 0 1 ∪ Set.Ioi 1 → x = 0) ∧
                        (b = 1 → x ∈ Set.Icc (-2) 2)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1224_122446


namespace NUMINAMATH_CALUDE_roof_difference_l1224_122457

theorem roof_difference (width : ℝ) (length : ℝ) (area : ℝ) : 
  width > 0 →
  length = 4 * width →
  area = 588 →
  length * width = area →
  length - width = 21 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_roof_difference_l1224_122457


namespace NUMINAMATH_CALUDE_weekly_earnings_proof_l1224_122462

/-- Calculates the total earnings for a repair shop given the number of repairs and their costs. -/
def total_earnings (phone_repairs laptop_repairs computer_repairs : ℕ) 
  (phone_cost laptop_cost computer_cost : ℕ) : ℕ :=
  phone_repairs * phone_cost + laptop_repairs * laptop_cost + computer_repairs * computer_cost

/-- Theorem: The total earnings for the week is $121 given the specified repairs and costs. -/
theorem weekly_earnings_proof :
  total_earnings 5 2 2 11 15 18 = 121 := by
  sorry

end NUMINAMATH_CALUDE_weekly_earnings_proof_l1224_122462


namespace NUMINAMATH_CALUDE_total_hours_spent_l1224_122418

def project_time : ℕ := 300
def research_time : ℕ := 45
def presentation_time : ℕ := 75

def total_minutes : ℕ := project_time + research_time + presentation_time

theorem total_hours_spent : (total_minutes : ℚ) / 60 = 7 := by
  sorry

end NUMINAMATH_CALUDE_total_hours_spent_l1224_122418


namespace NUMINAMATH_CALUDE_vertex_difference_hexagonal_pentagonal_prism_l1224_122452

/-- The number of vertices in a regular polygon. -/
def verticesInPolygon (sides : ℕ) : ℕ := sides

/-- The number of vertices in a prism with regular polygonal bases. -/
def verticesInPrism (baseSides : ℕ) : ℕ := 2 * (verticesInPolygon baseSides)

/-- The difference between the number of vertices of a hexagonal prism and a pentagonal prism. -/
theorem vertex_difference_hexagonal_pentagonal_prism : 
  verticesInPrism 6 - verticesInPrism 5 = 2 := by
  sorry


end NUMINAMATH_CALUDE_vertex_difference_hexagonal_pentagonal_prism_l1224_122452


namespace NUMINAMATH_CALUDE_inequality_proof_l1224_122412

theorem inequality_proof (a b c : ℕ) (h : c ≥ b) :
  a^b * (a + b)^c > c^b * a^c := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1224_122412


namespace NUMINAMATH_CALUDE_distribute_five_balls_three_boxes_l1224_122435

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes,
    with each box containing at least one ball. -/
def distribute_balls (n k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 150 ways to distribute 5 distinguishable balls into 3 indistinguishable boxes,
    with each box containing at least one ball. -/
theorem distribute_five_balls_three_boxes :
  distribute_balls 5 3 = 150 := by sorry

end NUMINAMATH_CALUDE_distribute_five_balls_three_boxes_l1224_122435


namespace NUMINAMATH_CALUDE_strawberry_price_difference_l1224_122448

/-- Proves that the difference in price per pint between the regular price and the sale price is $2 --/
theorem strawberry_price_difference
  (pints_sold : ℕ)
  (sale_revenue : ℚ)
  (revenue_difference : ℚ)
  (h1 : pints_sold = 54)
  (h2 : sale_revenue = 216)
  (h3 : revenue_difference = 108)
  : (sale_revenue + revenue_difference) / pints_sold - sale_revenue / pints_sold = 2 := by
  sorry

#check strawberry_price_difference

end NUMINAMATH_CALUDE_strawberry_price_difference_l1224_122448


namespace NUMINAMATH_CALUDE_green_light_probability_is_five_twelfths_l1224_122443

/-- Represents the duration of each light color in seconds -/
structure TrafficLightDuration where
  red : ℕ
  green : ℕ
  yellow : ℕ

/-- Represents the cycle time of the traffic light in seconds -/
def cycleDuration (d : TrafficLightDuration) : ℕ :=
  d.red + d.green + d.yellow

/-- The probability of seeing a green light -/
def greenLightProbability (d : TrafficLightDuration) : ℚ :=
  d.green / (cycleDuration d)

/-- Theorem stating the probability of seeing a green light
    given specific durations for each light color -/
theorem green_light_probability_is_five_twelfths :
  let d : TrafficLightDuration := { red := 30, green := 25, yellow := 5 }
  greenLightProbability d = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_green_light_probability_is_five_twelfths_l1224_122443


namespace NUMINAMATH_CALUDE_sum_nonnegative_implies_one_nonnegative_l1224_122467

theorem sum_nonnegative_implies_one_nonnegative (a b : ℝ) : 
  a + b ≥ 0 → (a ≥ 0 ∨ b ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_sum_nonnegative_implies_one_nonnegative_l1224_122467


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_l1224_122484

/-- The perimeter of a trapezoid JKLM with given coordinates is 36 units -/
theorem trapezoid_perimeter : 
  let J : ℝ × ℝ := (-2, -4)
  let K : ℝ × ℝ := (-2, 2)
  let L : ℝ × ℝ := (6, 8)
  let M : ℝ × ℝ := (6, -4)
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  dist J K + dist K L + dist L M + dist M J = 36 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_perimeter_l1224_122484


namespace NUMINAMATH_CALUDE_grade_distribution_l1224_122428

theorem grade_distribution (n₂ n₃ n₄ n₅ : ℕ) : 
  n₂ + n₃ + n₄ + n₅ = 25 →
  n₄ = n₃ + 4 →
  2 * n₂ + 3 * n₃ + 4 * n₄ + 5 * n₅ = 121 →
  n₂ = 0 := by
sorry

end NUMINAMATH_CALUDE_grade_distribution_l1224_122428


namespace NUMINAMATH_CALUDE_tenth_term_of_specific_geometric_sequence_l1224_122488

def geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r ^ (n - 1)

theorem tenth_term_of_specific_geometric_sequence :
  geometric_sequence 9 (1/3) 10 = 1/2187 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_specific_geometric_sequence_l1224_122488


namespace NUMINAMATH_CALUDE_ant_walk_probability_l1224_122420

/-- A point on a 2D lattice -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- The ant's walk on the lattice -/
def AntWalk :=
  { p : LatticePoint // |p.x| + |p.y| ≥ 2 }

/-- Probability measure on the ant's walk -/
noncomputable def ProbMeasure : Type := AntWalk → ℝ

/-- The starting point of the ant -/
def start : LatticePoint := ⟨1, 0⟩

/-- The target end point -/
def target : LatticePoint := ⟨1, 1⟩

/-- Adjacent points are those that differ by 1 in exactly one coordinate -/
def adjacent (p q : LatticePoint) : Prop :=
  (abs (p.x - q.x) + abs (p.y - q.y) = 1)

/-- The probability measure satisfies the uniform distribution on adjacent points -/
axiom uniform_distribution (μ : ProbMeasure) (p : LatticePoint) :
  ∀ q : LatticePoint, adjacent p q → μ ⟨q, sorry⟩ = (1 : ℝ) / 4

/-- The main theorem: probability of reaching (1,1) from (1,0) is 7/24 -/
theorem ant_walk_probability (μ : ProbMeasure) :
  μ ⟨target, sorry⟩ = 7 / 24 := by sorry

end NUMINAMATH_CALUDE_ant_walk_probability_l1224_122420


namespace NUMINAMATH_CALUDE_angle_double_quadrant_l1224_122470

/-- Given that α is an angle in the second quadrant, prove that 2α is an angle in the third or fourth quadrant. -/
theorem angle_double_quadrant (α : Real) (h : π/2 < α ∧ α < π) :
  π < 2*α ∧ 2*α < 2*π :=
by sorry

end NUMINAMATH_CALUDE_angle_double_quadrant_l1224_122470


namespace NUMINAMATH_CALUDE_class_mean_score_l1224_122490

theorem class_mean_score (total_students : ℕ) (first_day_students : ℕ) (second_day_students : ℕ)
  (first_day_mean : ℚ) (second_day_mean : ℚ) :
  total_students = first_day_students + second_day_students →
  first_day_students = 54 →
  second_day_students = 6 →
  first_day_mean = 76 / 100 →
  second_day_mean = 82 / 100 →
  let new_class_mean := (first_day_students * first_day_mean + second_day_students * second_day_mean) / total_students
  new_class_mean = 766 / 1000 :=
by sorry

end NUMINAMATH_CALUDE_class_mean_score_l1224_122490


namespace NUMINAMATH_CALUDE_sqrt_integer_part_problem_l1224_122403

theorem sqrt_integer_part_problem :
  ∃ n : ℕ, 
    (∀ k : ℕ, k < 35 → ⌊Real.sqrt (n^2 + k)⌋ = n) ∧ 
    (∀ m : ℕ, m > n → ∃ j : ℕ, j < 35 ∧ ⌊Real.sqrt (m^2 + j)⌋ ≠ m) :=
sorry

end NUMINAMATH_CALUDE_sqrt_integer_part_problem_l1224_122403


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l1224_122460

theorem square_plus_reciprocal_square (x : ℝ) (h : x + 1/x = 4) : x^2 + 1/x^2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l1224_122460


namespace NUMINAMATH_CALUDE_f_of_f_of_3_l1224_122423

def f (x : ℝ) : ℝ := 3 * x^2 + x - 4

theorem f_of_f_of_3 : f (f 3) = 2050 := by
  sorry

end NUMINAMATH_CALUDE_f_of_f_of_3_l1224_122423


namespace NUMINAMATH_CALUDE_base5_division_theorem_l1224_122466

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a number from base 10 to base 5 -/
def base10ToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
  aux n []

/-- Represents a number in base 5 -/
structure Base5 where
  digits : List Nat
  valid : ∀ d ∈ digits, d < 5

theorem base5_division_theorem (a b : Base5) :
  let a_base10 := base5ToBase10 a.digits
  let b_base10 := base5ToBase10 b.digits
  let quotient_base10 := a_base10 / b_base10
  let quotient_base5 := Base5.mk (base10ToBase5 quotient_base10) sorry
  a = Base5.mk [1, 3, 2, 4] sorry ∧ 
  b = Base5.mk [1, 2] sorry → 
  quotient_base5 = Base5.mk [1, 1, 0] sorry := by
  sorry

end NUMINAMATH_CALUDE_base5_division_theorem_l1224_122466


namespace NUMINAMATH_CALUDE_coefficient_x3y7_value_l1224_122434

/-- The coefficient of x^3 * y^7 in the expansion of (x + 1/x - y)^10 -/
def coefficient_x3y7 : ℤ :=
  let n : ℕ := 10
  let k : ℕ := 7
  let m : ℕ := 3
  (-1)^k * (n.choose k) * (m.choose 0)

theorem coefficient_x3y7_value : coefficient_x3y7 = -120 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x3y7_value_l1224_122434


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l1224_122429

theorem smallest_sum_of_reciprocals (x y : ℕ+) : 
  x ≠ y → (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 24 → 
  ∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 24 → 
  (x : ℤ) + y ≤ (a : ℤ) + b → (x : ℤ) + y = 100 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l1224_122429


namespace NUMINAMATH_CALUDE_fourth_roll_prob_is_five_sixths_l1224_122471

-- Define the types of dice
inductive DieType
| Fair
| BiasedSix
| BiasedOne

-- Define the probability of rolling a six for each die type
def probSix (d : DieType) : ℚ :=
  match d with
  | DieType.Fair => 1/6
  | DieType.BiasedSix => 1/2
  | DieType.BiasedOne => 1/10

-- Define the probability of selecting each die
def probSelectDie (d : DieType) : ℚ := 1/3

-- Define the probability of rolling three sixes in a row for a given die
def probThreeSixes (d : DieType) : ℚ := (probSix d) ^ 3

-- Define the total probability of rolling three sixes
def totalProbThreeSixes : ℚ :=
  (probSelectDie DieType.Fair) * (probThreeSixes DieType.Fair) +
  (probSelectDie DieType.BiasedSix) * (probThreeSixes DieType.BiasedSix) +
  (probSelectDie DieType.BiasedOne) * (probThreeSixes DieType.BiasedOne)

-- Define the updated probability of having used each die type given three sixes were rolled
def updatedProbDie (d : DieType) : ℚ :=
  (probSelectDie d) * (probThreeSixes d) / totalProbThreeSixes

-- The main theorem
theorem fourth_roll_prob_is_five_sixths :
  (updatedProbDie DieType.Fair) * (probSix DieType.Fair) +
  (updatedProbDie DieType.BiasedSix) * (probSix DieType.BiasedSix) +
  (updatedProbDie DieType.BiasedOne) * (probSix DieType.BiasedOne) = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_fourth_roll_prob_is_five_sixths_l1224_122471


namespace NUMINAMATH_CALUDE_remainder_zero_l1224_122487

theorem remainder_zero (k α : ℕ+) (h : 10 * k.val - α.val > 0) :
  (8^(10 * k.val + α.val) + 6^(10 * k.val - α.val) - 
   7^(10 * k.val - α.val) - 2^(10 * k.val + α.val)) % 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_zero_l1224_122487


namespace NUMINAMATH_CALUDE_farm_size_l1224_122482

/-- Represents a farm with sunflowers and flax -/
structure Farm where
  flax : ℕ
  sunflowers : ℕ

/-- The total size of the farm in acres -/
def Farm.total_size (f : Farm) : ℕ := f.flax + f.sunflowers

/-- Theorem: Given the conditions, the farm's total size is 240 acres -/
theorem farm_size (f : Farm) 
  (h1 : f.sunflowers = f.flax + 80)  -- 80 more acres of sunflowers than flax
  (h2 : f.flax = 80)                 -- 80 acres of flax
  : f.total_size = 240 := by
  sorry

end NUMINAMATH_CALUDE_farm_size_l1224_122482


namespace NUMINAMATH_CALUDE_cylinder_band_length_l1224_122437

theorem cylinder_band_length (m k n : ℕ) : 
  (m > 0) → (k > 0) → (n > 0) → 
  (∀ (p : ℕ), Prime p → ¬(p^2 ∣ k)) →
  (2 * (24 * Real.sqrt 3 + 28 * Real.pi) = m * Real.sqrt k + n * Real.pi) →
  m + k + n = 107 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_band_length_l1224_122437


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l1224_122440

-- 1. Prove that 522 - 112 ÷ 4 = 494
theorem problem_1 : 522 - 112 / 4 = 494 := by
  sorry

-- 2. Prove that (603 - 587) × 80 = 1280
theorem problem_2 : (603 - 587) * 80 = 1280 := by
  sorry

-- 3. Prove that 26 × 18 + 463 = 931
theorem problem_3 : 26 * 18 + 463 = 931 := by
  sorry

-- 4. Prove that 400 × (45 ÷ 9) = 2000
theorem problem_4 : 400 * (45 / 9) = 2000 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l1224_122440


namespace NUMINAMATH_CALUDE_max_both_writers_and_editors_l1224_122477

/-- Conference attendees --/
structure Conference where
  total : ℕ
  writers : ℕ
  editors : ℕ
  both : ℕ
  neither : ℕ

/-- Conference conditions --/
def ConferenceConditions (c : Conference) : Prop :=
  c.total = 100 ∧
  c.writers = 40 ∧
  c.editors > 38 ∧
  c.neither = 2 * c.both ∧
  c.writers + c.editors - c.both + c.neither = c.total

/-- Theorem: The maximum number of people who are both writers and editors is 21 --/
theorem max_both_writers_and_editors (c : Conference) 
  (h : ConferenceConditions c) : c.both ≤ 21 := by
  sorry

end NUMINAMATH_CALUDE_max_both_writers_and_editors_l1224_122477


namespace NUMINAMATH_CALUDE_derivative_value_implies_coefficient_l1224_122414

theorem derivative_value_implies_coefficient (f' : ℝ → ℝ) (a : ℝ) :
  (∀ x, f' x = 2 * x^3 + a * x^2 + x) →
  f' 1 = 9 →
  a = 6 := by
sorry

end NUMINAMATH_CALUDE_derivative_value_implies_coefficient_l1224_122414


namespace NUMINAMATH_CALUDE_subtraction_of_reciprocals_l1224_122402

theorem subtraction_of_reciprocals (p q : ℝ) : 
  3 / p = 6 → 3 / q = 15 → p - q = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_reciprocals_l1224_122402


namespace NUMINAMATH_CALUDE_julia_tuesday_kids_l1224_122495

/-- The number of kids Julia played with on Monday -/
def monday_kids : ℕ := 16

/-- The difference between the number of kids Julia played with on Monday and Tuesday -/
def difference : ℕ := 12

/-- The number of kids Julia played with on Tuesday -/
def tuesday_kids : ℕ := monday_kids - difference

theorem julia_tuesday_kids : tuesday_kids = 4 := by
  sorry

end NUMINAMATH_CALUDE_julia_tuesday_kids_l1224_122495


namespace NUMINAMATH_CALUDE_factorization_equality_l1224_122401

theorem factorization_equality (x : ℝ) :
  (x^4 - 4*x^2 + 1) * (x^4 + 3*x^2 + 1) + 10*x^4 = 
  (x + 1)^2 * (x - 1)^2 * (x^2 + x + 1) * (x^2 - x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1224_122401


namespace NUMINAMATH_CALUDE_function_monotonicity_l1224_122442

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def periodic_two (f : ℝ → ℝ) : Prop := ∀ x, f x = f (2 - x)

def decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem function_monotonicity (f : ℝ → ℝ) 
  (h1 : is_even f)
  (h2 : periodic_two f)
  (h3 : decreasing_on f 1 2) :
  increasing_on f (-2) (-1) ∧ decreasing_on f 3 4 := by
  sorry

end NUMINAMATH_CALUDE_function_monotonicity_l1224_122442


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1224_122458

/-- Triangle ABC with sides a, b, and c satisfying |a-3| + (b-4)^2 = 0 -/
structure TriangleABC where
  a : ℝ
  b : ℝ
  c : ℝ
  h : |a - 3| + (b - 4)^2 = 0

/-- The perimeter of an isosceles triangle -/
def isoscelesPerimeter (t : TriangleABC) : Set ℝ :=
  {10, 11}

theorem triangle_abc_properties (t : TriangleABC) :
  t.a = 3 ∧ t.b = 4 ∧ 1 < t.c ∧ t.c < 7 ∧
  (t.a = t.b ∨ t.a = t.c ∨ t.b = t.c → t.a + t.b + t.c ∈ isoscelesPerimeter t) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l1224_122458


namespace NUMINAMATH_CALUDE_song_time_is_125_minutes_l1224_122438

/-- Represents the duration of a radio show in minutes -/
def total_show_time : ℕ := 3 * 60

/-- Represents the duration of a single talking segment in minutes -/
def talking_segment_duration : ℕ := 10

/-- Represents the duration of a single ad break in minutes -/
def ad_break_duration : ℕ := 5

/-- Represents the number of talking segments in the show -/
def num_talking_segments : ℕ := 3

/-- Represents the number of ad breaks in the show -/
def num_ad_breaks : ℕ := 5

/-- Calculates the total time spent on talking segments -/
def total_talking_time : ℕ := talking_segment_duration * num_talking_segments

/-- Calculates the total time spent on ad breaks -/
def total_ad_time : ℕ := ad_break_duration * num_ad_breaks

/-- Calculates the total time spent on non-song content -/
def total_non_song_time : ℕ := total_talking_time + total_ad_time

/-- Theorem: The remaining time for songs in the radio show is 125 minutes -/
theorem song_time_is_125_minutes : 
  total_show_time - total_non_song_time = 125 := by sorry

end NUMINAMATH_CALUDE_song_time_is_125_minutes_l1224_122438


namespace NUMINAMATH_CALUDE_union_when_m_3_union_equals_A_iff_l1224_122480

def A : Set ℝ := {x | x^2 - x - 12 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

theorem union_when_m_3 : A ∪ B 3 = {x | -3 ≤ x ∧ x ≤ 5} := by sorry

theorem union_equals_A_iff (m : ℝ) : A ∪ B m = A ↔ m ≤ 5/2 := by sorry

end NUMINAMATH_CALUDE_union_when_m_3_union_equals_A_iff_l1224_122480


namespace NUMINAMATH_CALUDE_perpendicular_planes_through_perpendicular_line_non_perpendicular_line_in_perpendicular_planes_l1224_122499

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relationships
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Plane → Plane → Prop)
variable (passes_through : Plane → Line → Prop)
variable (perpendicular_line : Line → Plane → Prop)
variable (in_plane : Line → Plane → Prop)
variable (line_of_intersection : Plane → Plane → Line)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorems
theorem perpendicular_planes_through_perpendicular_line 
  (P Q : Plane) (l : Line) :
  passes_through Q l → perpendicular_line l P → perpendicular P Q := by sorry

theorem non_perpendicular_line_in_perpendicular_planes 
  (P Q : Plane) (l : Line) :
  perpendicular P Q → 
  in_plane l P → 
  ¬ perpendicular_lines l (line_of_intersection P Q) → 
  ¬ perpendicular_line l Q := by sorry

end NUMINAMATH_CALUDE_perpendicular_planes_through_perpendicular_line_non_perpendicular_line_in_perpendicular_planes_l1224_122499


namespace NUMINAMATH_CALUDE_lcm_factor_proof_l1224_122417

def is_hcf (a b h : ℕ) : Prop := Nat.gcd a b = h

def is_lcm (a b l : ℕ) : Prop := Nat.lcm a b = l

theorem lcm_factor_proof (A B : ℕ) 
  (h1 : is_hcf A B 23)
  (h2 : A = 322)
  (h3 : ∃ x : ℕ, is_lcm A B (23 * 13 * x)) :
  ∃ x : ℕ, is_lcm A B (23 * 13 * x) ∧ x = 14 := by
  sorry

end NUMINAMATH_CALUDE_lcm_factor_proof_l1224_122417


namespace NUMINAMATH_CALUDE_digital_earth_technologies_l1224_122459

-- Define the set of all possible technologies
def AllTechnologies : Set String :=
  {"Sustainable development", "Global positioning technology", "Geographic information system",
   "Global positioning system", "Virtual technology", "Network technology"}

-- Define the digital Earth as a complex computer technology system
structure DigitalEarth where
  technologies : Set String
  complex : Bool
  integrates_various_tech : Bool

-- Define the supporting technologies for the digital Earth
def SupportingTechnologies (de : DigitalEarth) : Set String := de.technologies

-- Theorem statement
theorem digital_earth_technologies (de : DigitalEarth) 
  (h1 : de.complex = true) 
  (h2 : de.integrates_various_tech = true) : 
  SupportingTechnologies de = AllTechnologies := by
  sorry

end NUMINAMATH_CALUDE_digital_earth_technologies_l1224_122459


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_range_l1224_122431

/-- The function f(x) = x^2 + ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 1

/-- The function g(x) = e^x -/
noncomputable def g (x : ℝ) : ℝ := Real.exp x

/-- The theorem statement -/
theorem function_inequality_implies_a_range :
  ∀ a : ℝ, 
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 2 → x₂ ∈ Set.Icc 0 2 → x₁ ≠ x₂ → 
    |f a x₁ - f a x₂| < |g x₁ - g x₂|) →
  a ∈ Set.Icc (-1) (2 - 2 * Real.log 2) :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_range_l1224_122431


namespace NUMINAMATH_CALUDE_visibility_time_proof_l1224_122424

/-- Alice's walking speed in feet per second -/
def alice_speed : ℝ := 2

/-- Bob's walking speed in feet per second -/
def bob_speed : ℝ := 4

/-- Distance between Alice and Bob's parallel paths in feet -/
def path_distance : ℝ := 300

/-- Diameter of the circular monument in feet -/
def monument_diameter : ℝ := 150

/-- Initial distance between Alice and Bob when the monument first blocks their line of sight -/
def initial_distance : ℝ := 300

/-- Time until Alice and Bob can see each other again -/
def visibility_time : ℝ := 48

theorem visibility_time_proof :
  alice_speed = 2 ∧
  bob_speed = 4 ∧
  path_distance = 300 ∧
  monument_diameter = 150 ∧
  initial_distance = 300 →
  visibility_time = 48 := by
  sorry

#check visibility_time_proof

end NUMINAMATH_CALUDE_visibility_time_proof_l1224_122424


namespace NUMINAMATH_CALUDE_second_integer_value_l1224_122405

theorem second_integer_value (n : ℝ) : 
  (n + (n + 3) = 150) → (n + 1 = 74.5) := by
  sorry

end NUMINAMATH_CALUDE_second_integer_value_l1224_122405


namespace NUMINAMATH_CALUDE_parabola_properties_l1224_122407

/-- Properties of a parabola y = ax^2 + bx + c with a > 0, b > 0, and c < 0 -/
theorem parabola_properties (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c < 0) :
  let f := fun x => a * x^2 + b * x + c
  let vertex_x := -b / (2 * a)
  (∀ x y : ℝ, f x < f y → x < y ∨ y < x) ∧  -- Opens upwards
  vertex_x < 0 ∧                            -- Vertex in left half-plane
  f 0 < 0                                   -- Y-intercept below origin
:= by sorry

end NUMINAMATH_CALUDE_parabola_properties_l1224_122407


namespace NUMINAMATH_CALUDE_sin_fourteen_pi_fifths_l1224_122427

theorem sin_fourteen_pi_fifths : 
  Real.sin (14 * π / 5) = (Real.sqrt (10 - 2 * Real.sqrt 5)) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_fourteen_pi_fifths_l1224_122427
