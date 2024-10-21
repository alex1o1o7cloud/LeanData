import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_first_quadrant_l478_47831

noncomputable def complex_number : ℂ := (-1 + Complex.I) / Complex.I

theorem point_in_first_quadrant : 
  complex_number.re > 0 ∧ complex_number.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_first_quadrant_l478_47831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizzeria_problem_l478_47843

def pizzeria_dietary_restrictions (total_dishes : ℕ) 
  (vegan_ratio : ℚ) (gluten_ratio : ℚ) (dairy_dishes : ℕ) : ℕ :=
  let vegan_dishes := (total_dishes : ℚ) * vegan_ratio
  let gluten_dishes := vegan_dishes * gluten_ratio
  let non_gluten_dishes := vegan_dishes - gluten_dishes
  let non_dairy_dishes := vegan_dishes - (dairy_dishes : ℚ)
  (Int.floor (min non_gluten_dishes non_dairy_dishes)).toNat

theorem pizzeria_problem : 
  pizzeria_dietary_restrictions 30 (1/6) (1/2) 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizzeria_problem_l478_47843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_inverse_values_l478_47897

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 2 then 3 * x + 1 else x^2

noncomputable def f_inverse (y : ℝ) : ℝ :=
  if y ≤ 7 then (y - 1) / 3 else Real.sqrt y

theorem sum_of_inverse_values (hf : Function.Bijective f) :
  f_inverse (-5) + f_inverse 0 + f_inverse 1 + f_inverse 2 + f_inverse 3 +
  f_inverse 4 + f_inverse 5 + f_inverse 6 + f_inverse 7 + f_inverse 8 + f_inverse 9 =
  22 + 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_inverse_values_l478_47897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_1983_equals_fib_1983_minus_1_l478_47892

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Polynomial of degree 990 matching Fibonacci sequence for specific range -/
def p : ℕ → ℕ := sorry

/-- Assumption that p matches Fibonacci sequence for k = 992 to 1982 -/
axiom p_matches_fib : ∀ k, 992 ≤ k → k ≤ 1982 → p k = fib k

/-- Theorem: p(1983) equals F_{1983} - 1 -/
theorem p_1983_equals_fib_1983_minus_1 : p 1983 = fib 1983 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_1983_equals_fib_1983_minus_1_l478_47892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_composites_l478_47871

/-- Definition of the sequence x_n -/
def x (a b n : ℕ) : ℕ := a * (10^n - 1) / 9 + b

/-- Theorem statement -/
theorem infinitely_many_composites (a b : ℕ) (ha : 1 ≤ a ∧ a ≤ 9) (hb : 1 ≤ b ∧ b ≤ 9) :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, ¬(Nat.Prime (x a b n)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_composites_l478_47871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_fluctuation_l478_47836

theorem salary_fluctuation (initial_salary : ℝ) (initial_salary_pos : initial_salary > 0) :
  let step1 := initial_salary * 0.4
  let step2 := step1 * 1.6
  let step3 := step2 * 0.55
  let step4 := step3 * 1.2
  let final_salary := step4 * 0.75
  (final_salary - initial_salary) / initial_salary = -0.6832 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_fluctuation_l478_47836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l478_47886

noncomputable section

-- Define a quadratic function
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the vertex of a quadratic function
def vertex (a b c : ℝ) : ℝ × ℝ := (-b / (2 * a), (4 * a * c - b^2) / (4 * a))

-- Define the condition for the vertex being on the positive y-axis
def vertex_on_positive_y_axis (a b c : ℝ) : Prop :=
  (vertex a b c).1 = 0 ∧ (vertex a b c).2 > 0

-- Define the condition for the function being increasing to the left of the axis of symmetry
def increasing_left_of_axis (a : ℝ) : Prop := a < 0

theorem quadratic_function_properties (a b c : ℝ) :
  vertex_on_positive_y_axis a b c ∧ increasing_left_of_axis a →
  a < 0 ∧ b = 0 ∧ c > 0 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l478_47886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_distribution_theorem_l478_47824

def distribute_candy (n : ℕ) (k : ℕ) : ℕ :=
  Finset.sum (Finset.range (n - k + 1)) (λ r => 
    Finset.sum (Finset.range (n - r - k + 2)) (λ b => 
      Nat.choose n r * Nat.choose (n - r) b))

theorem candy_distribution_theorem :
  distribute_candy 8 3 = 2187 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_distribution_theorem_l478_47824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l478_47862

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + 2*x + a) / x

-- State the theorem
theorem f_properties :
  -- Part 1: Minimum value when a = 1/2
  (∀ x : ℝ, x ≥ 1 → f (1/2) x ≥ 7/2) ∧
  (∃ x : ℝ, x ≥ 1 ∧ f (1/2) x = 7/2) ∧
  -- Part 2: Condition for f(x) > 0
  (∀ a : ℝ, (∀ x : ℝ, x ≥ 1 → f a x > 0) ↔ -3 < a ∧ a ≤ 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l478_47862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_2_in_Q_l478_47859

-- Define the set of all cards
def allCards : Finset Nat := Finset.range 8

-- Define the properties of the boxes
structure Box where
  cards : Finset Nat
  sum_eq_18 : cards.sum id = 18

-- Define the problem setup
structure Setup where
  P : Box
  Q : Box
  P_size_3 : P.cards.card = 3
  Q_size_5 : Q.cards.card = 5
  all_cards_used : P.cards ∪ Q.cards = allCards
  cards_disjoint : Disjoint P.cards Q.cards

-- The theorem to prove
theorem card_2_in_Q (s : Setup) : 2 ∈ s.Q.cards := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_2_in_Q_l478_47859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_zero_in_interval_l478_47841

-- Define the function f(x) = cos(log(x) + x)
noncomputable def f (x : ℝ) : ℝ := Real.cos (Real.log x + x)

-- Define the interval (1, 2)
def openInterval : Set ℝ := {x | 1 < x ∧ x < 2}

-- Theorem statement
theorem one_zero_in_interval :
  ∃! x, x ∈ openInterval ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_zero_in_interval_l478_47841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_root_implies_a_geq_neg_one_l478_47812

-- Define the function f
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - (1 / Real.exp 2) * x + a

-- State the theorem
theorem f_root_implies_a_geq_neg_one :
  ∀ a : ℝ, (∃ x : ℝ, x > 0 ∧ f x a = 0) → a ≥ -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_root_implies_a_geq_neg_one_l478_47812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l478_47806

/-- Given a right triangle with one leg of length 15 and the angle opposite that leg being 45°,
    prove that the hypotenuse has length 15√2. -/
theorem right_triangle_hypotenuse (leg : ℝ) (angle : ℝ) (hypotenuse : ℝ) 
    (h1 : leg = 15)
    (h2 : angle = 45)
    (h3 : angle * Real.pi / 180 = Real.pi / 4)
    (h4 : hypotenuse ^ 2 = leg ^ 2 + leg ^ 2) : 
  hypotenuse = 15 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l478_47806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_speed_minimizes_cost_l478_47866

/-- The total transportation cost as a function of speed -/
noncomputable def total_cost (v : ℝ) : ℝ := 50000 / v + 5 * v

/-- The theorem stating that 100 km/h minimizes the total transportation cost -/
theorem optimal_speed_minimizes_cost :
  ∀ v : ℝ, v ∈ Set.Ioo 0 100 → total_cost 100 ≤ total_cost v := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_speed_minimizes_cost_l478_47866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bankers_gain_calculation_l478_47835

/-- Banker's gain calculation -/
theorem bankers_gain_calculation 
  (time : ℕ) 
  (rate : ℚ) 
  (bankers_discount : ℚ) : 
  time = 6 → 
  rate = 12 / 100 → 
  bankers_discount = 1806 → 
  ∃ (face_value : ℚ), 
    (bankers_discount - (bankers_discount * 100) / (100 + (rate * ↑time))) = 756 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bankers_gain_calculation_l478_47835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l478_47847

theorem inequality_solution_set (x : ℝ) : (4 : ℝ)^x - (2 : ℝ)^(x+2) > 0 ↔ x > 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l478_47847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_quad_pyramid_volume_l478_47810

/-- Regular quadrilateral pyramid with specific angle condition -/
structure RegularQuadPyramid where
  a : ℝ  -- Side length of the base
  h : ℝ  -- Height of the pyramid
  α : ℝ  -- Angle between a lateral edge and the plane of the base
  apex_angle_condition : α = 2 * Real.arcsin ((1 / Real.sqrt 2) * Real.cos α)

/-- Volume of a regular quadrilateral pyramid with the given conditions -/
noncomputable def volume (p : RegularQuadPyramid) : ℝ := 
  (1 / 3) * p.a^3 * Real.cos (p.α / 2)

/-- Theorem stating the volume of the pyramid -/
theorem regular_quad_pyramid_volume (p : RegularQuadPyramid) : 
  volume p = (p.a^3 / 6) * Real.sqrt (1 + Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_quad_pyramid_volume_l478_47810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_carrot_potato_ratio_l478_47889

/-- Proves that the ratio of carrots to potatoes is 6, given the conditions from the vegetable chopping problem. -/
def carrot_potato_ratio (carrots potatoes onions green_beans : ℕ) : Prop :=
  green_beans = 8 ∧
  3 * green_beans = onions ∧
  onions = 2 * carrots ∧
  potatoes = 2 ∧
  carrots / potatoes = 6

/-- The main theorem stating the ratio of carrots to potatoes -/
theorem main_carrot_potato_ratio : ∃ (c p o g : ℕ), carrot_potato_ratio c p o g := by
  -- We'll use the values from our solution
  let c := 12
  let p := 2
  let o := 24
  let g := 8
  
  -- Now we'll prove that these values satisfy the conditions
  have h1 : g = 8 := rfl
  have h2 : 3 * g = o := by norm_num
  have h3 : o = 2 * c := by norm_num
  have h4 : p = 2 := rfl
  have h5 : c / p = 6 := by norm_num
  
  -- Combine all conditions
  have h : carrot_potato_ratio c p o g := ⟨h1, h2, h3, h4, h5⟩
  
  -- Prove existence
  exact ⟨c, p, o, g, h⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_carrot_potato_ratio_l478_47889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_team_count_l478_47817

/-- Represents the number of teams in the West -/
def west_teams : ℕ := 6

/-- Represents the number of teams in the East -/
def east_teams : ℕ := 7

/-- The total number of teams does not exceed 30 -/
axiom total_teams_bound : west_teams + east_teams ≤ 30

/-- The East has 1 more team than the West -/
axiom east_west_diff : east_teams = west_teams + 1

/-- Represents the number of wins for a given region -/
def wins (region : Fin 3) : ℕ := sorry

/-- Represents the number of draws for a given region -/
def draws (region : Fin 3) : ℕ := sorry

/-- Ratio of wins to draws is 2:1 -/
axiom win_draw_ratio : ∀ region : Fin 3, 
  (wins region : ℚ) / (draws region : ℚ) = 2

/-- Represents the total points for a given number of teams -/
def total_points (teams : ℕ) : ℕ := sorry

/-- Total points of East teams is 34 more than West teams -/
axiom points_difference : 
  total_points east_teams - total_points west_teams = 34

/-- Theorem stating the correct number of teams -/
theorem correct_team_count : 
  west_teams = 6 ∧ east_teams = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_team_count_l478_47817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C_to_l_l478_47875

/-- Line l in the xy-plane -/
def line_l (x y : ℝ) : Prop := x + y + 4 = 0

/-- Curve C in the xy-plane -/
def curve_C (x y : ℝ) : Prop := y^2 = 4*x

/-- Distance from a point (x, y) to line l -/
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |x + y + 4| / Real.sqrt 2

/-- Theorem: The minimum distance from any point on curve C to line l is 3√2/2 -/
theorem min_distance_C_to_l : 
  ∃ (d : ℝ), d = 3 * Real.sqrt 2 / 2 ∧ 
  ∀ (x y : ℝ), curve_C x y → 
    distance_to_line x y ≥ d ∧
    ∃ (x₀ y₀ : ℝ), curve_C x₀ y₀ ∧ distance_to_line x₀ y₀ = d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C_to_l_l478_47875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_hexagon_l478_47814

-- Define the triangle and squares
noncomputable def triangle_ABC : Set (ℝ × ℝ) := sorry
noncomputable def square_ABDE : Set (ℝ × ℝ) := sorry
noncomputable def square_BCHI : Set (ℝ × ℝ) := sorry
noncomputable def square_CAFG : Set (ℝ × ℝ) := sorry

-- Define the properties of the triangle
def IsoscelesRight (t : Set (ℝ × ℝ)) : Prop := sorry
def LegLength (t : Set (ℝ × ℝ)) : ℝ := sorry

axiom isosceles_right : IsoscelesRight triangle_ABC
axiom leg_length : LegLength triangle_ABC = 2

-- Define the squares lying outside the triangle
def SquaresOutside (t s1 s2 s3 : Set (ℝ × ℝ)) : Prop := sorry
axiom squares_outside : SquaresOutside triangle_ABC square_ABDE square_BCHI square_CAFG

-- Define the hexagon
noncomputable def hexagon_DEFGHI : Set (ℝ × ℝ) := sorry

-- Define the area function
noncomputable def Area (s : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem to prove
theorem area_of_hexagon :
  Area hexagon_DEFGHI = 4 * Real.sqrt 2 - 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_hexagon_l478_47814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roof_difference_approx_l478_47865

/-- Represents the dimensions of a rectangular roof -/
structure RoofDimensions where
  width : ℝ
  length : ℝ
  area : ℝ

/-- Calculates the difference between length and width of a roof -/
def lengthWidthDifference (roof : RoofDimensions) : ℝ :=
  roof.length - roof.width

/-- Theorem stating the properties of the roof and the result to be proven -/
theorem roof_difference_approx (roof : RoofDimensions) 
  (h1 : roof.length = 5 * roof.width) 
  (h2 : roof.area = 576) : 
  ∃ ε > 0, |lengthWidthDifference roof - 42.92| < ε := by
  sorry

#check roof_difference_approx

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roof_difference_approx_l478_47865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_call_graph_edge_bound_l478_47815

/-- A graph with the properties described in the problem -/
structure CallGraph where
  /-- The number of vertices (people) in the graph -/
  n : ℕ
  /-- The set of edges (calls) in the graph -/
  edges : Finset (Fin n × Fin n)
  /-- No self-calls -/
  no_self_calls : ∀ v, (v, v) ∉ edges
  /-- At most one call between any two people -/
  at_most_one_call : ∀ u v, (u, v) ∈ edges → (v, u) ∉ edges
  /-- No K₃,₃ subgraph between any two disjoint sets of three vertices -/
  no_K33 : ∀ A B : Finset (Fin n), A.card = 3 → B.card = 3 → A ∩ B = ∅ →
    ∃ a ∈ A, ∃ b ∈ B, (a, b) ∉ edges ∧ (b, a) ∉ edges

/-- The main theorem to prove -/
theorem call_graph_edge_bound (G : CallGraph) (h : G.n = 2000) :
  G.edges.card < 201000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_call_graph_edge_bound_l478_47815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_exponential_inequality_l478_47873

theorem negation_of_exponential_inequality :
  (¬ ∀ x : ℝ, x > 0 → Real.exp x > x + 1) ↔ (∃ x : ℝ, x > 0 ∧ Real.exp x ≤ x + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_exponential_inequality_l478_47873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l478_47852

theorem remainder_problem (j : ℕ) (hj : j > 0) (h : 75 % (j^2) = 3) : 125 % j = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l478_47852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_equals_ten_times_difference_l478_47808

theorem product_equals_ten_times_difference (a b : ℕ) : 
  a * b = 10 * (max a b - min a b) ↔ 
  ((a = 90 ∧ b = 9) ∨ (a = 40 ∧ b = 8) ∨ (a = 15 ∧ b = 6) ∨ (a = 10 ∧ b = 5) ∨
   (a = 9 ∧ b = 90) ∨ (a = 8 ∧ b = 40) ∨ (a = 6 ∧ b = 15) ∨ (a = 5 ∧ b = 10)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_equals_ten_times_difference_l478_47808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l478_47839

-- Define the min function
noncomputable def min_fun (f g : ℝ → ℝ) : ℝ → ℝ := λ x => min (f x) (g x)

-- Define our specific function f
noncomputable def f : ℝ → ℝ := min_fun (λ x => 2 - x^2) (λ x => x)

-- State the theorem
theorem range_of_f :
  Set.range f = Set.Iic 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l478_47839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_triangle_area_for_hyperbola_hyperbola_equation_satisfied_l478_47833

/-- The area of the "golden triangle" for the hyperbola 2y^2 = 4 -/
noncomputable def golden_triangle_area : ℝ := 2 * Real.sqrt 2 - 2

/-- The hyperbola equation -/
def hyperbola_equation (y : ℝ) : Prop := 2 * y^2 = 4

/-- Theorem: The area of the "golden triangle" for the hyperbola 2y^2 = 4 is 2√2 - 2 -/
theorem golden_triangle_area_for_hyperbola :
  golden_triangle_area = 2 * Real.sqrt 2 - 2 := by
  -- The proof is omitted
  sorry

/-- Theorem: The hyperbola equation is satisfied for y = ±1 -/
theorem hyperbola_equation_satisfied :
  hyperbola_equation 1 ∧ hyperbola_equation (-1) := by
  -- The proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_triangle_area_for_hyperbola_hyperbola_equation_satisfied_l478_47833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_five_l478_47883

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  h1 : ∀ n, a n > 0
  h2 : q > 1
  h3 : a 3 + a 5 = 20
  h4 : a 2 * a 6 = 64

/-- Sum of first n terms of a geometric sequence -/
noncomputable def geometricSum (seq : GeometricSequence) (n : ℕ) : ℝ :=
  (seq.a 1) * (1 - seq.q^n) / (1 - seq.q)

theorem geometric_sum_five (seq : GeometricSequence) :
  geometricSum seq 5 = 31 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_five_l478_47883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_is_geometric_not_arithmetic_l478_47802

def sequenceList : List ℕ := [3, 9, 27, 81, 243, 729]

def is_geometric (s : List ℕ) : Prop :=
  ∃ r : ℕ, r > 1 ∧ ∀ i, i < s.length - 1 → s[i + 1]! = s[i]! * r

def is_arithmetic (s : List ℕ) : Prop :=
  ∃ d : ℤ, ∀ i, i < s.length - 1 → s[i + 1]! = s[i]! + d

theorem sequence_is_geometric_not_arithmetic :
  is_geometric sequenceList ∧ ¬is_arithmetic sequenceList := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_is_geometric_not_arithmetic_l478_47802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_cube_properties_l478_47818

/-- A cube made of 27 dice -/
structure DiceCube where
  total_dice : Nat
  surface_faces : Nat
  center_prob : Rat
  edge_prob : Rat
  corner_prob : Rat

/-- Properties of the dice cube -/
def standard_cube : DiceCube where
  total_dice := 27
  surface_faces := 54
  center_prob := 1/6
  edge_prob := 1/3
  corner_prob := 1/2

/-- Theorems about the dice cube -/
theorem dice_cube_properties (c : DiceCube) (h : c = standard_cube) :
  /- Probability of exactly 25 sixes on the surface -/
  (31 : Rat) / (2^13 * 3^18) = (26 : Rat) * 5 / 6^26 ∧
  /- Probability of at least one one on the surface -/
  1 - (5^6 : Rat) / (2^2 * 3^18) = 1 - (5/6)^6 * (2/3)^12 * (1/2)^8 ∧
  /- Expected number of sixes facing outward -/
  (9 : Rat) = 6 * (1/6) + 12 * (1/3) + 8 * (1/2) ∧
  /- Expected sum of numbers on the surface -/
  (189 : Rat) = 54 * (7/2) ∧
  /- Expected number of different digits on the surface -/
  (6 : Rat) - (5^6 : Rat) / (2 * 3^17) = 6 * (1 - (5/6)^6 * (2/3)^12 * (1/2)^8) := by
  sorry

#check dice_cube_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_cube_properties_l478_47818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisible_by_11_eq_8_l478_47851

/-- Represents the number obtained by concatenating integers from 1 to n -/
def a (n : ℕ) : ℕ := sorry

/-- Determines if a number is divisible by 11 based on the alternating sum of its digits -/
def divisible_by_11 (n : ℕ) : Bool :=
  if n % 2 = 0
  then (n / 2) % 11 = 0
  else ((n + 1) / 2) % 11 = 0

/-- The count of numbers a_k divisible by 11 for 1 ≤ k ≤ 100 -/
def count_divisible_by_11 : ℕ :=
  (Finset.range 100).filter (λ k => divisible_by_11 (k + 1)) |>.card

theorem count_divisible_by_11_eq_8 :
  count_divisible_by_11 = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisible_by_11_eq_8_l478_47851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_52_04567_to_nearest_tenth_l478_47850

noncomputable def round_to_nearest_tenth (x : ℝ) : ℝ :=
  ⌊x * 10 + 0.5⌋ / 10

theorem round_52_04567_to_nearest_tenth :
  round_to_nearest_tenth 52.04567 = 52.0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_52_04567_to_nearest_tenth_l478_47850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_with_specific_rate_of_change_l478_47860

def f (x : ℝ) : ℝ := 2 * x^2 + 1

theorem point_with_specific_rate_of_change :
  ∃ (x₀ y₀ : ℝ), (deriv f x₀ = -8) ∧ 
  f x₀ = y₀ ∧ x₀ = -2 ∧ y₀ = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_with_specific_rate_of_change_l478_47860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_circle_l478_47881

-- Define the circle C₁
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 8*y + 15 = 0

-- Define the point Q
def Q : ℝ × ℝ := (-2, 1)

-- Define the distance between two points
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

-- Theorem statement
theorem min_distance_to_circle :
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 13 - 1 ∧
  ∀ (P : ℝ × ℝ), C₁ P.1 P.2 → distance P Q ≥ min_dist := by
  sorry

#check min_distance_to_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_circle_l478_47881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_division_l478_47825

/-- Represents a point on the 8x8 grid -/
structure GridPoint where
  x : Fin 8
  y : Fin 8

/-- Represents a line parallel to either x-axis or y-axis -/
inductive DividingLine
  | Horizontal (y : Fin 8)
  | Vertical (x : Fin 8)

/-- Theorem: For any two distinct points on an 8x8 grid, there exists a dividing line
    that separates the grid into two equal parts, each containing one point -/
theorem chessboard_division (p1 p2 : GridPoint) (h : p1 ≠ p2) :
  ∃ (l : DividingLine), (∃ y : Fin 8, l = DividingLine.Horizontal y) ∨ 
                        (∃ x : Fin 8, l = DividingLine.Vertical x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_division_l478_47825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increase_mid_priced_sales_l478_47819

/-- Represents a product with its quality and price --/
structure Product where
  quality : ℝ
  price : ℝ

/-- Represents a store shelf with three products --/
structure ShelfLayout where
  productA : Product
  productB : Product
  productC : Option Product

/-- Represents a customer's perception of value --/
noncomputable def perceivedValue (p : Product) : ℝ := p.quality / p.price

/-- Represents the likelihood of a customer purchasing a product --/
noncomputable def purchaseLikelihood (p : Product) (s : ShelfLayout) : ℝ :=
  sorry

/-- The main theorem stating that the described strategy increases sales of product B --/
theorem increase_mid_priced_sales (s : ShelfLayout) 
  (h1 : s.productA.quality > s.productB.quality)
  (h2 : s.productA.price > s.productB.price)
  (h3 : s.productB.price > (match s.productC with | some c => c.price | none => 0))
  (h4 : perceivedValue s.productB > perceivedValue s.productA) :
  purchaseLikelihood s.productB s > purchaseLikelihood s.productA s ∧ 
  (match s.productC with
   | some c => purchaseLikelihood s.productB s > purchaseLikelihood c s
   | none => True) :=
by
  sorry

#check increase_mid_priced_sales

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increase_mid_priced_sales_l478_47819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_CD_from_volume_l478_47853

/-- The volume of a cylinder with hemispheres at both ends -/
noncomputable def cylinderWithHemispheres (r : ℝ) (h : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3 + Real.pi * r^2 * h

/-- The theorem stating the length of CD given the volume of the region -/
theorem length_CD_from_volume (r h : ℝ) :
  r = 4 → cylinderWithHemispheres r h = 384 * Real.pi → h = 56 / 3 := by
  sorry

#check length_CD_from_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_CD_from_volume_l478_47853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_and_even_functions_sum_l478_47872

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (9^x - a) / 3^x
noncomputable def g (b : ℝ) (x : ℝ) : ℝ := Real.log (10^x + 1) / Real.log 10 + b * x

-- State the theorem
theorem symmetric_and_even_functions_sum (a b : ℝ) :
  (∀ x, f a x = -f a (-x)) →  -- f is symmetric about the origin
  (∀ x, g b x = g b (-x)) →   -- g is an even function
  a + b = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_and_even_functions_sum_l478_47872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_and_angle_l478_47858

-- Define the circle C
def circleC (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 5

-- Define the line L
def lineL (m x y : ℝ) : Prop := m * x - y + 1 - m = 0

-- Theorem statement
theorem circle_line_intersection_and_angle :
  ∀ m : ℝ, 
  (∃! A B : ℝ × ℝ, A ≠ B ∧ circleC A.1 A.2 ∧ circleC B.1 B.2 ∧ lineL m A.1 A.2 ∧ lineL m B.1 B.2) ∧
  (∀ A B : ℝ × ℝ, A ≠ B → circleC A.1 A.2 → circleC B.1 B.2 → lineL m A.1 A.2 → lineL m B.1 B.2 →
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 17 →
    (m = Real.sqrt 3 ∨ m = -Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_and_angle_l478_47858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_expression_l478_47845

theorem odd_expression (a b c : ℕ) (ha : Odd a) (hb : Odd b) : 
  Odd (3^a + (b - 1)^2 * c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_expression_l478_47845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_value_min_shift_value_is_pi_l478_47804

-- Define the functions f and g as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sin (1/2 * x)
noncomputable def g (x : ℝ) : ℝ := Real.cos (1/2 * x)

-- State the theorem
theorem min_shift_value (φ : ℝ) (h1 : φ > 0) (h2 : ∀ x, f (x + φ) = g x) :
  φ ≥ Real.pi := by
  sorry

-- Define a corollary to state that π is the minimum value
theorem min_shift_value_is_pi :
  ∃ φ > 0, (∀ x, f (x + φ) = g x) ∧ φ = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_value_min_shift_value_is_pi_l478_47804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grammar_check_l478_47890

def correct_answer : String := "to reject"

theorem grammar_check : correct_answer = "to reject" := by
  rfl

#eval correct_answer

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grammar_check_l478_47890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_and_properties_l478_47878

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -a * Real.log x

noncomputable def φ (a : ℝ) : ℝ := -a * Real.log a

theorem minimum_value_and_properties :
  ∀ a b : ℝ,
  (a > 0 → (∀ x > 0, f a x ≥ φ a)) ∧
  (a > 0 → φ a ≤ 1) ∧
  (a > 0 → b > 0 → (deriv φ a) ≤ (deriv φ b)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_and_properties_l478_47878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_value_l478_47894

-- Define the variables and constants
variable (x : ℝ)
variable (some_constant : ℝ)
variable (a k n : ℝ)

-- Define the given equations
def equation1 (x some_constant a k n : ℝ) : Prop :=
  (3*x + 2)*(2*x - some_constant) = a*x^2 + k*x + n

def equation2 (a k n : ℝ) : Prop :=
  a - n + k = 3

-- State the theorem
theorem constant_value :
  ∀ (x some_constant a k n : ℝ),
  equation1 x some_constant a k n →
  equation2 a k n →
  some_constant = 7/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_value_l478_47894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_graphs_sum_l478_47879

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (2 + a * x) / Real.log a

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.log (a + 2 * x) / Real.log (1 / a)

theorem symmetric_graphs_sum (a b : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x : ℝ, f a x + g a x = 2 * b) → a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_graphs_sum_l478_47879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_specific_l478_47867

/-- The area of a triangle given two sides and a median to the third side -/
noncomputable def triangle_area (a b m : ℝ) : ℝ :=
  let c := Real.sqrt (2 * (a^2 + b^2) - 4 * m^2)
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Theorem stating that the area of a triangle with sides 6 and 8, and a median of 5 to the third side, is 24 -/
theorem triangle_area_specific : triangle_area 6 8 5 = 24 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_specific_l478_47867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_is_minimal_l478_47834

/-- The smallest integer m ≥ 7 such that for any partition of {7, 8, ..., m} into two subsets, 
    at least one subset contains integers a, b, and c where ab = c. -/
def smallest_m : ℕ := 16807

/-- The set S = {7, 8, ..., m} -/
def S (m : ℕ) : Set ℕ := {x | 7 ≤ x ∧ x ≤ m}

/-- Predicate that checks if a subset contains a, b, c where ab = c -/
def has_product_relation (A : Set ℕ) : Prop :=
  ∃ a b c, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a * b = c

/-- Main theorem stating that smallest_m is the smallest value satisfying the condition -/
theorem smallest_m_is_minimal :
  (∀ m < smallest_m, ∃ A B : Set ℕ, A ∪ B = S m ∧ A ∩ B = ∅ ∧ 
    ¬has_product_relation A ∧ ¬has_product_relation B) ∧
  (∀ A B : Set ℕ, A ∪ B = S smallest_m → A ∩ B = ∅ → 
    has_product_relation A ∨ has_product_relation B) :=
by sorry

#check smallest_m_is_minimal

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_is_minimal_l478_47834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_zero_property_l478_47874

theorem complex_equation_zero_property :
  ∃ (z : ℂ),
    (z + 2*Complex.I) * (z + 4*Complex.I) * z = 502*Complex.I ∧
    (∃ (w : ℂ), w ≠ z ∧ (w + 2*Complex.I) * (w + 4*Complex.I) * w = 502*Complex.I ∧ w.re = 0) ∧
    z.re ≠ Real.sqrt 50 ∧
    z.re ≠ Real.sqrt 502 ∧
    z.re ≠ 2 * Real.sqrt 502 ∧
    z.re ≠ Real.sqrt 2510 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_zero_property_l478_47874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_black_triangles_l478_47820

/-- A valid coloring of a dissected convex n-gon. -/
structure ValidColoring (n : ℕ) where
  (n_ge_3 : n ≥ 3)
  (black_triangles : ℕ)
  (white_triangles : ℕ)
  (total_triangles : black_triangles + white_triangles = n - 2)
  (adjacent_different : ∀ t1 t2 : ℕ, t1 ≠ t2 → black_triangles + white_triangles > t1 ∧ black_triangles + white_triangles > t2)

/-- The least possible number of black triangles in any valid coloring of a dissected convex n-gon. -/
def least_black_triangles (n : ℕ) : ℕ := (n - 1) / 3

/-- The main theorem stating that the least possible number of black triangles in any valid coloring
    of a dissected convex n-gon is ⌊(n-1)/3⌋. -/
theorem min_black_triangles (n : ℕ) (h : n ≥ 3) :
  (∀ c : ValidColoring n, c.black_triangles ≥ least_black_triangles n) ∧
  (∃ c : ValidColoring n, c.black_triangles = least_black_triangles n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_black_triangles_l478_47820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_mile_equals_500_rods_l478_47801

-- Define the units
def mile : ℕ → ℕ := λ n => n

def furlong : ℕ → ℕ := λ n => n

def rod : ℕ → ℕ := λ n => n

-- Define the conversion rates
axiom mile_to_furlong : ∀ n : ℕ, mile n = furlong (10 * n)
axiom furlong_to_rod : ∀ n : ℕ, furlong n = rod (50 * n)

-- Theorem statement
theorem one_mile_equals_500_rods : mile 1 = rod 500 := by
  -- Apply the conversion from mile to furlong
  have h1 : mile 1 = furlong 10 := mile_to_furlong 1
  -- Apply the conversion from furlong to rod
  have h2 : furlong 10 = rod 500 := furlong_to_rod 10
  -- Combine the two steps
  rw [h1, h2]
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_mile_equals_500_rods_l478_47801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minute_hand_distance_45_min_l478_47857

/-- The distance traveled by the tip of a clock's minute hand -/
noncomputable def minute_hand_distance (length : ℝ) (minutes : ℝ) : ℝ :=
  2 * Real.pi * length * (minutes / 60)

/-- Theorem: The distance traveled by the tip of an 8 cm long minute hand in 45 minutes is 12π cm -/
theorem minute_hand_distance_45_min :
  minute_hand_distance 8 45 = 12 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minute_hand_distance_45_min_l478_47857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_change_over_five_years_l478_47888

/-- Calculates the net percent change in population over five years given specific yearly changes -/
theorem population_change_over_five_years :
  let year1_change : ℚ := 120 / 100  -- 20% increase
  let year2_change : ℚ := 90 / 100   -- 10% decrease
  let year3_change : ℚ := 115 / 100  -- 15% increase
  let year4_change : ℚ := 70 / 100   -- 30% decrease
  let year5_change : ℚ := 120 / 100  -- 20% increase
  let net_change : ℚ := (year1_change * year2_change * year3_change * year4_change * year5_change - 1) * 100
  ∃ ε : ℚ, ε > 0 ∧ |net_change - 422| < ε :=
by
  -- The proof goes here
  sorry

#eval (120 / 100 * 90 / 100 * 115 / 100 * 70 / 100 * 120 / 100 - 1) * 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_change_over_five_years_l478_47888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourSqrtTwoIs11thTerm_l478_47848

noncomputable def my_sequence (n : ℕ) : ℝ := Real.sqrt (3 * n - 1)

theorem fourSqrtTwoIs11thTerm : my_sequence 11 = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourSqrtTwoIs11thTerm_l478_47848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_is_existential_l478_47898

/-- A type representing triangles -/
def Triangle : Type := sorry

/-- A predicate representing the existence of a circumcircle for a triangle -/
def has_circumcircle : Triangle → Prop := sorry

/-- Theorem stating that the negation of "all triangles have a circumcircle" 
    is equivalent to "there exists a triangle without a circumcircle" -/
theorem negation_of_universal_is_existential :
  (¬ ∀ t : Triangle, has_circumcircle t) ↔ (∃ t : Triangle, ¬ has_circumcircle t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_is_existential_l478_47898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_president_wins_l478_47809

/-- A graph representing the cities and roads. -/
structure CityGraph where
  vertices : Finset ℕ
  edges : Finset (ℕ × ℕ)
  vertex_count : vertices.card = 2010
  degree_three : ∀ v, v ∈ vertices → (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = 3

/-- The game state, tracking colored edges. -/
structure GameState where
  graph : CityGraph
  colored_edges : Finset ((ℕ × ℕ) × ℕ)
  valid_coloring : ∀ e, e ∈ colored_edges → e.1 ∈ graph.edges ∧ e.2 < 3

/-- A player's strategy. -/
def Strategy := GameState → Option ((ℕ × ℕ) × ℕ)

/-- Checks if a player has won. -/
def winning_state (state : GameState) : Prop :=
  ∃ v ∈ state.graph.vertices,
    ∃ e₁ e₂ e₃, e₁ ∈ state.colored_edges ∧ e₂ ∈ state.colored_edges ∧ e₃ ∈ state.colored_edges ∧
      (e₁.1.1 = v ∨ e₁.1.2 = v) ∧
      (e₂.1.1 = v ∨ e₂.1.2 = v) ∧
      (e₃.1.1 = v ∨ e₃.1.2 = v) ∧
      e₁.2 ≠ e₂.2 ∧ e₂.2 ≠ e₃.2 ∧ e₃.2 ≠ e₁.2

/-- Function to compute the game state after n moves. -/
def nth_state (g : CityGraph) (opponent_strategy : Strategy) (strategy : Strategy) (n : ℕ) : GameState :=
  sorry

/-- The main theorem stating that the second player (President) has a winning strategy. -/
theorem president_wins (g : CityGraph) :
  ∃ (strategy : Strategy), ∀ (opponent_strategy : Strategy),
    ∃ (n : ℕ), winning_state (nth_state g opponent_strategy strategy n) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_president_wins_l478_47809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l478_47811

def U : Set ℤ := {x | 0 < x ∧ x < 9}
def S : Set ℤ := {1, 3, 5}
def T : Set ℤ := {3, 6}

theorem set_operations :
  (S ∩ T = {3}) ∧
  ((U \ (S ∪ T)) = {2, 4, 7, 8}) := by
  constructor
  · -- Proof for S ∩ T = {3}
    sorry
  · -- Proof for (U \ (S ∪ T)) = {2, 4, 7, 8}
    sorry

#check set_operations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l478_47811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_box_contains_hearts_or_spades_l478_47854

-- Define the cards and boxes
inductive Card : Type
| Hearts : Card
| Clubs : Card
| Diamonds : Card
| Spades : Card

def Box := Fin 4

-- Define the guesses made by each person
def XiaoMingGuess : Box → Card → Prop
| ⟨0, _⟩, Card.Clubs => True
| ⟨2, _⟩, Card.Diamonds => True
| _, _ => False

def XiaoHongGuess : Box → Card → Prop
| ⟨1, _⟩, Card.Clubs => True
| ⟨2, _⟩, Card.Spades => True
| _, _ => False

def XiaoZhangGuess : Box → Card → Prop
| ⟨3, _⟩, Card.Spades => True
| ⟨1, _⟩, Card.Diamonds => True
| _, _ => False

def XiaoLiGuess : Box → Card → Prop
| ⟨3, _⟩, Card.Hearts => True
| ⟨2, _⟩, Card.Diamonds => True
| _, _ => False

-- Define the condition that each person got half of their guesses right
def HalfRight (guess : Box → Card → Prop) (actual : Box → Card) : Prop :=
  (∃ b c, guess b c ∧ actual b = c) ∧
  (∃ b c, guess b c ∧ actual b ≠ c)

-- State the theorem
theorem fourth_box_contains_hearts_or_spades 
  (actual : Box → Card)
  (all_different : ∀ b1 b2, b1 ≠ b2 → actual b1 ≠ actual b2)
  (xiaoming_half_right : HalfRight XiaoMingGuess actual)
  (xiaohong_half_right : HalfRight XiaoHongGuess actual)
  (xiaozhang_half_right : HalfRight XiaoZhangGuess actual)
  (xiaoli_half_right : HalfRight XiaoLiGuess actual) :
  actual ⟨3, by norm_num⟩ = Card.Hearts ∨ actual ⟨3, by norm_num⟩ = Card.Spades :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_box_contains_hearts_or_spades_l478_47854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dataset_variance_l478_47869

noncomputable def dataset : List ℝ := [3, 6, 9, 8, 4]

noncomputable def mean (data : List ℝ) : ℝ :=
  (data.sum) / data.length

noncomputable def variance (data : List ℝ) : ℝ :=
  let μ := mean data
  (data.map (λ x => (x - μ)^2)).sum / data.length

theorem dataset_variance :
  variance dataset = 26 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dataset_variance_l478_47869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_waiter_tables_l478_47838

/-- Calculates the minimum number of tables required for restaurant seating --/
def minimum_tables (initial_customers : ℕ) (departed_customers : ℕ) 
  (small_table_capacity : ℕ) (large_table_capacity : ℕ) 
  (small_table_percentage : ℚ) : ℕ :=
  let remaining_customers := initial_customers - departed_customers
  let total_tables := (remaining_customers : ℚ) / 
    (small_table_percentage * small_table_capacity + 
     (1 - small_table_percentage) * large_table_capacity)
  Int.ceil total_tables |>.toNat

theorem waiter_tables : 
  minimum_tables 44 12 4 8 (1/4) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_waiter_tables_l478_47838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_AB_CD_l478_47823

noncomputable section

def A : ℝ × ℝ := (0, 1)  -- We assume the x-coordinate of A is 0
def B : ℝ × ℝ := (1, 2)
def C : ℝ × ℝ := (2, -1)
def D : ℝ × ℝ := (0, 4)  -- We assume the x-coordinate of D is 0

def vector_AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def vector_CD : ℝ × ℝ := (D.1 - C.1, D.2 - C.2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

noncomputable def projection (v w : ℝ × ℝ) : ℝ := 
  (dot_product v w) / (vector_magnitude w)

theorem projection_AB_CD : 
  projection vector_AB vector_CD = 3 * Real.sqrt 2 / 2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_AB_CD_l478_47823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_condition_l478_47828

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 + Real.log x / Real.log a

theorem inverse_function_condition (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a ((f a).invFun 3) = 3 ∧ (f a).invFun 3 = 4 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_condition_l478_47828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_sum_bounds_l478_47813

noncomputable section

open Real

theorem vector_magnitude_sum_bounds (a b : EuclideanSpace ℝ (Fin 3)) 
  (h1 : ‖a + b‖ = 5) (h2 : ‖a - b‖ = 5) : 
  5 ≤ ‖a‖ + ‖b‖ ∧ ‖a‖ + ‖b‖ ≤ 5 * sqrt 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_sum_bounds_l478_47813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l478_47870

theorem tan_alpha_plus_pi_fourth (α : Real) 
  (h1 : Real.cos α = -4/5) 
  (h2 : α ∈ Set.Ioo (π/2) π) : 
  Real.tan (α + π/4) = 1/7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l478_47870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_box_berries_l478_47826

/-- Represents the number of blueberries in each blue box -/
def B : ℤ := sorry

/-- Represents the number of strawberries in each red box -/
def S : ℤ := sorry

/-- Disposing of one blue box for one additional red box increases the total number of berries by 20 -/
axiom berry_increase : -B + S = 20

/-- The difference between the total number of strawberries and blueberries increases by 80 after the exchange -/
axiom difference_increase : S + B - (S - B) = 80

theorem blue_box_berries : B = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_box_berries_l478_47826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_coefficients_exists_min_parabola_l478_47895

/-- A parabola that intersects the x-axis at two distinct points within 1 unit of the origin -/
structure Parabola where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  intersects_x_axis : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0
  roots_near_origin : ∀ x : ℝ, a * x^2 + b * x + c = 0 → |x| < 1

/-- The minimum sum of coefficients for a parabola meeting the given conditions is 11 -/
theorem min_sum_of_coefficients (p : Parabola) : p.a + p.b + p.c ≥ 11 := by
  sorry

/-- There exists a parabola meeting the conditions with sum of coefficients equal to 11 -/
theorem exists_min_parabola : ∃ p : Parabola, p.a + p.b + p.c = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_coefficients_exists_min_parabola_l478_47895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_and_jill_meeting_point_l478_47829

noncomputable section

/-- The distance Jack and Jill run in total -/
def total_distance : ℝ := 8

/-- The distance to the top of the hill -/
def hill_top : ℝ := 4

/-- Jack's head start in hours -/
def head_start : ℝ := 1/4

/-- Jack's uphill speed in km/hr -/
def jack_uphill_speed : ℝ := 14

/-- Jack's downhill speed in km/hr -/
def jack_downhill_speed : ℝ := 18

/-- Jill's uphill speed in km/hr -/
def jill_uphill_speed : ℝ := 15

/-- Jill's downhill speed in km/hr -/
def jill_downhill_speed : ℝ := 21

/-- The theorem stating where Jack and Jill meet -/
theorem jack_and_jill_meeting_point : 
  ∃ t : ℝ, t > head_start ∧ 
  (if t ≤ hill_top / jack_uphill_speed + head_start
   then jack_uphill_speed * t
   else hill_top + jack_downhill_speed * (t - (hill_top / jack_uphill_speed + head_start))) =
  jill_uphill_speed * (t - head_start) ∧
  hill_top - jill_uphill_speed * (t - head_start) = 851 / 154 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_and_jill_meeting_point_l478_47829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_satellite_lifting_work_l478_47805

/-- Work done in lifting a satellite -/
noncomputable def work_done (m : ℝ) (g : ℝ) (R₃ : ℝ) (H : ℝ) : ℝ :=
  m * g * R₃^2 * (1/R₃ - 1/(R₃+H))

/-- Theorem stating the work done in lifting a satellite -/
theorem satellite_lifting_work 
  (m : ℝ) 
  (g : ℝ) 
  (R₃ : ℝ) 
  (H : ℝ) 
  (h_m_pos : m > 0) 
  (h_g_pos : g > 0) 
  (h_R₃_pos : R₃ > 0) 
  (h_H_pos : H > 0) :
  ∃ (A : ℝ), A = work_done m g R₃ H ∧ A > 0 := by
  sorry

#check satellite_lifting_work

end NUMINAMATH_CALUDE_ERRORFEEDBACK_satellite_lifting_work_l478_47805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_area_interval_l478_47816

noncomputable def τ (s : ℝ) : Set (ℝ × ℝ × ℝ) :=
  {t : ℝ × ℝ × ℝ | let (x, y, z) := t
    x > 0 ∧ y > 0 ∧ z > 0 ∧ 
    x + y > z ∧ x + z > y ∧ y + z > x ∧
    x^2 + y^2 < z^2 ∧
    ((x = 4 ∧ y = 10) ∨ (x = 4 ∧ z = 10) ∨ (y = 4 ∧ z = 10)) ∧
    1/2 * x * y * Real.sin (Real.arccos ((x^2 + y^2 - z^2) / (2 * x * y))) = s}

theorem obtuse_triangle_area_interval :
  ∃ (a b : ℝ), a = 4 * Real.sqrt 21 ∧ b = 20 ∧
  (∀ s, Set.Nonempty (τ s) ∧ (∀ t₁ t₂, t₁ ∈ τ s → t₂ ∈ τ s → t₁ = t₂) ↔ a ≤ s ∧ s < b) ∧
  a^2 + b^2 = 736 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_area_interval_l478_47816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_socks_ratio_l478_47849

/-- Represents the number of pairs of blue socks -/
def b : ℕ := sorry

/-- Represents the price of one pair of blue socks -/
def x : ℝ := sorry

/-- The original cost of the socks order -/
def original_cost : ℝ := 15 * x + b * x

/-- The cost after interchanging the quantities -/
def interchanged_cost : ℝ := 3 * b * x + 5 * x

/-- Theorem stating that under the given conditions, b must equal 14 -/
theorem socks_ratio : interchanged_cost = 1.6 * original_cost → b = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_socks_ratio_l478_47849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_complement_N_l478_47899

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x ≥ -2}
def N : Set ℝ := {x : ℝ | Real.exp (x * Real.log 2) - 1 > 0}

-- Define the complement of N in the real numbers
def C_ℝN : Set ℝ := {x : ℝ | ¬(x ∈ N)}

-- State the theorem
theorem intersection_M_complement_N :
  M ∩ C_ℝN = {x : ℝ | -2 ≤ x ∧ x ≤ 0} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_complement_N_l478_47899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_relation_l478_47891

noncomputable def g (n : ℤ) : ℝ := (2 + Real.sqrt 2) / 4 * ((1 + Real.sqrt 2) / 2) ^ n + 
                                   (2 - Real.sqrt 2) / 4 * ((1 - Real.sqrt 2) / 2) ^ n

theorem g_relation (n : ℤ) : g (n + 1) - g (n - 1) = g n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_relation_l478_47891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_binomial_sum_l478_47885

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_binomial_sum : 
  (1 : ℂ) + 6 * i + 15 * i^2 + 20 * i^3 + 15 * i^4 + 6 * i^5 + i^6 = -8*i :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_binomial_sum_l478_47885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_unique_base_l478_47882

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

theorem exponential_function_unique_base 
  (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : f a 2 = 4) : 
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_unique_base_l478_47882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sound_pressure_comparison_l478_47846

noncomputable def sound_pressure_level (p p₀ : ℝ) : ℝ := 20 * Real.log (p / p₀) / Real.log 10

theorem sound_pressure_comparison 
  (p₀ p₁ p₂ p₃ : ℝ) 
  (h_p₀ : p₀ > 0)
  (h_gasoline : 60 ≤ sound_pressure_level p₁ p₀ ∧ sound_pressure_level p₁ p₀ ≤ 90)
  (h_hybrid : 50 ≤ sound_pressure_level p₂ p₀ ∧ sound_pressure_level p₂ p₀ ≤ 60)
  (h_electric : sound_pressure_level p₃ p₀ = 40) :
  p₁ ≥ p₂ ∧ p₃ = 100 * p₀ ∧ p₁ ≤ 100 * p₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sound_pressure_comparison_l478_47846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_parametric_function_l478_47868

open Real

-- Define the parametric functions
noncomputable def x (t : ℝ) : ℝ := cos t + sin t
noncomputable def y (t : ℝ) : ℝ := sin (2 * t)

-- State the theorem
theorem second_derivative_parametric_function :
  ∀ t : ℝ, deriv (deriv (y ∘ (x⁻¹))) (x t) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_parametric_function_l478_47868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_segment_region_area_l478_47887

-- Define the circle
noncomputable def circle_radius : ℝ := 4

-- Define the length of the line segments
noncomputable def segment_length : ℝ := 8

-- Define the area of the region
noncomputable def region_area : ℝ := 16 * Real.pi

-- Theorem statement
theorem tangent_segment_region_area :
  let inner_radius := circle_radius
  let outer_radius := circle_radius * Real.sqrt 2
  π * (outer_radius^2 - inner_radius^2) = region_area :=
by
  -- Placeholder for the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_segment_region_area_l478_47887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_on_interval_l478_47896

noncomputable def f (x : ℝ) : ℝ := (5 - 4*x + x^2) / (2 - x)

theorem f_minimum_on_interval :
  (∀ x < 2, f x ≥ 2) ∧ (∃ x < 2, f x = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_on_interval_l478_47896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_36_l478_47877

noncomputable section

/-- The total cost function for producing x units -/
def C (x : ℝ) : ℝ := 300 + (1/12) * x^3 - 5 * x^2 + 170 * x

/-- The price per unit of the product -/
def price_per_unit : ℝ := 134

/-- The profit function -/
def L (x : ℝ) : ℝ := price_per_unit * x - C x

/-- Theorem stating that 36 units maximizes the profit -/
theorem max_profit_at_36 :
  ∀ x ≥ 0, L 36 ≥ L x := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_36_l478_47877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dual_of_regular_is_regular_convex_polyhedron_from_face_centers_is_regular_l478_47827

/-- A regular polyhedron -/
structure RegularPolyhedron where
  vertices : Set (Fin 3 → ℝ)
  faces : Set (Set (Fin 3 → ℝ))
  is_regular : Prop

/-- The dual of a polyhedron -/
noncomputable def dual (p : RegularPolyhedron) : Set (Fin 3 → ℝ) :=
  { v | ∃ f ∈ p.faces, v = center_of_face f }
where
  center_of_face (f : Set (Fin 3 → ℝ)) : Fin 3 → ℝ := sorry

/-- Theorem: The dual of a regular polyhedron is regular -/
theorem dual_of_regular_is_regular (p : RegularPolyhedron) (h : p.is_regular) :
  ∃ q : RegularPolyhedron, q.vertices = dual p ∧ q.is_regular := by
  sorry

/-- Main theorem: A convex polyhedron whose vertices are the centers of the faces
    of a regular polyhedron is itself regular -/
theorem convex_polyhedron_from_face_centers_is_regular
  (p : RegularPolyhedron)
  (h : p.is_regular)
  (q : Set (Fin 3 → ℝ))
  (hq : q = dual p) :
  ∃ r : RegularPolyhedron, r.vertices = q ∧ r.is_regular := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dual_of_regular_is_regular_convex_polyhedron_from_face_centers_is_regular_l478_47827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_F_l478_47837

open Real

-- Define the function F
noncomputable def F (A B x : ℝ) : ℝ :=
  |cos x^2 + 2 * sin x * cos x - sin x^2 + A * x + B|

-- Define the domain
def domain : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3/2 * π}

-- Define the maximum value M
noncomputable def M (A B : ℝ) : ℝ := ⨆ (x ∈ domain), F A B x

-- Theorem statement
theorem min_max_F :
  ∃ (A B : ℝ), ∀ (A' B' : ℝ), M A B ≤ M A' B' ∧ M A B = sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_F_l478_47837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trig_identity_l478_47803

-- Define the triangle PQR
def Triangle (P Q R : ℝ) := P + Q + R = Real.pi

-- Define the side lengths
def SideLengths (PQ PR QR : ℝ) := PQ = 8 ∧ PR = 7 ∧ QR = 5

-- Theorem statement
theorem triangle_trig_identity 
  (P Q R PQ PR QR : ℝ) 
  (h1 : Triangle P Q R) 
  (h2 : SideLengths PQ PR QR) : 
  (Real.cos ((P - Q)/2) / Real.sin (R/2)) - (Real.sin ((P - Q)/2) / Real.cos (R/2)) = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trig_identity_l478_47803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_three_squares_power_l478_47842

theorem sum_of_three_squares_power (n k : ℕ) 
  (hn : n > 0)
  (hk : k > 0)
  (h : ∃ (a b c : ℕ), n = a^2 + b^2 + c^2 ∧ a > 0 ∧ b > 0 ∧ c > 0) : 
  ∃ (A B C : ℕ), n^(2*k) = A^2 + B^2 + C^2 ∧ A > 0 ∧ B > 0 ∧ C > 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_three_squares_power_l478_47842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hostel_expenditure_l478_47864

theorem hostel_expenditure 
  (initial_students : ℕ) 
  (additional_students : ℕ) 
  (average_decrease : ℚ) 
  (total_increase : ℚ) 
  (h1 : initial_students = 100)
  (h2 : additional_students = 25)
  (h3 : average_decrease = 10)
  (h4 : total_increase = 500) :
  let new_total_students := initial_students + additional_students
  let original_average := (total_increase + (new_total_students : ℚ) * average_decrease) / ((new_total_students : ℚ) - (initial_students : ℚ))
  let new_average := original_average - average_decrease
  new_average * (new_total_students : ℚ) = 7500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hostel_expenditure_l478_47864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_differentiable_function_property_l478_47830

theorem continuous_differentiable_function_property (f : ℝ → ℝ) 
  (hf_cont : ContinuousOn f (Set.Icc 0 1))
  (hf_diff : DifferentiableOn ℝ f (Set.Ioo 0 1)) :
  (∃ a b : ℝ, ∀ x ∈ Set.Icc 0 1, f x = a * x + b) ∨
  (∃ t ∈ Set.Ioo 0 1, |f 1 - f 0| < |deriv f t|) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_differentiable_function_property_l478_47830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elder_age_is_thirty_l478_47855

/-- Given two persons whose ages differ by 20 years, and 5 years ago the elder was 5 times as old as the younger, prove that the present age of the elder person is 30 years. -/
theorem elder_age_is_thirty (y e : ℕ) : 
  e = y + 20 →                  -- The ages differ by 20 years
  e - 5 = 5 * (y - 5) →         -- 5 years ago, elder was 5 times younger's age
  e = 30 := by 
  intro h1 h2
  -- The proof steps would go here
  sorry

#check elder_age_is_thirty

end NUMINAMATH_CALUDE_ERRORFEEDBACK_elder_age_is_thirty_l478_47855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_real_domain_interval_range_positive_l478_47822

-- Define the function f
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := 2 / Real.sqrt (k * x^2 + 4 * k * x + 3)

-- Theorem 1
theorem domain_real (k : ℝ) :
  (∀ x, f k x ∈ Set.univ) → k ∈ Set.Ici 0 ∩ Set.Iio (3/4) := by
  sorry

-- Theorem 2
theorem domain_interval (k : ℝ) :
  (∀ x ∈ Set.Ioo (-6) 2, f k x ∈ Set.univ) ∧ 
  (∀ x ∉ Set.Ioo (-6) 2, f k x ∉ Set.univ) → k = -1/4 := by
  sorry

-- Theorem 3
theorem range_positive (k : ℝ) :
  (∀ y > 0, ∃ x, f k x = y) → k ∈ Set.Ici (3/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_real_domain_interval_range_positive_l478_47822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_approx_l478_47840

/-- The area of a circular sector with given radius and central angle -/
noncomputable def sectorArea (radius : ℝ) (angle : ℝ) : ℝ :=
  (angle / 360) * Real.pi * radius^2

theorem sector_area_approx : 
  let radius : ℝ := 15
  let angle : ℝ := 42
  abs (sectorArea radius angle - 82.47) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_approx_l478_47840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_and_sum_l478_47893

noncomputable def original_fraction : ℝ := 5 / (3 * Real.rpow 7 (1/3))

noncomputable def rationalized_fraction : ℝ := 5 * Real.rpow 49 (1/3) / 21

theorem rationalize_and_sum :
  (original_fraction = rationalized_fraction) ∧
  (5 + 49 + 21 = 75) := by
  sorry

#eval 5 + 49 + 21

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_and_sum_l478_47893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_plus_inverse_cube_l478_47861

theorem cube_plus_inverse_cube (m : ℝ) : m^2 - 8*m + 1 = 0 → m^3 + 1/m^3 = 61 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_plus_inverse_cube_l478_47861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_division_result_two_l478_47821

theorem no_division_result_two (digits : List Nat) 
  (h_digits : digits = [2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9]) : 
  ¬ ∃ (A B : Nat), 
    (A ≠ 0 ∧ B ≠ 0) ∧
    (∃ (l1 l2 : List Nat), l1 ++ l2 = digits ∧ 
      A = l1.foldl (fun acc d => acc * 10 + d) 0 ∧
      B = l2.foldl (fun acc d => acc * 10 + d) 0) ∧
    A / B = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_division_result_two_l478_47821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_is_integer_l478_47844

def P (x : ℤ) : ℚ :=
  (1 : ℚ) / 630 - (1 : ℚ) / 21 * x^7 + (13 : ℚ) / 30 * x^5 - (82 : ℚ) / 63 * x^3 + (32 : ℚ) / 35 * x

theorem P_is_integer : ∀ x : ℤ, ∃ n : ℤ, P x = n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_is_integer_l478_47844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_original_f_f_odd_f_increasing_f_inequality_iff_l478_47832

noncomputable section

variable (a : ℝ)

def f (x : ℝ) := (a / (a^2 - 1)) * (a^x - a^(-x))

def original_f (x : ℝ) := (a / (a^2 - 1)) * (Real.exp (x * Real.log a) - 1 / Real.exp (x * Real.log a))

theorem f_eq_original_f (h1 : a > 0) (h2 : a ≠ 1) : ∀ x, f a x = original_f a x := by sorry

theorem f_odd (h1 : a > 0) (h2 : a ≠ 1) : ∀ x, f a (-x) = -(f a x) := by sorry

theorem f_increasing (h1 : a > 0) (h2 : a ≠ 1) : ∀ x y, x < y → f a x < f a y := by sorry

theorem f_inequality_iff (h1 : a > 0) (h2 : a ≠ 1) (k : ℝ) : 
  (∀ t, f a (t^2 - 2*t) + f a (2*t^2 - k) > 0) ↔ k < -1/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_original_f_f_odd_f_increasing_f_inequality_iff_l478_47832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_value_l478_47880

def sequence_a : ℕ → ℚ
  | 0 => 2  -- We define the base case for 0 to match Lean's natural number indexing
  | n + 1 => 1 - 1 / sequence_a n

theorem a_5_value : sequence_a 4 = 1 / 2 := by
  -- We use 4 instead of 5 because Lean's indexing starts at 0
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_value_l478_47880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_piece_sum_l478_47863

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a triangular piece cut from a rectangular prism -/
noncomputable def triangularPieceVolume (d : PrismDimensions) : ℝ :=
  (1/2) * d.length * d.width * d.height

/-- Calculates the surface area of icing on a triangular piece (top and three sides) -/
noncomputable def triangularPieceIcingArea (d : PrismDimensions) : ℝ :=
  (1/2) * d.length * d.width + 3 * ((1/2) * d.height * (Real.sqrt (d.length^2 + d.width^2)))

/-- Theorem: The sum of the volume of the triangular piece and its icing area is 15 -/
theorem triangular_piece_sum (d : PrismDimensions) 
    (h1 : d.length = 3) 
    (h2 : d.width = 2) 
    (h3 : d.height = 2) : 
    triangularPieceVolume d + triangularPieceIcingArea d = 15 := by
  sorry

#check triangular_piece_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_piece_sum_l478_47863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_in_fourth_quadrant_l478_47876

def Z1 : ℂ := 2 + Complex.I
def Z2 : ℂ := 1 + Complex.I

theorem fraction_in_fourth_quadrant :
  let z : ℂ := Z1 / Z2
  (z.re > 0 ∧ z.im < 0) := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_in_fourth_quadrant_l478_47876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_graph_shift_right_l478_47884

/-- Moving a sine function graph to the right -/
theorem sine_graph_shift_right (f : ℝ → ℝ) (x : ℝ) :
  (fun x => f (x - π/6)) x = Real.sin (2*x - π/6) ↔ 
  f x = Real.sin (2*x + π/6) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_graph_shift_right_l478_47884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_two_black_two_white_given_first_black_l478_47807

/-- The number of black balls in the bin -/
def black_balls : ℕ := 10

/-- The number of white balls in the bin -/
def white_balls : ℕ := 8

/-- The total number of balls drawn -/
def balls_drawn : ℕ := 4

/-- The probability of drawing 2 black balls and 2 white balls, given that the first ball drawn is black -/
def probability : ℚ := 63 / 170

theorem probability_of_two_black_two_white_given_first_black :
  probability = (Nat.choose (black_balls - 1) 1 * Nat.choose white_balls 2) / Nat.choose (black_balls + white_balls - 1) 3 := by
  sorry

#eval probability
#eval (Nat.choose (black_balls - 1) 1 * Nat.choose white_balls 2) / Nat.choose (black_balls + white_balls - 1) 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_two_black_two_white_given_first_black_l478_47807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_sine_equality_l478_47800

theorem contrapositive_sine_equality (x y : ℝ) : (¬(Real.sin x = Real.sin y) → ¬(x = y)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_sine_equality_l478_47800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_a_l478_47856

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - (3/2) * x^2

-- State the theorem
theorem determine_a :
  ∀ a : ℝ,
  (∀ x : ℝ, f a x ≤ 1/6) →
  (∀ x : ℝ, x ∈ Set.Icc (1/4 : ℝ) (1/2 : ℝ) → f a x ≥ 1/8) →
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_a_l478_47856
