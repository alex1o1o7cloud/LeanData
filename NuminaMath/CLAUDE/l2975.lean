import Mathlib

namespace sin_sum_of_roots_l2975_297514

theorem sin_sum_of_roots (a b c : ℝ) (α β : ℝ) : 
  (∀ x, a * Real.cos x + b * Real.sin x + c = 0 ↔ x = α ∨ x = β) →
  0 < α → α < π →
  0 < β → β < π →
  α ≠ β →
  Real.sin (α + β) = (2 * a * b) / (a^2 + b^2) := by
sorry

end sin_sum_of_roots_l2975_297514


namespace eggs_sold_equals_450_l2975_297585

/-- The number of eggs in one tray -/
def eggs_per_tray : ℕ := 30

/-- The initial number of trays to be collected -/
def initial_trays : ℕ := 10

/-- The number of trays dropped (lost) -/
def dropped_trays : ℕ := 2

/-- The number of additional trays added after the accident -/
def additional_trays : ℕ := 7

/-- The total number of eggs sold -/
def eggs_sold : ℕ := (initial_trays - dropped_trays + additional_trays) * eggs_per_tray

theorem eggs_sold_equals_450 : eggs_sold = 450 := by
  sorry

end eggs_sold_equals_450_l2975_297585


namespace largest_divisor_of_n_l2975_297508

theorem largest_divisor_of_n (n : ℕ) (h1 : n > 0) (h2 : 1080 ∣ n^2) : ∃ q : ℕ, q > 0 ∧ q ∣ n ∧ ∀ m : ℕ, m > 0 ∧ m ∣ n → m ≤ q ∧ q = 6 := by
  sorry

end largest_divisor_of_n_l2975_297508


namespace b₁_value_l2975_297528

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := 8 + 32*x - 12*x^2 - 4*x^3 + x^4

-- Define the set of roots of f(x)
def roots_f : Set ℝ := {x | f x = 0}

-- Define the polynomial g(x)
def g (b₀ b₁ b₂ b₃ : ℝ) (x : ℝ) : ℝ := b₀ + b₁*x + b₂*x^2 + b₃*x^3 + x^4

-- Define the set of roots of g(x)
def roots_g (b₀ b₁ b₂ b₃ : ℝ) : Set ℝ := {x | g b₀ b₁ b₂ b₃ x = 0}

theorem b₁_value (x₁ x₂ x₃ x₄ : ℝ) 
  (h₁ : roots_f = {x₁, x₂, x₃, x₄})
  (h₂ : x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄)
  (h₃ : ∃ b₀ b₁ b₂ b₃, roots_g b₀ b₁ b₂ b₃ = {x₁^2, x₂^2, x₃^2, x₄^2}) :
  ∃ b₀ b₂ b₃, g b₀ (-1024) b₂ b₃ = g b₀ b₁ b₂ b₃ := by sorry

end b₁_value_l2975_297528


namespace fraction_equality_l2975_297541

theorem fraction_equality : 
  (12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) / 
  (1 - 2 + 3 - 4 + 5 - 6 + 7 - 8 + 9 - 10 + 11) = 1 := by
  sorry

end fraction_equality_l2975_297541


namespace roots_sum_bound_l2975_297583

theorem roots_sum_bound (z x : ℂ) : 
  z ≠ x → 
  z^2017 = 1 → 
  x^2017 = 1 → 
  Complex.abs (z + x) < Real.sqrt (2 + Real.sqrt 5) :=
sorry

end roots_sum_bound_l2975_297583


namespace largest_power_of_two_dividing_seven_power_minus_one_l2975_297560

theorem largest_power_of_two_dividing_seven_power_minus_one :
  (∀ k : ℕ, k > 14 → ¬(2^k ∣ 7^2048 - 1)) ∧
  (2^14 ∣ 7^2048 - 1) :=
sorry

end largest_power_of_two_dividing_seven_power_minus_one_l2975_297560


namespace geometric_sequence_third_term_l2975_297566

theorem geometric_sequence_third_term (a : ℕ → ℝ) (q : ℝ) (S₄ : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- Geometric sequence condition
  q = 2 →                       -- Common ratio
  S₄ = 60 →                     -- Sum of first 4 terms
  (a 0 * (1 - q^4)) / (1 - q) = S₄ →  -- Sum formula for geometric sequence
  a 2 = 16 := by               -- Third term (index 2 in 0-based indexing)
sorry

end geometric_sequence_third_term_l2975_297566


namespace cost_per_side_l2975_297573

-- Define the park as a square
structure SquarePark where
  side_cost : ℝ
  total_cost : ℝ

-- Define the properties of the square park
def is_valid_square_park (park : SquarePark) : Prop :=
  park.total_cost = 224 ∧ park.total_cost = 4 * park.side_cost

-- Theorem statement
theorem cost_per_side (park : SquarePark) (h : is_valid_square_park park) : 
  park.side_cost = 56 := by
  sorry

end cost_per_side_l2975_297573


namespace quadratic_linear_third_quadrant_l2975_297588

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadratic equation ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a linear function y = mx + b -/
structure LinearFunction where
  m : ℝ
  b : ℝ

/-- Checks if a point is in the third quadrant -/
def isInThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Checks if a quadratic equation has no real roots -/
def hasNoRealRoots (eq : QuadraticEquation) : Prop :=
  eq.b^2 - 4*eq.a*eq.c < 0

/-- Checks if a linear function passes through a point -/
def passesThrough (f : LinearFunction) (p : Point) : Prop :=
  p.y = f.m * p.x + f.b

theorem quadratic_linear_third_quadrant 
  (b : ℝ) 
  (quad : QuadraticEquation) 
  (lin : LinearFunction) :
  quad = QuadraticEquation.mk 1 2 (b - 3) →
  hasNoRealRoots quad →
  lin = LinearFunction.mk (-2) b →
  ¬ ∃ (p : Point), isInThirdQuadrant p ∧ passesThrough lin p :=
sorry

end quadratic_linear_third_quadrant_l2975_297588


namespace exists_special_function_l2975_297590

/-- A function satisfying specific properties --/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧ 
  (f 0 = 1) ∧
  (∀ x, f (x + 3) = -f (-(x + 3))) ∧
  (f (-9) = 0) ∧
  (f 18 = -1) ∧
  (f 24 = 1)

/-- Theorem stating the existence of a function with the specified properties --/
theorem exists_special_function : ∃ f : ℝ → ℝ, special_function f := by
  sorry

end exists_special_function_l2975_297590


namespace probability_not_monday_l2975_297567

theorem probability_not_monday (p_monday : ℚ) (h : p_monday = 1/7) : 
  1 - p_monday = 6/7 := by
  sorry

end probability_not_monday_l2975_297567


namespace peter_large_glasses_bought_l2975_297597

def small_glass_cost : ℕ := 3
def large_glass_cost : ℕ := 5
def initial_amount : ℕ := 50
def small_glasses_bought : ℕ := 8
def change : ℕ := 1

def large_glasses_bought : ℕ := (initial_amount - change - small_glass_cost * small_glasses_bought) / large_glass_cost

theorem peter_large_glasses_bought :
  large_glasses_bought = 5 :=
by sorry

end peter_large_glasses_bought_l2975_297597


namespace average_books_theorem_l2975_297592

/-- Represents the number of books borrowed by a student -/
structure BooksBorrowed where
  count : ℕ
  is_valid : count ≤ 6

/-- Represents the distribution of books borrowed in the class -/
structure ClassDistribution where
  total_students : ℕ
  zero_books : ℕ
  one_book : ℕ
  two_books : ℕ
  at_least_three : ℕ
  is_valid : total_students = zero_books + one_book + two_books + at_least_three

def average_books (dist : ClassDistribution) : ℚ :=
  let total_books := dist.one_book + 2 * dist.two_books + 3 * dist.at_least_three
  total_books / dist.total_students

theorem average_books_theorem (dist : ClassDistribution) 
  (h1 : dist.total_students = 40)
  (h2 : dist.zero_books = 2)
  (h3 : dist.one_book = 12)
  (h4 : dist.two_books = 13)
  (h5 : dist.at_least_three = dist.total_students - (dist.zero_books + dist.one_book + dist.two_books)) :
  average_books dist = 77 / 40 := by
  sorry

#eval (77 : ℚ) / 40

end average_books_theorem_l2975_297592


namespace family_pizza_order_l2975_297506

/-- Calculates the number of pizzas needed for a family -/
def pizzas_needed (adults : ℕ) (children : ℕ) (adult_slices : ℕ) (child_slices : ℕ) (slices_per_pizza : ℕ) : ℕ :=
  ((adults * adult_slices + children * child_slices) + slices_per_pizza - 1) / slices_per_pizza

/-- Proves that a family of 2 adults and 6 children needs 3 pizzas -/
theorem family_pizza_order : pizzas_needed 2 6 3 1 4 = 3 := by
  sorry

end family_pizza_order_l2975_297506


namespace plate_arrangement_theorem_l2975_297586

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def circular_permutations (n : ℕ) (groups : List ℕ) : ℕ :=
  factorial (n - 1) / (groups.map factorial).prod

theorem plate_arrangement_theorem : 
  let total_plates := 14
  let blue_plates := 6
  let red_plates := 3
  let green_plates := 3
  let orange_plates := 2
  let total_arrangements := circular_permutations total_plates [blue_plates, red_plates, green_plates, orange_plates]
  let adjacent_green_arrangements := circular_permutations (total_plates - green_plates + 1) [blue_plates, red_plates, 1, orange_plates]
  total_arrangements - adjacent_green_arrangements = 1349070 := by
  sorry

end plate_arrangement_theorem_l2975_297586


namespace max_value_quadratic_l2975_297530

theorem max_value_quadratic (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + 2*x + 1) :
  ∃ y ∈ Set.Icc (-2 : ℝ) 2, ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x ≤ f y ∧ f y = 9 :=
sorry

end max_value_quadratic_l2975_297530


namespace min_sum_squares_l2975_297569

theorem min_sum_squares (x y z : ℝ) (h : x^2 + y^2 + z^2 - 3*x*y*z = 1) :
  ∀ a b c : ℝ, a^2 + b^2 + c^2 - 3*a*b*c = 1 → x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2 :=
by sorry

end min_sum_squares_l2975_297569


namespace diamonds_15_diamonds_eq_diamonds_closed_diamonds_closed_15_l2975_297510

/-- The number of diamonds in the nth figure of the sequence -/
def diamonds (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else diamonds (n - 1) + 4 * n

/-- The theorem stating that the 15th figure contains 480 diamonds -/
theorem diamonds_15 : diamonds 15 = 480 := by
  sorry

/-- Alternative definition using the closed form formula -/
def diamonds_closed (n : ℕ) : ℕ := 2 * n * (n + 1)

/-- Theorem stating the equivalence of the recursive and closed form definitions -/
theorem diamonds_eq_diamonds_closed (n : ℕ) : diamonds n = diamonds_closed n := by
  sorry

/-- The theorem stating that the 15th figure contains 480 diamonds using the closed form -/
theorem diamonds_closed_15 : diamonds_closed 15 = 480 := by
  sorry

end diamonds_15_diamonds_eq_diamonds_closed_diamonds_closed_15_l2975_297510


namespace curve_C_equation_min_distance_QM_l2975_297571

-- Define points A and B
def A : ℝ × ℝ := (-1, 2)
def B : ℝ × ℝ := (0, 1)

-- Define the distance condition for point P
def distance_condition (P : ℝ × ℝ) : Prop :=
  (P.1 + 1)^2 + (P.2 - 2)^2 = 2 * (P.1^2 + (P.2 - 1)^2)

-- Define curve C
def C : Set (ℝ × ℝ) := {P | distance_condition P}

-- Define line l₁
def l₁ : Set (ℝ × ℝ) := {Q | 3 * Q.1 - 4 * Q.2 + 12 = 0}

-- Theorem for the equation of curve C
theorem curve_C_equation : C = {P : ℝ × ℝ | (P.1 - 1)^2 + P.2^2 = 4} := by sorry

-- Theorem for the minimum distance
theorem min_distance_QM : 
  ∀ Q ∈ l₁, ∃ M ∈ C, ∀ M' ∈ C, dist Q M ≤ dist Q M' ∧ dist Q M = Real.sqrt 5 := by sorry


end curve_C_equation_min_distance_QM_l2975_297571


namespace polynomial_multiplication_l2975_297572

theorem polynomial_multiplication (t : ℝ) :
  (3 * t^3 - 2 * t^2 + 4 * t - 1) * (2 * t^2 - 5 * t + 3) =
  6 * t^5 - 19 * t^4 + 27 * t^3 - 28 * t^2 + 17 * t - 3 :=
by sorry

end polynomial_multiplication_l2975_297572


namespace max_m_value_l2975_297526

theorem max_m_value (m : ℝ) : 
  (¬ ∃ x : ℝ, x ≥ 3 ∧ 2*x - 1 < m) → m ≤ 5 :=
by sorry

end max_m_value_l2975_297526


namespace rectangle_only_convex_four_right_angles_l2975_297532

/-- A polygon is a set of points in the plane -/
def Polygon : Type := Set (ℝ × ℝ)

/-- A polygon is convex if for any two points in the polygon, the line segment between them is entirely contained within the polygon -/
def is_convex (p : Polygon) : Prop := sorry

/-- The number of sides in a polygon -/
def num_sides (p : Polygon) : ℕ := sorry

/-- The number of right angles in a polygon -/
def num_right_angles (p : Polygon) : ℕ := sorry

/-- A rectangle is a polygon with exactly four sides and four right angles -/
def is_rectangle (p : Polygon) : Prop :=
  num_sides p = 4 ∧ num_right_angles p = 4

theorem rectangle_only_convex_four_right_angles (p : Polygon) :
  is_convex p ∧ num_right_angles p = 4 → is_rectangle p :=
sorry

end rectangle_only_convex_four_right_angles_l2975_297532


namespace inequality_proof_l2975_297538

theorem inequality_proof (x y z t : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (ht : t > 0) :
  (x + y + z + t) / 2 + 4 / (x*y + y*z + z*t + t*x) ≥ 3 := by
  sorry

end inequality_proof_l2975_297538


namespace pages_per_day_l2975_297559

theorem pages_per_day (book1_pages book2_pages : ℕ) (days : ℕ) 
  (h1 : book1_pages = 180) 
  (h2 : book2_pages = 100) 
  (h3 : days = 14) : 
  (book1_pages + book2_pages) / days = 20 := by
  sorry

end pages_per_day_l2975_297559


namespace solve_exponential_equation_l2975_297599

theorem solve_exponential_equation :
  ∀ x : ℝ, (64 : ℝ)^(3*x + 1) = (16 : ℝ)^(4*x - 5) ↔ x = -13 :=
by
  sorry

end solve_exponential_equation_l2975_297599


namespace range_of_a_l2975_297513

-- Define propositions p and q
def p (a : ℝ) : Prop := ∀ x, x^2 - (a-1)*x + 1 > 0

def q (a : ℝ) : Prop := ∀ x y, x < y → (a+1)^x < (a+1)^y

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  (¬(p a ∧ q a) ∧ (p a ∨ q a)) → 
  ((-1 < a ∧ a ≤ 0) ∨ a ≥ 3) :=
by sorry

end range_of_a_l2975_297513


namespace shooting_competition_solution_l2975_297505

/-- Represents the number of shots for each score (8, 9, 10) -/
structure ScoreCounts where
  eight : ℕ
  nine : ℕ
  ten : ℕ

/-- Checks if a ScoreCounts satisfies the competition conditions -/
def is_valid_score (s : ScoreCounts) : Prop :=
  s.eight + s.nine + s.ten > 11 ∧
  8 * s.eight + 9 * s.nine + 10 * s.ten = 100

/-- The set of all valid score combinations -/
def valid_scores : Set ScoreCounts :=
  { s | is_valid_score s }

/-- The theorem stating the unique solution to the shooting competition problem -/
theorem shooting_competition_solution :
  valid_scores = { ⟨10, 0, 2⟩, ⟨9, 2, 1⟩, ⟨8, 4, 0⟩ } :=
sorry

end shooting_competition_solution_l2975_297505


namespace savings_calculation_l2975_297577

theorem savings_calculation (income : ℕ) (ratio_income : ℕ) (ratio_expenditure : ℕ) :
  income = 21000 →
  ratio_income = 7 →
  ratio_expenditure = 6 →
  income - (income * ratio_expenditure / ratio_income) = 3000 := by
  sorry

end savings_calculation_l2975_297577


namespace max_distance_with_turns_l2975_297561

theorem max_distance_with_turns (total_distance : ℕ) (num_turns : ℕ) 
  (h1 : total_distance = 500) (h2 : num_turns = 300) :
  ∃ (d : ℝ), d ≤ Real.sqrt 145000 ∧ 
  (∀ (a b : ℕ), a + b = total_distance → a ≥ num_turns / 2 → b ≥ num_turns / 2 → 
    Real.sqrt (a^2 + b^2 : ℝ) ≤ d) ∧
  ⌊d⌋ = 380 := by
  sorry

end max_distance_with_turns_l2975_297561


namespace necessary_condition_for_inequality_l2975_297551

theorem necessary_condition_for_inequality (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 2 3 → x^2 - a ≤ 0) → a ≥ 8 := by
  sorry

end necessary_condition_for_inequality_l2975_297551


namespace gourmet_smores_cost_l2975_297552

/-- Represents the cost and pack information for an ingredient --/
structure IngredientInfo where
  single_cost : ℚ
  pack_size : ℕ
  pack_cost : ℚ

/-- Calculates the minimum cost to buy a certain quantity of an ingredient --/
def min_cost (info : IngredientInfo) (quantity : ℕ) : ℚ :=
  let packs_needed := (quantity + info.pack_size - 1) / info.pack_size
  packs_needed * info.pack_cost

/-- Calculates the total cost for all ingredients --/
def total_cost (people : ℕ) (smores_per_person : ℕ) : ℚ :=
  let graham_crackers := min_cost ⟨0.1, 20, 1.8⟩ (people * smores_per_person * 1)
  let marshmallows := min_cost ⟨0.15, 15, 2.0⟩ (people * smores_per_person * 1)
  let chocolate := min_cost ⟨0.25, 10, 2.0⟩ (people * smores_per_person * 1)
  let caramel := min_cost ⟨0.2, 25, 4.5⟩ (people * smores_per_person * 2)
  let toffee := min_cost ⟨0.05, 50, 2.0⟩ (people * smores_per_person * 4)
  graham_crackers + marshmallows + chocolate + caramel + toffee

theorem gourmet_smores_cost : total_cost 8 3 = 26.6 := by
  sorry

end gourmet_smores_cost_l2975_297552


namespace odd_function_root_property_l2975_297579

/-- A function f : ℝ → ℝ is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- x₀ is a root of f(x) + exp(x) = 0 -/
def IsRootOf (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  f x₀ + Real.exp x₀ = 0

theorem odd_function_root_property (f : ℝ → ℝ) (x₀ : ℝ) 
    (h_odd : IsOdd f) (h_root : IsRootOf f x₀) :
    Real.exp (-x₀) * f (-x₀) - 1 = 0 := by
  sorry

end odd_function_root_property_l2975_297579


namespace cube_volume_ratio_l2975_297537

/-- Given two cubes with edge lengths a and b, where a/b = 3/1 and the volume of the cube
    with edge length a is 27 units, prove that the volume of the cube with edge length b is 1 unit. -/
theorem cube_volume_ratio (a b : ℝ) (h1 : a / b = 3 / 1) (h2 : a^3 = 27) : b^3 = 1 := by
  sorry

end cube_volume_ratio_l2975_297537


namespace least_squares_for_25x25_l2975_297558

theorem least_squares_for_25x25 (n : Nat) (h1 : n = 25) (h2 : n * n = 625) :
  ∃ f : Nat → Nat, f n ≥ (n^2 - 1) / 2 ∧ f n ≥ 312 := by
  sorry

end least_squares_for_25x25_l2975_297558


namespace cell_growth_theorem_l2975_297587

def cell_growth (initial : ℕ) (hours : ℕ) : ℕ :=
  if hours = 0 then
    initial
  else
    2 * (cell_growth initial (hours - 1) - 2)

theorem cell_growth_theorem :
  cell_growth 9 8 = 1284 := by
  sorry

end cell_growth_theorem_l2975_297587


namespace range_of_f_l2975_297543

-- Define the linear function
def f (x : ℝ) : ℝ := -2 * x + 5

-- Define the domain
def domain : Set ℝ := {x | -1 < x ∧ x < 1}

-- Theorem statement
theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {y | 3 < y ∧ y < 7} := by sorry

end range_of_f_l2975_297543


namespace complex_circle_equation_l2975_297509

open Complex

theorem complex_circle_equation (z : ℂ) (x y : ℝ) :
  z = x + y * I →
  abs (z - 2) = 1 →
  (x - 2)^2 + y^2 = 1 := by
sorry

end complex_circle_equation_l2975_297509


namespace no_four_distinct_squares_sum_to_100_l2975_297519

theorem no_four_distinct_squares_sum_to_100 : 
  ¬ ∃ (a b c d : ℕ), 
    (0 < a) ∧ (a < b) ∧ (b < c) ∧ (c < d) ∧ 
    (a^2 + b^2 + c^2 + d^2 = 100) :=
sorry

end no_four_distinct_squares_sum_to_100_l2975_297519


namespace sqrt_400_divided_by_2_l2975_297522

theorem sqrt_400_divided_by_2 : Real.sqrt 400 / 2 = 10 := by sorry

end sqrt_400_divided_by_2_l2975_297522


namespace correct_mean_calculation_l2975_297576

theorem correct_mean_calculation (n : ℕ) (initial_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 25 ∧ initial_mean = 190 ∧ incorrect_value = 130 ∧ correct_value = 165 →
  (n : ℚ) * initial_mean - incorrect_value + correct_value = n * 191.4 := by
  sorry

end correct_mean_calculation_l2975_297576


namespace weight_division_l2975_297557

theorem weight_division (n : ℕ) : 
  (∃ (a b c : ℕ), a + b + c = n * (n + 1) / 2 ∧ a = b ∧ b = c) ↔ 
  (n > 3 ∧ (n % 3 = 0 ∨ n % 3 = 2)) :=
by sorry

end weight_division_l2975_297557


namespace probability_even_sum_l2975_297553

def card_set : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def is_even_sum (pair : Nat × Nat) : Bool :=
  (pair.1 + pair.2) % 2 == 0

def total_combinations : Nat :=
  Nat.choose 9 2

def even_sum_combinations : Nat :=
  Nat.choose 4 2 + Nat.choose 5 2

theorem probability_even_sum :
  (even_sum_combinations : ℚ) / total_combinations = 4 / 9 := by
  sorry

end probability_even_sum_l2975_297553


namespace negation_equivalence_l2975_297591

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x < 1 ∧ x^2 + 2*x + 1 ≤ 0) ↔ (∀ x : ℝ, x < 1 → x^2 + 2*x + 1 > 0) := by
  sorry

end negation_equivalence_l2975_297591


namespace semicircular_plot_radius_l2975_297518

/-- The radius of a semicircular plot given the total fence length and opening length. -/
theorem semicircular_plot_radius 
  (total_fence_length : ℝ) 
  (opening_length : ℝ) 
  (h1 : total_fence_length = 33) 
  (h2 : opening_length = 3) : 
  ∃ (radius : ℝ), radius = (total_fence_length - opening_length) / (Real.pi + 2) :=
sorry

end semicircular_plot_radius_l2975_297518


namespace percentage_difference_l2975_297525

theorem percentage_difference : 
  (80 / 100 * 40) - (4 / 5 * 15) = 20 := by
  sorry

end percentage_difference_l2975_297525


namespace farmer_brown_animals_legs_l2975_297562

/-- The number of legs for each animal type -/
def chicken_legs : ℕ := 2
def sheep_legs : ℕ := 4
def grasshopper_legs : ℕ := 6
def spider_legs : ℕ := 8

/-- The number of each animal type -/
def num_chickens : ℕ := 7
def num_sheep : ℕ := 5
def num_grasshoppers : ℕ := 10
def num_spiders : ℕ := 3

/-- The total number of legs -/
def total_legs : ℕ := 
  num_chickens * chicken_legs + 
  num_sheep * sheep_legs + 
  num_grasshoppers * grasshopper_legs + 
  num_spiders * spider_legs

theorem farmer_brown_animals_legs : total_legs = 118 := by
  sorry

end farmer_brown_animals_legs_l2975_297562


namespace arithmetic_simplification_l2975_297554

theorem arithmetic_simplification : 4 * (8 - 3 + 2) / 2 = 14 := by
  sorry

end arithmetic_simplification_l2975_297554


namespace cube_root_y_fourth_root_y_five_eq_four_l2975_297516

theorem cube_root_y_fourth_root_y_five_eq_four (y : ℝ) :
  (y * (y^5)^(1/4))^(1/3) = 4 → y = 2^(8/3) := by
  sorry

end cube_root_y_fourth_root_y_five_eq_four_l2975_297516


namespace heartsuit_three_eight_l2975_297574

-- Define the operation ⊛
def heartsuit (x y : ℝ) : ℝ := 4 * x + 6 * y

-- Theorem statement
theorem heartsuit_three_eight : heartsuit 3 8 = 60 := by
  sorry

end heartsuit_three_eight_l2975_297574


namespace root_equation_problem_l2975_297536

/-- Given two equations with constants p and q, prove that p = 5, q = -10, and 50p + q = 240 -/
theorem root_equation_problem (p q : ℝ) : 
  (∃! x y : ℝ, x ≠ y ∧ ((x + p) * (x + q) * (x - 8) = 0 ∨ x = 5)) →
  (∃! x y : ℝ, x ≠ y ∧ ((x + 2*p) * (x - 5) * (x - 10) = 0 ∨ x = -q ∨ x = 8)) →
  p = 5 ∧ q = -10 ∧ 50*p + q = 240 := by
sorry


end root_equation_problem_l2975_297536


namespace investment_growth_l2975_297580

/-- Given an initial investment that grows to $400 after 4 years at 25% simple interest per year,
    prove that the value after 6 years is $500. -/
theorem investment_growth (P : ℝ) : 
  P + P * 0.25 * 4 = 400 → 
  P + P * 0.25 * 6 = 500 := by
  sorry

end investment_growth_l2975_297580


namespace quadratic_equation_solution_property_l2975_297595

theorem quadratic_equation_solution_property (k : ℝ) : 
  (∃ a b : ℝ, 
    (3 * a^2 + 6 * a + k = 0) ∧ 
    (3 * b^2 + 6 * b + k = 0) ∧ 
    (abs (a - b) = 2 * (a^2 + b^2))) ↔ 
  (k = 3 ∨ k = 45/16) :=
sorry

end quadratic_equation_solution_property_l2975_297595


namespace students_meeting_time_l2975_297568

/-- Two students walking towards each other -/
theorem students_meeting_time 
  (distance : ℝ) 
  (speed1 : ℝ) 
  (speed2 : ℝ) 
  (h1 : distance = 350) 
  (h2 : speed1 = 1.6) 
  (h3 : speed2 = 1.9) : 
  distance / (speed1 + speed2) = 100 := by
  sorry

end students_meeting_time_l2975_297568


namespace rollercoaster_interval_l2975_297589

/-- Given that 7 students ride a rollercoaster every certain minutes,
    and 21 students rode the rollercoaster in 15 minutes,
    prove that the time interval for 7 students to ride the rollercoaster is 5 minutes. -/
theorem rollercoaster_interval (students_per_ride : ℕ) (total_students : ℕ) (total_time : ℕ) :
  students_per_ride = 7 →
  total_students = 21 →
  total_time = 15 →
  (total_time / (total_students / students_per_ride) : ℚ) = 5 :=
by sorry

end rollercoaster_interval_l2975_297589


namespace sum_of_solutions_eq_seventeen_sixths_l2975_297584

theorem sum_of_solutions_eq_seventeen_sixths :
  let f : ℝ → ℝ := λ x => (3*x + 5) * (2*x - 9)
  (∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) →
  (∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ + x₂ = 17/6) :=
by sorry

end sum_of_solutions_eq_seventeen_sixths_l2975_297584


namespace systematic_sampling_interval_count_l2975_297555

theorem systematic_sampling_interval_count :
  let total_employees : ℕ := 840
  let sample_size : ℕ := 42
  let interval_start : ℕ := 481
  let interval_end : ℕ := 720
  let interval_size : ℕ := interval_end - interval_start + 1
  let sampling_interval : ℕ := total_employees / sample_size
  (interval_size : ℚ) / (total_employees : ℚ) * (sample_size : ℚ) = 12 := by
  sorry

end systematic_sampling_interval_count_l2975_297555


namespace range_of_a_l2975_297524

theorem range_of_a (x a : ℝ) : 
  (∀ x, (1/2 ≤ x ∧ x ≤ 1) → ¬((x-a)*(x-a-1) > 0)) ∧ 
  (∃ x, ¬(1/2 ≤ x ∧ x ≤ 1) ∧ ¬((x-a)*(x-a-1) > 0)) →
  (0 ≤ a ∧ a ≤ 1/2) :=
by sorry

end range_of_a_l2975_297524


namespace alternating_color_probability_l2975_297521

/-- The number of white balls in the box -/
def white_balls : ℕ := 5

/-- The number of black balls in the box -/
def black_balls : ℕ := 5

/-- The total number of balls in the box -/
def total_balls : ℕ := white_balls + black_balls

/-- The number of successful alternating sequences -/
def successful_sequences : ℕ := 2

/-- The probability of drawing all balls with alternating colors -/
def alternating_probability : ℚ := successful_sequences / (total_balls.choose white_balls)

theorem alternating_color_probability :
  alternating_probability = 1 / 126 := by
  sorry

end alternating_color_probability_l2975_297521


namespace empty_solution_set_iff_a_in_range_l2975_297529

-- Define the quadratic function
def f (a x : ℝ) : ℝ := (a - 1) * x^2 + 2 * (a - 1) * x - 4

-- Define the property of empty solution set
def has_empty_solution_set (a : ℝ) : Prop :=
  ∀ x : ℝ, f a x < 0

-- State the theorem
theorem empty_solution_set_iff_a_in_range :
  ∀ a : ℝ, has_empty_solution_set a ↔ -3 < a ∧ a ≤ 1 := by sorry

end empty_solution_set_iff_a_in_range_l2975_297529


namespace inequality_proof_l2975_297563

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  1 / (a + b + c) + 1 / (b + c + d) + 1 / (c + d + a) + 1 / (a + b + d) ≥ 4 / (3 * (a + b + c + d)) :=
by sorry

end inequality_proof_l2975_297563


namespace vector_dot_product_l2975_297581

/-- Given two vectors a and b in ℝ², prove that their dot product is -29 -/
theorem vector_dot_product (a b : ℝ × ℝ) 
  (h1 : a.1 + b.1 = 2 ∧ a.2 + b.2 = -4)
  (h2 : 3 * a.1 - b.1 = -10 ∧ 3 * a.2 - b.2 = 16) :
  a.1 * b.1 + a.2 * b.2 = -29 := by
  sorry

end vector_dot_product_l2975_297581


namespace vacation_cost_per_person_l2975_297504

theorem vacation_cost_per_person (num_people : ℕ) (airbnb_cost car_cost : ℚ) :
  num_people = 8 ∧ airbnb_cost = 3200 ∧ car_cost = 800 →
  (airbnb_cost + car_cost) / num_people = 500 := by
  sorry

end vacation_cost_per_person_l2975_297504


namespace tangent_circle_equations_l2975_297515

def given_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y - 4 = 0

def tangent_line (y : ℝ) : Prop :=
  y = 0

def is_tangent_circles (x1 y1 r1 x2 y2 r2 : ℝ) : Prop :=
  (x1 - x2)^2 + (y1 - y2)^2 = (r1 + r2)^2 ∨ (x1 - x2)^2 + (y1 - y2)^2 = (r1 - r2)^2

def is_tangent_circle_line (x y r : ℝ) : Prop :=
  y = r ∨ y = -r

theorem tangent_circle_equations :
  ∃ (a b c d : ℝ),
    (∀ x y : ℝ, ((x - (2 + 2 * Real.sqrt 10))^2 + (y - 4)^2 = 16 ↔ (x - a)^2 + (y - 4)^2 = 16) ∧
                ((x - (2 - 2 * Real.sqrt 10))^2 + (y - 4)^2 = 16 ↔ (x - b)^2 + (y - 4)^2 = 16) ∧
                ((x - (2 + 2 * Real.sqrt 6))^2 + (y + 4)^2 = 16 ↔ (x - c)^2 + (y + 4)^2 = 16) ∧
                ((x - (2 - 2 * Real.sqrt 6))^2 + (y + 4)^2 = 16 ↔ (x - d)^2 + (y + 4)^2 = 16)) ∧
    (∀ x y : ℝ, given_circle x y →
      (is_tangent_circles x y 3 a 4 4 ∧ is_tangent_circle_line a 4 4) ∨
      (is_tangent_circles x y 3 b 4 4 ∧ is_tangent_circle_line b 4 4) ∨
      (is_tangent_circles x y 3 c (-4) 4 ∧ is_tangent_circle_line c (-4) 4) ∨
      (is_tangent_circles x y 3 d (-4) 4 ∧ is_tangent_circle_line d (-4) 4)) ∧
    (∀ x y : ℝ, tangent_line y →
      ((x - a)^2 + (y - 4)^2 = 16 ∨
       (x - b)^2 + (y - 4)^2 = 16 ∨
       (x - c)^2 + (y + 4)^2 = 16 ∨
       (x - d)^2 + (y + 4)^2 = 16)) := by
  sorry

end tangent_circle_equations_l2975_297515


namespace tangent_parallel_to_x_axis_y_coordinates_l2975_297550

-- Define the function
def f (x : ℝ) : ℝ := x * (x - 4)^3

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 4 * (x - 4)^2 * (x - 1)

-- Theorem statement
theorem tangent_parallel_to_x_axis :
  ∀ x : ℝ, f' x = 0 ↔ x = 4 ∨ x = 1 :=
sorry

-- Verify the y-coordinates
theorem y_coordinates :
  f 4 = 0 ∧ f 1 = -27 :=
sorry

end tangent_parallel_to_x_axis_y_coordinates_l2975_297550


namespace stars_per_bottle_l2975_297527

/-- Given that Shiela prepared 45 paper stars and has 9 classmates,
    prove that the number of stars per bottle is 5. -/
theorem stars_per_bottle (total_stars : ℕ) (num_classmates : ℕ) 
  (h1 : total_stars = 45) (h2 : num_classmates = 9) :
  total_stars / num_classmates = 5 := by
  sorry

end stars_per_bottle_l2975_297527


namespace smaller_factor_of_5610_l2975_297547

theorem smaller_factor_of_5610 (a b : Nat) : 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 5610 → 
  min a b = 34 := by
sorry

end smaller_factor_of_5610_l2975_297547


namespace pi_greater_than_314_l2975_297500

theorem pi_greater_than_314 : π > 3.14 := by
  sorry

end pi_greater_than_314_l2975_297500


namespace smallest_n_with_four_pairs_l2975_297502

/-- g(n) returns the number of distinct ordered pairs of positive integers (a, b) such that a^2 + b^2 = n -/
def g (n : ℕ) : ℕ := (Finset.filter (fun p : ℕ × ℕ => p.1^2 + p.2^2 = n ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range n) (Finset.range n))).card

/-- 65 is the smallest positive integer n for which g(n) = 4 -/
theorem smallest_n_with_four_pairs : (∀ m : ℕ, 0 < m → m < 65 → g m ≠ 4) ∧ g 65 = 4 := by sorry

end smallest_n_with_four_pairs_l2975_297502


namespace arithmetic_sequence_properties_l2975_297539

/-- Arithmetic sequence with first term 20 and common difference -2 -/
def arithmetic_sequence (n : ℕ) : ℤ := 20 - 2 * (n - 1)

/-- Sum of first n terms of the arithmetic sequence -/
def sum_arithmetic_sequence (n : ℕ) : ℤ := -n^2 + 21*n

theorem arithmetic_sequence_properties :
  ∀ n : ℕ,
  (arithmetic_sequence n = -2*n + 22) ∧
  (sum_arithmetic_sequence n = -n^2 + 21*n) ∧
  (∀ k : ℕ, sum_arithmetic_sequence k ≤ 110) ∧
  (∃ m : ℕ, sum_arithmetic_sequence m = 110) := by
  sorry

end arithmetic_sequence_properties_l2975_297539


namespace complex_arithmetic_l2975_297564

theorem complex_arithmetic (B N T Q : ℂ) : 
  B = 5 - 2*I ∧ N = -5 + 2*I ∧ T = 3*I ∧ Q = 3 →
  B - N + T - Q = 7 - I :=
by sorry

end complex_arithmetic_l2975_297564


namespace perpendicular_vectors_x_value_l2975_297593

/-- Given two vectors a and b in ℝ², where a = (3, -2) and b = (x, 1),
    prove that if a ⊥ b, then x = 2/3 -/
theorem perpendicular_vectors_x_value (x : ℝ) : 
  let a : Fin 2 → ℝ := ![3, -2]
  let b : Fin 2 → ℝ := ![x, 1]
  (∀ i j, i ≠ j → a i * a j + b i * b j = 0) →
  x = 2/3 :=
by sorry

end perpendicular_vectors_x_value_l2975_297593


namespace ski_race_minimum_participants_l2975_297598

theorem ski_race_minimum_participants : ∀ n : ℕ,
  (∃ k : ℕ, 
    (k : ℝ) / n ≥ 0.035 ∧ 
    (k : ℝ) / n ≤ 0.045 ∧ 
    k > 0) →
  n ≥ 23 :=
by sorry

end ski_race_minimum_participants_l2975_297598


namespace jenny_games_against_mark_l2975_297512

theorem jenny_games_against_mark (mark_wins : ℕ) (jenny_wins : ℕ) 
  (h1 : mark_wins = 1)
  (h2 : jenny_wins = 14) :
  ∃ m : ℕ,
    (m - mark_wins) + (2 * m - (3/4 * 2 * m)) = jenny_wins ∧ 
    m = 30 := by
  sorry

end jenny_games_against_mark_l2975_297512


namespace sol_earnings_l2975_297578

/-- Calculates the earnings from candy bar sales over a week -/
def candy_bar_earnings (initial_sales : ℕ) (daily_increase : ℕ) (days : ℕ) (price_cents : ℕ) : ℚ :=
  let total_sales := (List.range days).map (fun i => initial_sales + i * daily_increase) |>.sum
  (total_sales * price_cents : ℕ) / 100

/-- Theorem: Sol's earnings from candy bar sales over a week -/
theorem sol_earnings : candy_bar_earnings 10 4 6 10 = 12 := by
  sorry

end sol_earnings_l2975_297578


namespace johns_total_time_l2975_297507

theorem johns_total_time (exploring_time writing_book_time : ℝ) : 
  exploring_time = 3 →
  writing_book_time = 0.5 →
  exploring_time + (exploring_time / 2) + writing_book_time = 5 := by
  sorry

end johns_total_time_l2975_297507


namespace sams_initial_dimes_l2975_297531

/-- The problem of determining Sam's initial number of dimes -/
theorem sams_initial_dimes : 
  ∀ (initial final given : ℕ), 
  given = 7 →                 -- Sam's dad gave him 7 dimes
  final = 16 →                -- After receiving the dimes, Sam has 16 dimes
  final = initial + given →   -- The final amount is the sum of initial and given
  initial = 9 :=              -- Prove that the initial amount was 9 dimes
by sorry

end sams_initial_dimes_l2975_297531


namespace misha_second_round_score_l2975_297544

/-- Represents the points scored in each round of dart throwing -/
structure DartScores where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Defines the conditions of Misha's dart game -/
def valid_dart_game (scores : DartScores) : Prop :=
  scores.second = 2 * scores.first ∧
  scores.third = (3 * scores.second) / 2 ∧
  scores.first ≥ 24 ∧
  scores.third ≤ 72

/-- Theorem stating that Misha must have scored 48 points in the second round -/
theorem misha_second_round_score (scores : DartScores) 
  (h : valid_dart_game scores) : scores.second = 48 := by
  sorry

end misha_second_round_score_l2975_297544


namespace guppies_theorem_l2975_297548

def guppies_problem (haylee jose charliz nicolai : ℕ) : Prop :=
  haylee = 3 * 12 ∧
  jose = haylee / 2 ∧
  charliz = jose / 3 ∧
  nicolai = 4 * charliz ∧
  haylee + jose + charliz + nicolai = 84

theorem guppies_theorem : ∃ haylee jose charliz nicolai : ℕ, guppies_problem haylee jose charliz nicolai :=
sorry

end guppies_theorem_l2975_297548


namespace tangent_point_and_perpendicular_line_l2975_297523

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Define the condition for the tangent line being parallel to 4x - y - 1 = 0
def tangent_parallel (x : ℝ) : Prop := f' x = 4

-- Define the third quadrant condition
def third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

-- Main theorem
theorem tangent_point_and_perpendicular_line :
  ∃ (x₀ y₀ : ℝ), 
    y₀ = f x₀ ∧ 
    tangent_parallel x₀ ∧ 
    third_quadrant x₀ y₀ ∧ 
    x₀ = -1 ∧ 
    y₀ = -4 ∧ 
    ∀ (x y : ℝ), x + 4*y + 17 = 0 ↔ y - y₀ = -(1/4) * (x - x₀) :=
by sorry

end tangent_point_and_perpendicular_line_l2975_297523


namespace dance_troupe_arrangement_l2975_297596

theorem dance_troupe_arrangement (n : ℕ) : n > 0 ∧ 
  6 ∣ n ∧ 9 ∣ n ∧ 12 ∣ n ∧ 5 ∣ n → n ≥ 180 :=
by sorry

end dance_troupe_arrangement_l2975_297596


namespace beverlys_bottle_caps_l2975_297575

def bottle_caps_per_box : ℝ := 35.0
def total_bottle_caps : ℕ := 245

theorem beverlys_bottle_caps :
  (total_bottle_caps : ℝ) / bottle_caps_per_box = 7 :=
sorry

end beverlys_bottle_caps_l2975_297575


namespace boots_discounted_price_l2975_297501

/-- Calculates the discounted price of an item given its original price and discount percentage. -/
def discountedPrice (originalPrice : ℚ) (discountPercentage : ℚ) : ℚ :=
  originalPrice * (1 - discountPercentage / 100)

/-- Proves that the discounted price of boots with an original price of $90 and a 20% discount is $72. -/
theorem boots_discounted_price :
  discountedPrice 90 20 = 72 := by
  sorry

end boots_discounted_price_l2975_297501


namespace min_distance_between_points_l2975_297540

/-- Given four points P, Q, R, and S in a metric space, with distances PQ = 12, QR = 5, and RS = 8,
    the minimum possible distance between P and S is 1. -/
theorem min_distance_between_points (X : Type*) [MetricSpace X] 
  (P Q R S : X) 
  (h_PQ : dist P Q = 12)
  (h_QR : dist Q R = 5)
  (h_RS : dist R S = 8) : 
  ∃ (configuration : X → X), dist (configuration P) (configuration S) = 1 ∧ 
    (∀ (config : X → X), dist (config P) (config S) ≥ 1) :=
sorry

end min_distance_between_points_l2975_297540


namespace prism_no_circular_section_l2975_297520

/-- A solid object that can be cut by a plane -/
class Solid :=
  (can_produce_circular_section : Bool)

/-- A cone is a solid that can produce a circular cross-section -/
def Cone : Solid :=
  { can_produce_circular_section := true }

/-- A cylinder is a solid that can produce a circular cross-section -/
def Cylinder : Solid :=
  { can_produce_circular_section := true }

/-- A sphere is a solid that can produce a circular cross-section -/
def Sphere : Solid :=
  { can_produce_circular_section := true }

/-- A prism is a solid that cannot produce a circular cross-section -/
def Prism : Solid :=
  { can_produce_circular_section := false }

/-- Theorem: Among cones, cylinders, spheres, and prisms, only a prism cannot produce a circular cross-section -/
theorem prism_no_circular_section :
  ∀ s : Solid, s.can_produce_circular_section = false → s = Prism :=
by sorry

end prism_no_circular_section_l2975_297520


namespace triple_hash_45_l2975_297534

-- Define the # operation
def hash (N : ℝ) : ℝ := 0.4 * N + 3

-- State the theorem
theorem triple_hash_45 : hash (hash (hash 45)) = 7.56 := by
  sorry

end triple_hash_45_l2975_297534


namespace floor_of_negative_three_point_seven_l2975_297533

theorem floor_of_negative_three_point_seven :
  ⌊(-3.7 : ℝ)⌋ = -4 := by sorry

end floor_of_negative_three_point_seven_l2975_297533


namespace jogger_speed_l2975_297545

/-- The speed of the jogger in km/hr given the following conditions:
  1. The jogger is 200 m ahead of the train engine
  2. The train is 210 m long
  3. The train is running at 45 km/hr
  4. The train and jogger are moving in the same direction
  5. The train passes the jogger in 41 seconds
-/
theorem jogger_speed : ℝ := by
  -- Define the given conditions
  let initial_distance : ℝ := 200 -- meters
  let train_length : ℝ := 210 -- meters
  let train_speed : ℝ := 45 -- km/hr
  let passing_time : ℝ := 41 -- seconds

  -- Define the jogger's speed as a variable
  let jogger_speed : ℝ := 9 -- km/hr

  sorry -- Proof omitted

#check jogger_speed

end jogger_speed_l2975_297545


namespace remainder_problem_l2975_297549

theorem remainder_problem (k : ℕ) 
  (h1 : k % 6 = 5) 
  (h2 : k % 7 = 3) 
  (h3 : k < 41) : 
  k % 5 = 2 := by
  sorry

end remainder_problem_l2975_297549


namespace initial_alcohol_content_75_percent_l2975_297535

/-- Represents the alcohol content of a solution as a real number between 0 and 1 -/
def AlcoholContent := { x : ℝ // 0 ≤ x ∧ x ≤ 1 }

/-- Proves that the initial alcohol content was 75% given the problem conditions -/
theorem initial_alcohol_content_75_percent 
  (initial_volume : ℝ) 
  (drained_volume : ℝ) 
  (added_content : AlcoholContent) 
  (final_content : AlcoholContent) 
  (h1 : initial_volume = 1)
  (h2 : drained_volume = 0.4)
  (h3 : added_content.val = 0.5)
  (h4 : final_content.val = 0.65) :
  ∃ (initial_content : AlcoholContent), 
    initial_content.val = 0.75 ∧
    (initial_volume - drained_volume) * initial_content.val + 
    drained_volume * added_content.val = 
    initial_volume * final_content.val :=
by sorry


end initial_alcohol_content_75_percent_l2975_297535


namespace no_solution_exists_l2975_297511

theorem no_solution_exists : 
  ¬∃ (a b : ℕ+), a * b + 82 = 25 * Nat.lcm a b + 15 * Nat.gcd a b :=
sorry

end no_solution_exists_l2975_297511


namespace parallel_vectors_l2975_297556

/-- Given vectors in R² -/
def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-1, 2)
def c : ℝ × ℝ := (1, 6)

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), v.1 * w.2 = t * v.2 * w.1

/-- The main theorem -/
theorem parallel_vectors (k : ℝ) :
  are_parallel (a.1 + k * c.1, a.2 + k * c.2) (a.1 + b.1, a.2 + b.2) ↔ k = 1 := by
  sorry

end parallel_vectors_l2975_297556


namespace arithmetic_computation_l2975_297517

theorem arithmetic_computation : -9 * 5 - (-7 * -2) + (-11 * -4) = -15 := by
  sorry

end arithmetic_computation_l2975_297517


namespace common_difference_is_three_l2975_297594

/-- Arithmetic sequence with 10 terms -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, n < 9 → a (n + 1) = a n + d

/-- Sum of odd terms is 15 -/
def sum_odd_terms (a : ℕ → ℝ) : Prop :=
  a 1 + a 3 + a 5 + a 7 + a 9 = 15

/-- Sum of even terms is 30 -/
def sum_even_terms (a : ℕ → ℝ) : Prop :=
  a 2 + a 4 + a 6 + a 8 + a 10 = 30

theorem common_difference_is_three (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : sum_odd_terms a) 
  (h3 : sum_even_terms a) : 
  ∃ d : ℝ, d = 3 ∧ ∀ n : ℕ, n < 9 → a (n + 1) = a n + d :=
sorry

end common_difference_is_three_l2975_297594


namespace second_bag_kernels_l2975_297582

/-- Represents the number of kernels in a bag of popcorn -/
structure PopcornBag where
  total : ℕ
  popped : ℕ

/-- Calculates the percentage of popped kernels in a bag -/
def poppedPercentage (bag : PopcornBag) : ℚ :=
  (bag.popped : ℚ) / (bag.total : ℚ) * 100

theorem second_bag_kernels (bag1 bag2 bag3 : PopcornBag)
  (h1 : bag1.total = 75 ∧ bag1.popped = 60)
  (h2 : bag2.popped = 42)
  (h3 : bag3.total = 100 ∧ bag3.popped = 82)
  (h_avg : (poppedPercentage bag1 + poppedPercentage bag2 + poppedPercentage bag3) / 3 = 82) :
  bag2.total = 50 := by
  sorry


end second_bag_kernels_l2975_297582


namespace circle_radius_from_area_circumference_ratio_l2975_297546

theorem circle_radius_from_area_circumference_ratio 
  (M N : ℝ) (h : M / N = 25) : 
  ∃ (r : ℝ), r > 0 ∧ M = π * r^2 ∧ N = 2 * π * r ∧ r = 50 :=
by sorry

end circle_radius_from_area_circumference_ratio_l2975_297546


namespace range_of_m_l2975_297542

theorem range_of_m (m : ℝ) : 
  (∀ x y : ℝ, x > 0 → y > 0 → (2 * y / x + 9 * x / (2 * y) ≥ m^2 + m)) → 
  (-3 ≤ m ∧ m ≤ 2) :=
by sorry

end range_of_m_l2975_297542


namespace vector_magnitude_proof_l2975_297570

theorem vector_magnitude_proof (a b : ℝ × ℝ) : 
  a = (-1, 2) → b = (1, 3) → ‖(2 • a) - b‖ = Real.sqrt 10 := by
  sorry

end vector_magnitude_proof_l2975_297570


namespace acute_angles_sum_l2975_297503

theorem acute_angles_sum (a b : Real) : 
  0 < a ∧ a < π / 2 →
  0 < b ∧ b < π / 2 →
  3 * (Real.sin a) ^ 2 + 2 * (Real.sin b) ^ 2 = 1 →
  3 * Real.sin (2 * a) - 2 * Real.sin (2 * b) = 0 →
  a + 2 * b = π / 2 := by
  sorry

end acute_angles_sum_l2975_297503


namespace total_annual_insurance_cost_l2975_297565

def car_insurance_quarterly : ℕ := 378
def home_insurance_monthly : ℕ := 125
def health_insurance_annual : ℕ := 5045

theorem total_annual_insurance_cost :
  car_insurance_quarterly * 4 + home_insurance_monthly * 12 + health_insurance_annual = 8057 := by
  sorry

end total_annual_insurance_cost_l2975_297565
