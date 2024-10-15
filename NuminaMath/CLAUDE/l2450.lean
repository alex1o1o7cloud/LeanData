import Mathlib

namespace NUMINAMATH_CALUDE_largest_value_at_negative_one_l2450_245097

/-- A monic cubic polynomial with non-negative real roots and f(0) = -64 -/
def MonicCubicPolynomial : Type := 
  {f : ℝ → ℝ // ∃ (r₁ r₂ r₃ : ℝ), (∀ x, f x = (x - r₁) * (x - r₂) * (x - r₃)) ∧ 
                                  (r₁ ≥ 0 ∧ r₂ ≥ 0 ∧ r₃ ≥ 0) ∧
                                  (f 0 = -64)}

/-- The largest possible value of f(-1) for a MonicCubicPolynomial is -125 -/
theorem largest_value_at_negative_one (f : MonicCubicPolynomial) : 
  f.val (-1) ≤ -125 := by
  sorry

end NUMINAMATH_CALUDE_largest_value_at_negative_one_l2450_245097


namespace NUMINAMATH_CALUDE_horners_method_polynomial_transformation_l2450_245053

theorem horners_method_polynomial_transformation (x : ℝ) :
  6 * x^3 + 5 * x^2 + 4 * x + 3 = x * (x * (6 * x + 5) + 4) + 3 := by
  sorry

end NUMINAMATH_CALUDE_horners_method_polynomial_transformation_l2450_245053


namespace NUMINAMATH_CALUDE_bus_capacity_l2450_245076

/-- The number of rows in a bus -/
def rows : ℕ := 9

/-- The number of children that can be accommodated in each row -/
def children_per_row : ℕ := 4

/-- The total number of children a bus can accommodate -/
def total_children : ℕ := rows * children_per_row

theorem bus_capacity : total_children = 36 := by
  sorry

end NUMINAMATH_CALUDE_bus_capacity_l2450_245076


namespace NUMINAMATH_CALUDE_vector_at_zero_given_two_points_l2450_245075

/-- A parameterized line in 2D space -/
structure ParameterizedLine where
  -- The vector on the line at parameter t
  vector : ℝ → ℝ × ℝ

theorem vector_at_zero_given_two_points (L : ParameterizedLine) :
  L.vector 1 = (2, 3) →
  L.vector 4 = (8, -5) →
  L.vector 0 = (0, 17/3) := by
  sorry

end NUMINAMATH_CALUDE_vector_at_zero_given_two_points_l2450_245075


namespace NUMINAMATH_CALUDE_rowing_speed_in_still_water_l2450_245036

/-- Represents the rowing scenario with upstream and downstream times, current speed, and still water speed. -/
structure RowingScenario where
  upstream_time : ℝ
  downstream_time : ℝ
  current_speed : ℝ
  still_water_speed : ℝ

/-- Theorem stating that given the conditions, the man's rowing speed in still water is 3.6 km/hr. -/
theorem rowing_speed_in_still_water 
  (scenario : RowingScenario)
  (h1 : scenario.upstream_time = 2 * scenario.downstream_time)
  (h2 : scenario.current_speed = 1.2)
  : scenario.still_water_speed = 3.6 :=
by sorry

end NUMINAMATH_CALUDE_rowing_speed_in_still_water_l2450_245036


namespace NUMINAMATH_CALUDE_jogging_distance_l2450_245039

/-- Calculates the distance traveled given a constant rate and time. -/
def distance (rate : ℝ) (time : ℝ) : ℝ := rate * time

/-- Proves that jogging at 4 miles per hour for 2 hours results in a distance of 8 miles. -/
theorem jogging_distance : distance 4 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_jogging_distance_l2450_245039


namespace NUMINAMATH_CALUDE_mall_promotion_max_purchase_l2450_245047

/-- Calculates the maximum value of goods that can be purchased given a cashback rule and initial amount --/
def max_purchase_value (cashback_amount : ℕ) (cashback_threshold : ℕ) (initial_amount : ℕ) : ℕ :=
  sorry

/-- The maximum value of goods that can be purchased given the specific conditions --/
theorem mall_promotion_max_purchase :
  max_purchase_value 40 200 650 = 770 :=
sorry

end NUMINAMATH_CALUDE_mall_promotion_max_purchase_l2450_245047


namespace NUMINAMATH_CALUDE_waiting_by_stump_is_random_waiting_by_stump_unique_random_l2450_245009

-- Define the type for idioms
inductive Idiom
  | FishingForMoon
  | CastlesInAir
  | WaitingByStump
  | CatchingTurtle

-- Define a property for idioms
def describesRandomEvent (i : Idiom) : Prop :=
  match i with
  | Idiom.FishingForMoon => False
  | Idiom.CastlesInAir => False
  | Idiom.WaitingByStump => True
  | Idiom.CatchingTurtle => False

-- Theorem stating that "Waiting by a stump for a hare" describes a random event
theorem waiting_by_stump_is_random :
  describesRandomEvent Idiom.WaitingByStump :=
by sorry

-- Theorem stating that "Waiting by a stump for a hare" is the only idiom
-- among the given options that describes a random event
theorem waiting_by_stump_unique_random :
  ∀ (i : Idiom), describesRandomEvent i ↔ i = Idiom.WaitingByStump :=
by sorry

end NUMINAMATH_CALUDE_waiting_by_stump_is_random_waiting_by_stump_unique_random_l2450_245009


namespace NUMINAMATH_CALUDE_fourth_derivative_y_l2450_245020

noncomputable def y (x : ℝ) : ℝ := (3 * x - 7) * (3 : ℝ)^(-x)

theorem fourth_derivative_y (x : ℝ) :
  (deriv^[4] y) x = (7 * Real.log 3 - 12 - 3 * Real.log 3 * x) * (Real.log 3)^3 * (3 : ℝ)^(-x) :=
by sorry

end NUMINAMATH_CALUDE_fourth_derivative_y_l2450_245020


namespace NUMINAMATH_CALUDE_circle_area_sum_l2450_245078

/-- The sum of the areas of an infinite sequence of circles, where the radius of the first circle
    is 1 and each subsequent circle's radius is 2/3 of the previous one, is equal to 9π/5. -/
theorem circle_area_sum : 
  let radius : ℕ → ℝ := λ n => (2/3)^(n-1)
  let area : ℕ → ℝ := λ n => π * (radius n)^2
  (∑' n, area n) = 9*π/5 := by sorry

end NUMINAMATH_CALUDE_circle_area_sum_l2450_245078


namespace NUMINAMATH_CALUDE_earnings_left_over_l2450_245041

/-- Calculates the percentage of earnings left over after spending on rent and dishwasher -/
theorem earnings_left_over (rent_percentage : ℝ) (dishwasher_discount : ℝ) : 
  rent_percentage = 25 →
  dishwasher_discount = 10 →
  100 - (rent_percentage + (rent_percentage - rent_percentage * dishwasher_discount / 100)) = 52.5 := by
  sorry


end NUMINAMATH_CALUDE_earnings_left_over_l2450_245041


namespace NUMINAMATH_CALUDE_vectors_form_basis_l2450_245032

def e₁ : ℝ × ℝ := (-1, 2)
def e₂ : ℝ × ℝ := (5, 7)

theorem vectors_form_basis : 
  LinearIndependent ℝ (![e₁, e₂] : Fin 2 → ℝ × ℝ) :=
sorry

end NUMINAMATH_CALUDE_vectors_form_basis_l2450_245032


namespace NUMINAMATH_CALUDE_shirt_sales_price_solution_l2450_245042

/-- Represents the selling price and profit calculation for shirts -/
def ShirtSales (x : ℝ) : Prop :=
  let cost_price : ℝ := 80
  let initial_daily_sales : ℝ := 30
  let price_reduction : ℝ := 130 - x
  let additional_sales : ℝ := 2 * price_reduction
  let total_daily_sales : ℝ := initial_daily_sales + additional_sales
  let profit_per_shirt : ℝ := x - cost_price
  let daily_profit : ℝ := profit_per_shirt * total_daily_sales
  daily_profit = 2000

theorem shirt_sales_price_solution :
  ∃ x : ℝ, ShirtSales x ∧ (x = 105 ∨ x = 120) :=
sorry

end NUMINAMATH_CALUDE_shirt_sales_price_solution_l2450_245042


namespace NUMINAMATH_CALUDE_childrens_ticket_cost_l2450_245044

/-- Prove that the cost of a children's ticket is $4.50 -/
theorem childrens_ticket_cost
  (adult_ticket_cost : ℝ)
  (total_tickets : ℕ)
  (total_revenue : ℝ)
  (childrens_tickets : ℕ)
  (h1 : adult_ticket_cost = 6)
  (h2 : total_tickets = 400)
  (h3 : total_revenue = 2100)
  (h4 : childrens_tickets = 200) :
  ∃ (childrens_ticket_cost : ℝ),
    childrens_ticket_cost * childrens_tickets +
    adult_ticket_cost * (total_tickets - childrens_tickets) = total_revenue ∧
    childrens_ticket_cost = 4.5 :=
by
  sorry


end NUMINAMATH_CALUDE_childrens_ticket_cost_l2450_245044


namespace NUMINAMATH_CALUDE_complex_equation_to_parabola_l2450_245081

/-- The set of points (x, y) satisfying the complex equation is equivalent to a parabola with two holes -/
theorem complex_equation_to_parabola (x y : ℝ) :
  (Complex.I + x^2 - 2*x + 2*y*Complex.I = 
   (y - 1 : ℂ) + ((4*y^2 - 1)/(2*y - 1) : ℝ)*Complex.I) ↔ 
  (y = (x - 1)^2 ∧ y ≠ (1/2 : ℝ)) :=
sorry

end NUMINAMATH_CALUDE_complex_equation_to_parabola_l2450_245081


namespace NUMINAMATH_CALUDE_positive_real_inequality_l2450_245017

theorem positive_real_inequality (x : ℝ) (h : x > 0) : x + 1/x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequality_l2450_245017


namespace NUMINAMATH_CALUDE_small_square_area_l2450_245056

theorem small_square_area (n : ℕ) : n > 0 → (
  let outer_square_area : ℝ := 1
  let small_square_area : ℝ := 1 / 1985
  let side_length : ℝ := 1 / n
  let diagonal_segment : ℝ := (n - 1) / n
  let small_square_side : ℝ := diagonal_segment / Real.sqrt 1985
  small_square_side * small_square_side = small_square_area
) ↔ n = 32 := by sorry

end NUMINAMATH_CALUDE_small_square_area_l2450_245056


namespace NUMINAMATH_CALUDE_base_five_product_l2450_245025

/-- Represents a number in base 5 --/
def BaseFive : Type := ℕ

/-- Converts a natural number to its base 5 representation --/
def toBaseFive (n : ℕ) : BaseFive := sorry

/-- Multiplies two numbers in base 5 --/
def multBaseFive (a b : BaseFive) : BaseFive := sorry

/-- Adds two numbers in base 5 --/
def addBaseFive (a b : BaseFive) : BaseFive := sorry

theorem base_five_product :
  let a : BaseFive := toBaseFive 121
  let b : BaseFive := toBaseFive 11
  let c : BaseFive := toBaseFive 1331
  multBaseFive a b = c := by sorry

end NUMINAMATH_CALUDE_base_five_product_l2450_245025


namespace NUMINAMATH_CALUDE_edge_sum_is_112_l2450_245033

/-- A rectangular solid with specific properties -/
structure RectangularSolid where
  x : ℝ
  d : ℝ
  volume_eq : x^3 * (d + 1)^3 = 512
  surface_area_eq : 2 * (x^2 * (d + 1) + x^2 * (d + 1)^2 + x^2 * (d + 1)^3) = 448

/-- The sum of the lengths of all edges of the rectangular solid -/
def edge_sum (solid : RectangularSolid) : ℝ :=
  4 * (solid.x + solid.x * (solid.d + 1) + solid.x * (solid.d + 1)^2)

/-- Theorem stating that the sum of the lengths of all edges is 112 -/
theorem edge_sum_is_112 (solid : RectangularSolid) : edge_sum solid = 112 := by
  sorry

#check edge_sum_is_112

end NUMINAMATH_CALUDE_edge_sum_is_112_l2450_245033


namespace NUMINAMATH_CALUDE_special_polyhedron_body_diagonals_l2450_245055

/-- A convex polyhedron with specific face composition -/
structure SpecialPolyhedron where
  /-- The polyhedron is convex -/
  is_convex : Bool
  /-- Number of square faces -/
  num_squares : Nat
  /-- Number of regular hexagon faces -/
  num_hexagons : Nat
  /-- Number of regular octagon faces -/
  num_octagons : Nat
  /-- At each vertex, a square, a hexagon, and an octagon meet -/
  vertex_composition : Bool
  /-- The surface is composed of exactly 12 squares, 8 hexagons, and 6 octagons -/
  face_composition : num_squares = 12 ∧ num_hexagons = 8 ∧ num_octagons = 6

/-- The number of body diagonals in the special polyhedron -/
def num_body_diagonals (p : SpecialPolyhedron) : Nat :=
  sorry

/-- Theorem: The number of body diagonals in the special polyhedron is 840 -/
theorem special_polyhedron_body_diagonals (p : SpecialPolyhedron) : 
  num_body_diagonals p = 840 := by
  sorry

end NUMINAMATH_CALUDE_special_polyhedron_body_diagonals_l2450_245055


namespace NUMINAMATH_CALUDE_impossible_perpendicular_intersection_l2450_245096

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (intersect : Line → Line → Prop)
variable (coincident : Line → Line → Prop)

-- Define the planes and lines
variable (α : Plane)
variable (a b : Line)

-- State the theorem
theorem impossible_perpendicular_intersection 
  (h1 : ¬ coincident a b)
  (h2 : perpendicular a α)
  (h3 : intersect a b) :
  ¬ (perpendicular b α) :=
sorry

end NUMINAMATH_CALUDE_impossible_perpendicular_intersection_l2450_245096


namespace NUMINAMATH_CALUDE_gcd_2952_1386_l2450_245048

theorem gcd_2952_1386 : Nat.gcd 2952 1386 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2952_1386_l2450_245048


namespace NUMINAMATH_CALUDE_couples_after_dance_l2450_245087

/-- The number of initial couples at the ball. -/
def n : ℕ := 2018

/-- The function that determines the source area for a couple at minute i. -/
def s (i : ℕ) : ℕ := i % n + 1

/-- The function that determines the destination area for a couple at minute i. -/
def r (i : ℕ) : ℕ := (2 * i) % n + 1

/-- Predicate to determine if a couple in area k survives after t minutes. -/
def survives (k t : ℕ) : Prop := sorry

/-- The number of couples remaining after t minutes. -/
def remaining_couples (t : ℕ) : ℕ := sorry

/-- The main theorem stating that after n² minutes, 505 couples remain. -/
theorem couples_after_dance : remaining_couples (n^2) = 505 := by sorry

end NUMINAMATH_CALUDE_couples_after_dance_l2450_245087


namespace NUMINAMATH_CALUDE_total_balloons_l2450_245029

theorem total_balloons (tom_balloons sara_balloons alex_balloons : ℕ) 
  (h1 : tom_balloons = 18) 
  (h2 : sara_balloons = 12) 
  (h3 : alex_balloons = 7) : 
  tom_balloons + sara_balloons + alex_balloons = 37 := by
  sorry

end NUMINAMATH_CALUDE_total_balloons_l2450_245029


namespace NUMINAMATH_CALUDE_sequence_increasing_iff_a0_eq_one_fifth_l2450_245080

/-- The sequence defined by a(n+1) = 2^n - 3*a(n) -/
def a : ℕ → ℝ → ℝ 
  | 0, a₀ => a₀
  | n + 1, a₀ => 2^n - 3 * a n a₀

/-- The sequence is increasing -/
def is_increasing (a₀ : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) a₀ > a n a₀

/-- Theorem: The sequence is increasing if and only if a₀ = 1/5 -/
theorem sequence_increasing_iff_a0_eq_one_fifth :
  ∀ a₀ : ℝ, is_increasing a₀ ↔ a₀ = 1/5 := by sorry

end NUMINAMATH_CALUDE_sequence_increasing_iff_a0_eq_one_fifth_l2450_245080


namespace NUMINAMATH_CALUDE_prime_sum_to_square_l2450_245011

theorem prime_sum_to_square (a b : ℕ) : 
  let P := (Nat.lcm a b / (a + 1)) + (Nat.lcm a b / (b + 1))
  Prime P → ∃ n : ℕ, 4 * P + 5 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_to_square_l2450_245011


namespace NUMINAMATH_CALUDE_base_b_121_is_perfect_square_l2450_245014

/-- Represents a number in base b as a list of digits --/
def BaseRepresentation (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun digit acc => acc * b + digit) 0

/-- Checks if a number is a perfect square --/
def IsPerfectSquare (n : Nat) : Prop :=
  ∃ m : Nat, m * m = n

theorem base_b_121_is_perfect_square (b : Nat) :
  (b > 2) ↔ IsPerfectSquare (BaseRepresentation [1, 2, 1] b) :=
by sorry

end NUMINAMATH_CALUDE_base_b_121_is_perfect_square_l2450_245014


namespace NUMINAMATH_CALUDE_set_conditions_l2450_245091

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

theorem set_conditions (m : ℝ) :
  (B m = ∅ ↔ m < 2) ∧
  (A ∩ B m = ∅ ↔ m > 4 ∨ m < 2) := by
  sorry

end NUMINAMATH_CALUDE_set_conditions_l2450_245091


namespace NUMINAMATH_CALUDE_a_worked_six_days_l2450_245019

/-- Represents the number of days worked by person a -/
def days_a : ℕ := sorry

/-- Represents the daily wage of person a -/
def wage_a : ℕ := sorry

/-- Represents the daily wage of person b -/
def wage_b : ℕ := sorry

/-- Represents the daily wage of person c -/
def wage_c : ℕ := sorry

/-- The theorem stating that person a worked for 6 days given the conditions -/
theorem a_worked_six_days :
  wage_c = 105 ∧
  wage_a / wage_b = 3 / 4 ∧
  wage_b / wage_c = 4 / 5 ∧
  days_a * wage_a + 9 * wage_b + 4 * wage_c = 1554 →
  days_a = 6 := by sorry

end NUMINAMATH_CALUDE_a_worked_six_days_l2450_245019


namespace NUMINAMATH_CALUDE_inverse_composition_equals_negative_one_l2450_245038

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x + 5

-- Define the inverse function f⁻¹
noncomputable def f_inv (y : ℝ) : ℝ := (y - 5) / 4

-- Theorem statement
theorem inverse_composition_equals_negative_one :
  f_inv (f_inv 9) = -1 := by sorry

end NUMINAMATH_CALUDE_inverse_composition_equals_negative_one_l2450_245038


namespace NUMINAMATH_CALUDE_max_two_scoop_sundaes_l2450_245023

theorem max_two_scoop_sundaes (n : ℕ) (h : n = 8) : 
  Nat.choose n 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_max_two_scoop_sundaes_l2450_245023


namespace NUMINAMATH_CALUDE_company_managers_count_l2450_245049

theorem company_managers_count 
  (num_associates : ℕ) 
  (avg_salary_managers : ℝ) 
  (avg_salary_associates : ℝ) 
  (avg_salary_company : ℝ) 
  (h1 : num_associates = 75)
  (h2 : avg_salary_managers = 90000)
  (h3 : avg_salary_associates = 30000)
  (h4 : avg_salary_company = 40000) :
  ∃ (num_managers : ℕ), 
    (num_managers : ℝ) * avg_salary_managers + (num_associates : ℝ) * avg_salary_associates = 
    ((num_managers : ℝ) + (num_associates : ℝ)) * avg_salary_company ∧ 
    num_managers = 15 := by
  sorry

end NUMINAMATH_CALUDE_company_managers_count_l2450_245049


namespace NUMINAMATH_CALUDE_exists_cycle_not_div_by_three_l2450_245094

/-- A graph is a structure with vertices and edges. -/
structure Graph (V : Type) :=
  (edges : V → V → Prop)

/-- The degree of a vertex in a graph is the number of edges incident to it. -/
def degree {V : Type} (G : Graph V) (v : V) : ℕ := sorry

/-- A path in a graph is a sequence of vertices where each consecutive pair is connected by an edge. -/
def is_path {V : Type} (G : Graph V) (path : List V) : Prop := sorry

/-- A cycle in a graph is a path that starts and ends at the same vertex. -/
def is_cycle {V : Type} (G : Graph V) (cycle : List V) : Prop := sorry

/-- The length of a path or cycle is the number of edges it contains. -/
def length {V : Type} (path : List V) : ℕ := path.length - 1

/-- Main theorem: In a graph where each vertex has degree at least 3, 
    there exists a cycle whose length is not divisible by 3. -/
theorem exists_cycle_not_div_by_three {V : Type} (G : Graph V) :
  (∀ v : V, degree G v ≥ 3) → 
  ∃ cycle : List V, is_cycle G cycle ∧ ¬(length cycle % 3 = 0) := by sorry

end NUMINAMATH_CALUDE_exists_cycle_not_div_by_three_l2450_245094


namespace NUMINAMATH_CALUDE_income_expenditure_ratio_l2450_245095

def income : ℕ := 19000
def savings : ℕ := 3800
def expenditure : ℕ := income - savings

theorem income_expenditure_ratio :
  (income : ℚ) / (expenditure : ℚ) = 5 / 4 := by sorry

end NUMINAMATH_CALUDE_income_expenditure_ratio_l2450_245095


namespace NUMINAMATH_CALUDE_f_min_at_neg_two_l2450_245034

/-- The polynomial f(x) = x^2 + 4x + 4 -/
def f (x : ℝ) : ℝ := x^2 + 4*x + 4

/-- The minimum value of f occurs at x = -2 -/
theorem f_min_at_neg_two :
  ∀ x : ℝ, f x ≥ f (-2) :=
by
  sorry

end NUMINAMATH_CALUDE_f_min_at_neg_two_l2450_245034


namespace NUMINAMATH_CALUDE_sum_of_squares_positive_l2450_245070

theorem sum_of_squares_positive (k : ℝ) (x₁ x₂ : ℝ) : 
  x₁ ≠ x₂ → 
  x₁^2 + 2*x₁ - k = 0 → 
  x₂^2 + 2*x₂ - k = 0 → 
  x₁^2 + x₂^2 > 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_positive_l2450_245070


namespace NUMINAMATH_CALUDE_gcd_2028_2100_l2450_245059

theorem gcd_2028_2100 : Nat.gcd 2028 2100 = 36 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2028_2100_l2450_245059


namespace NUMINAMATH_CALUDE_shaded_area_fraction_l2450_245066

/-- Represents a square with two internal unshaded squares -/
structure SquareWithInternalSquares where
  /-- Side length of the large square -/
  side : ℝ
  /-- Side length of the bottom-left unshaded square -/
  bottomLeftSide : ℝ
  /-- Side length of the top-right unshaded square -/
  topRightSide : ℝ
  /-- The bottom-left square's side is half of the large square's side -/
  bottomLeftHalf : bottomLeftSide = side / 2
  /-- The top-right square's diagonal is one-third of the large square's diagonal -/
  topRightThird : topRightSide * Real.sqrt 2 = side * Real.sqrt 2 / 3

/-- The fraction of the shaded area in a square with two internal unshaded squares is 19/36 -/
theorem shaded_area_fraction (s : SquareWithInternalSquares) :
  (s.side^2 - s.bottomLeftSide^2 - s.topRightSide^2) / s.side^2 = 19 / 36 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_fraction_l2450_245066


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2450_245065

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | 1 < x ∧ x ≤ 3}
def B : Set ℝ := {x | x > 2}

-- State the theorem
theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2450_245065


namespace NUMINAMATH_CALUDE_unique_satisfying_function_l2450_245083

/-- A function satisfying the given functional equation. -/
def SatisfyingFunction (f : ℤ → ℤ) : Prop :=
  ∀ m n : ℤ, f (m + f (f n)) = -f (f (m + 1)) - n

/-- The theorem stating that the only function satisfying the equation is f(p) = 1 - p. -/
theorem unique_satisfying_function :
  ∀ f : ℤ → ℤ, SatisfyingFunction f ↔ (∀ p : ℤ, f p = 1 - p) :=
sorry

end NUMINAMATH_CALUDE_unique_satisfying_function_l2450_245083


namespace NUMINAMATH_CALUDE_opposite_of_three_l2450_245024

theorem opposite_of_three (x : ℝ) : -x = 3 → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_three_l2450_245024


namespace NUMINAMATH_CALUDE_normal_distribution_std_dev_l2450_245004

/-- Represents a normal distribution --/
structure NormalDistribution where
  mean : ℝ
  stdDev : ℝ

/-- The value that is exactly k standard deviations from the mean --/
def valueAtStdDev (d : NormalDistribution) (k : ℝ) : ℝ :=
  d.mean + k * d.stdDev

theorem normal_distribution_std_dev (d : NormalDistribution) :
  d.mean = 15 ∧ valueAtStdDev d (-2) = 12 → d.stdDev = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_normal_distribution_std_dev_l2450_245004


namespace NUMINAMATH_CALUDE_systematic_sampling_elimination_l2450_245077

theorem systematic_sampling_elimination (population : Nat) (sample_size : Nat) 
    (h1 : population = 1252) 
    (h2 : sample_size = 50) : 
  population % sample_size = 2 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_elimination_l2450_245077


namespace NUMINAMATH_CALUDE_same_color_probability_value_l2450_245007

/-- The number of green balls in the bag -/
def green_balls : ℕ := 8

/-- The number of red balls in the bag -/
def red_balls : ℕ := 5

/-- The number of blue balls in the bag -/
def blue_balls : ℕ := 3

/-- The total number of balls in the bag -/
def total_balls : ℕ := green_balls + red_balls + blue_balls

/-- The probability of drawing two balls of the same color with replacement -/
def same_color_probability : ℚ :=
  (green_balls / total_balls) ^ 2 +
  (red_balls / total_balls) ^ 2 +
  (blue_balls / total_balls) ^ 2

theorem same_color_probability_value :
  same_color_probability = 49 / 128 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_value_l2450_245007


namespace NUMINAMATH_CALUDE_linear_equation_solve_l2450_245057

theorem linear_equation_solve (x y : ℝ) :
  2 * x + y = 5 → y = -2 * x + 5 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solve_l2450_245057


namespace NUMINAMATH_CALUDE_absolute_value_equality_l2450_245060

theorem absolute_value_equality (x : ℝ) : 
  |(-x)| = |(-8)| → x = 8 ∨ x = -8 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l2450_245060


namespace NUMINAMATH_CALUDE_a_4_equals_zero_l2450_245086

def sequence_a (n : ℕ+) : ℤ := n.val^2 - 2*n.val - 8

theorem a_4_equals_zero : sequence_a 4 = 0 := by sorry

end NUMINAMATH_CALUDE_a_4_equals_zero_l2450_245086


namespace NUMINAMATH_CALUDE_count_possible_roots_l2450_245084

/-- A polynomial with integer coefficients of the form 12x^5 + b₄x^4 + b₃x^3 + b₂x^2 + b₁x + 24 = 0 -/
def IntegerPolynomial (b₄ b₃ b₂ b₁ : ℤ) (x : ℚ) : ℚ :=
  12 * x^5 + b₄ * x^4 + b₃ * x^3 + b₂ * x^2 + b₁ * x + 24

/-- The set of possible rational roots for the polynomial -/
def PossibleRoots : Finset ℚ :=
  {1, 2, 3, 4, 6, 8, 12, 24, 1/2, 1/3, 1/4, 1/6, 2/3, 3/2, 3/4, 4/3,
   -1, -2, -3, -4, -6, -8, -12, -24, -1/2, -1/3, -1/4, -1/6, -2/3, -3/2, -3/4, -4/3}

/-- Theorem stating that the number of possible rational roots is 32 -/
theorem count_possible_roots (b₄ b₃ b₂ b₁ : ℤ) :
  Finset.card PossibleRoots = 32 ∧
  ∀ q : ℚ, q ∉ PossibleRoots → IntegerPolynomial b₄ b₃ b₂ b₁ q ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_count_possible_roots_l2450_245084


namespace NUMINAMATH_CALUDE_preferred_pets_combinations_l2450_245040

/-- The number of puppies available in the pet store -/
def num_puppies : Nat := 20

/-- The number of kittens available in the pet store -/
def num_kittens : Nat := 10

/-- The number of hamsters available in the pet store -/
def num_hamsters : Nat := 12

/-- The number of ways Alice, Bob, and Charlie can buy their preferred pets -/
def num_ways : Nat := num_puppies * num_kittens * num_hamsters

/-- Theorem stating that the number of ways to buy preferred pets is 2400 -/
theorem preferred_pets_combinations : num_ways = 2400 := by
  sorry

end NUMINAMATH_CALUDE_preferred_pets_combinations_l2450_245040


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2450_245092

/-- An isosceles triangle with two sides of length 3 and one side of length 1 -/
structure IsoscelesTriangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (base : ℝ)
  (isIsosceles : side1 = side2)
  (side1_eq_3 : side1 = 3)
  (base_eq_1 : base = 1)

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ :=
  t.side1 + t.side2 + t.base

/-- Theorem: The perimeter of the specified isosceles triangle is 7 -/
theorem isosceles_triangle_perimeter :
  ∀ t : IsoscelesTriangle, perimeter t = 7 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2450_245092


namespace NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l2450_245089

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 28 ways to distribute 6 indistinguishable balls into 3 distinguishable boxes -/
theorem distribute_six_balls_three_boxes : distribute_balls 6 3 = 28 := by
  sorry

end NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l2450_245089


namespace NUMINAMATH_CALUDE_isosceles_triangles_l2450_245027

/-- A circle with two equal chords that extend to intersect -/
structure CircleWithIntersectingChords where
  /-- The circle -/
  circle : Set (ℝ × ℝ)
  /-- The center of the circle -/
  center : ℝ × ℝ
  /-- First chord endpoint -/
  A : ℝ × ℝ
  /-- Second chord endpoint -/
  B : ℝ × ℝ
  /-- Third chord endpoint -/
  C : ℝ × ℝ
  /-- Fourth chord endpoint -/
  D : ℝ × ℝ
  /-- Intersection point of extended chords -/
  P : ℝ × ℝ
  /-- A and B are on the circle -/
  hAB : A ∈ circle ∧ B ∈ circle
  /-- C and D are on the circle -/
  hCD : C ∈ circle ∧ D ∈ circle
  /-- AB and CD are equal chords -/
  hEqualChords : dist A B = dist C D
  /-- P is on the extension of AB beyond B -/
  hPAB : ∃ t > 1, P = A + t • (B - A)
  /-- P is on the extension of CD beyond C -/
  hPCD : ∃ t > 1, P = D + t • (C - D)

/-- The main theorem: triangles APD and BPC are isosceles -/
theorem isosceles_triangles (cfg : CircleWithIntersectingChords) :
  dist cfg.A cfg.P = dist cfg.D cfg.P ∧ dist cfg.B cfg.P = dist cfg.C cfg.P := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangles_l2450_245027


namespace NUMINAMATH_CALUDE_sqrt_seven_irrational_rational_numbers_sqrt_seven_is_irrational_l2450_245010

theorem sqrt_seven_irrational :
  ∀ (a b : ℚ), a^2 ≠ 7 * b^2 :=
sorry

theorem rational_numbers :
  ∃ (q₁ q₂ q₃ : ℚ),
    (q₁ : ℝ) = 3.14159265 ∧
    (q₂ : ℝ) = Real.sqrt 36 ∧
    (q₃ : ℝ) = 4.1 :=
sorry

theorem sqrt_seven_is_irrational :
  Irrational (Real.sqrt 7) :=
sorry

end NUMINAMATH_CALUDE_sqrt_seven_irrational_rational_numbers_sqrt_seven_is_irrational_l2450_245010


namespace NUMINAMATH_CALUDE_quadratic_properties_l2450_245015

def quadratic_function (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 2

theorem quadratic_properties :
  ∀ (a b m : ℝ),
  (quadratic_function a b 2 = 0) →
  (quadratic_function a b 1 = m) →
  (
    (m = 3 → a = -2 ∧ b = 3) ∧
    (m = 3 → ∀ x, -1 ≤ x ∧ x ≤ 2 → -3 ≤ quadratic_function a b x ∧ quadratic_function a b x ≤ 25/8) ∧
    (a > 0 → m < 1)
  ) := by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l2450_245015


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2450_245064

/-- The eccentricity of a hyperbola with given equation and asymptote -/
theorem hyperbola_eccentricity (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∀ x y : ℝ, Real.sqrt 5 * x - 2 * y = 0 → y = (Real.sqrt 5 / 2) * x) →
  b / a = Real.sqrt 5 / 2 →
  Real.sqrt ((a^2 + b^2) / a^2) = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2450_245064


namespace NUMINAMATH_CALUDE_subset_implies_lower_bound_l2450_245067

theorem subset_implies_lower_bound (a : ℝ) : 
  (∀ x : ℝ, x < 5 → x < a) → a ≥ 5 := by sorry

end NUMINAMATH_CALUDE_subset_implies_lower_bound_l2450_245067


namespace NUMINAMATH_CALUDE_price_restoration_l2450_245045

theorem price_restoration (original_price : ℝ) (original_price_pos : 0 < original_price) : 
  let price_after_increases := original_price * (1 + 0.1) * (1 + 0.1) * (1 + 0.05)
  let price_after_decrease := price_after_increases * (1 - 0.22)
  price_after_decrease = original_price := by sorry

end NUMINAMATH_CALUDE_price_restoration_l2450_245045


namespace NUMINAMATH_CALUDE_M_remainder_81_l2450_245073

/-- The largest integer multiple of 9 with no repeated digits and all non-zero digits -/
def M : ℕ :=
  sorry

/-- M is a multiple of 9 -/
axiom M_multiple_of_9 : M % 9 = 0

/-- All digits of M are different -/
axiom M_distinct_digits : ∀ i j, i ≠ j → (M / 10^i % 10) ≠ (M / 10^j % 10)

/-- All digits of M are non-zero -/
axiom M_nonzero_digits : ∀ i, (M / 10^i % 10) ≠ 0

/-- M is the largest such number -/
axiom M_largest : ∀ n, n % 9 = 0 → (∀ i j, i ≠ j → (n / 10^i % 10) ≠ (n / 10^j % 10)) → 
                  (∀ i, (n / 10^i % 10) ≠ 0) → n ≤ M

theorem M_remainder_81 : M % 100 = 81 :=
  sorry

end NUMINAMATH_CALUDE_M_remainder_81_l2450_245073


namespace NUMINAMATH_CALUDE_triangle_problem_l2450_245008

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Given conditions
  b * Real.sin A = Real.sqrt 3 * a * Real.cos B →
  b = 3 →
  Real.sin C = 2 * Real.sin A →
  -- Conclusions
  B = π / 3 ∧
  a = Real.sqrt 3 ∧
  c = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l2450_245008


namespace NUMINAMATH_CALUDE_area_of_triangle_ABC_l2450_245085

-- Define the square WXYZ
def WXYZ : Real := 36

-- Define the side length of smaller squares
def small_square_side : Real := 2

-- Define the triangle ABC
structure Triangle :=
  (AB : Real)
  (AC : Real)
  (BC : Real)

-- Define the coincidence of point A with center O when folded
def coincides_with_center (t : Triangle) : Prop :=
  t.AB = t.AC ∧ t.AB = (WXYZ.sqrt / 2) + small_square_side

-- Define the theorem
theorem area_of_triangle_ABC :
  ∀ t : Triangle,
  coincides_with_center t →
  t.BC = WXYZ.sqrt - 2 * small_square_side →
  (1 / 2) * t.BC * ((WXYZ.sqrt / 2) + small_square_side) = 5 :=
sorry

end NUMINAMATH_CALUDE_area_of_triangle_ABC_l2450_245085


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2450_245005

theorem sum_of_squares_of_roots (x : ℝ) : 
  x^2 - 10*x + 9 = 0 → ∃ s₁ s₂ : ℝ, s₁^2 + s₂^2 = 82 ∧ (x = s₁ ∨ x = s₂) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2450_245005


namespace NUMINAMATH_CALUDE_inverse_value_l2450_245074

noncomputable section

variables (f g : ℝ → ℝ)

-- f⁻¹(g(x)) = x^4 - 1
axiom inverse_composition (x : ℝ) : f⁻¹ (g x) = x^4 - 1

-- g has an inverse
axiom g_has_inverse : Function.Bijective g

theorem inverse_value : g⁻¹ (f 10) = (11 : ℝ)^(1/4) := by sorry

end NUMINAMATH_CALUDE_inverse_value_l2450_245074


namespace NUMINAMATH_CALUDE_simplify_power_expression_l2450_245068

theorem simplify_power_expression (x : ℝ) : (3 * x^4)^4 = 81 * x^16 := by
  sorry

end NUMINAMATH_CALUDE_simplify_power_expression_l2450_245068


namespace NUMINAMATH_CALUDE_binomial_expansion_theorem_l2450_245026

theorem binomial_expansion_theorem (n k : ℕ) (a b : ℝ) : 
  n ≥ 3 →
  a * b ≠ 0 →
  a = k^2 * b →
  k > 0 →
  (n.choose 2) * (a + b)^(n - 2) * a * b + (n.choose 3) * (a + b)^(n - 3) * a^2 * b = 0 →
  n = 3 * k + 2 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_theorem_l2450_245026


namespace NUMINAMATH_CALUDE_alice_unanswered_questions_l2450_245012

/-- Represents a scoring system for a math competition -/
structure ScoringSystem where
  correct : Int
  wrong : Int
  unanswered : Int
  initial : Int

/-- Represents the results of a math competition -/
structure CompetitionResult where
  correct : Nat
  wrong : Nat
  unanswered : Nat
  total_questions : Nat
  new_score : Int
  old_score : Int

def new_system : ScoringSystem := ⟨6, 0, 3, 0⟩
def old_system : ScoringSystem := ⟨5, -2, 0, 20⟩

/-- Calculates the score based on a given scoring system and competition result -/
def calculate_score (system : ScoringSystem) (result : CompetitionResult) : Int :=
  system.initial + 
  system.correct * result.correct + 
  system.wrong * result.wrong + 
  system.unanswered * result.unanswered

theorem alice_unanswered_questions 
  (result : CompetitionResult)
  (h1 : result.new_score = 105)
  (h2 : result.old_score = 75)
  (h3 : result.total_questions = 30)
  (h4 : calculate_score new_system result = result.new_score)
  (h5 : calculate_score old_system result = result.old_score)
  (h6 : result.correct + result.wrong + result.unanswered = result.total_questions) :
  result.unanswered = 5 := by
  sorry

#check alice_unanswered_questions

end NUMINAMATH_CALUDE_alice_unanswered_questions_l2450_245012


namespace NUMINAMATH_CALUDE_marble_problem_l2450_245028

theorem marble_problem (total initial_marbles : ℕ) 
  (white red blue : ℕ) 
  (h1 : total = 50)
  (h2 : red = blue)
  (h3 : white + red + blue = total)
  (h4 : total - (2 * (white - blue)) = 40) :
  white = 5 := by sorry

end NUMINAMATH_CALUDE_marble_problem_l2450_245028


namespace NUMINAMATH_CALUDE_congruence_solution_l2450_245052

theorem congruence_solution (n : ℤ) : 13 * n ≡ 19 [ZMOD 47] ↔ n ≡ 30 [ZMOD 47] := by sorry

end NUMINAMATH_CALUDE_congruence_solution_l2450_245052


namespace NUMINAMATH_CALUDE_problem_solution_l2450_245061

theorem problem_solution (f : ℝ → ℝ) (m a b c : ℝ) 
  (h1 : ∀ x, f x = |x - m|)
  (h2 : Set.Icc (-1) 5 = {x | f x ≤ 3})
  (h3 : a - 2*b + 2*c = m) : 
  m = 2 ∧ (∃ (min : ℝ), min = 4/9 ∧ a^2 + b^2 + c^2 ≥ min) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2450_245061


namespace NUMINAMATH_CALUDE_solution_difference_l2450_245063

theorem solution_difference (x y : ℝ) : 
  (Int.floor x + (y - Int.floor y) = 3.7) →
  ((x - Int.floor x) + Int.floor y = 4.2) →
  |x - y| = 1.5 := by sorry

end NUMINAMATH_CALUDE_solution_difference_l2450_245063


namespace NUMINAMATH_CALUDE_cloth_selling_price_l2450_245000

/-- Calculates the total selling price of cloth given the quantity, profit per meter, and cost price per meter. -/
def totalSellingPrice (quantity : ℕ) (profitPerMeter : ℕ) (costPricePerMeter : ℕ) : ℕ :=
  quantity * (costPricePerMeter + profitPerMeter)

/-- Theorem stating that the total selling price for 85 meters of cloth with a profit of 10 Rs per meter and a cost price of 95 Rs per meter is 8925 Rs. -/
theorem cloth_selling_price :
  totalSellingPrice 85 10 95 = 8925 := by
  sorry

end NUMINAMATH_CALUDE_cloth_selling_price_l2450_245000


namespace NUMINAMATH_CALUDE_min_volume_pyramid_l2450_245021

/-- A pyramid with a regular triangular base -/
structure Pyramid where
  base_side_length : ℝ
  apex_angle : ℝ

/-- The volume of the pyramid -/
noncomputable def volume (p : Pyramid) : ℝ := sorry

/-- The constraint on the apex angle -/
def apex_angle_constraint (p : Pyramid) : Prop :=
  p.apex_angle ≤ 2 * Real.arcsin (1/3)

theorem min_volume_pyramid :
  ∃ (p : Pyramid),
    p.base_side_length = 6 ∧
    apex_angle_constraint p ∧
    (∀ (q : Pyramid),
      q.base_side_length = 6 →
      apex_angle_constraint q →
      volume p ≤ volume q) ∧
    volume p = 5 * Real.sqrt 23 :=
sorry

end NUMINAMATH_CALUDE_min_volume_pyramid_l2450_245021


namespace NUMINAMATH_CALUDE_hooper_bay_lobster_ratio_l2450_245013

/-- The ratio of lobster in Hooper Bay to other harbors -/
theorem hooper_bay_lobster_ratio :
  let total_lobster : ℕ := 480
  let other_harbors_lobster : ℕ := 80 + 80
  let hooper_bay_lobster : ℕ := total_lobster - other_harbors_lobster
  (hooper_bay_lobster : ℚ) / other_harbors_lobster = 2 := by
  sorry

end NUMINAMATH_CALUDE_hooper_bay_lobster_ratio_l2450_245013


namespace NUMINAMATH_CALUDE_parallel_vectors_condition_l2450_245051

/-- Given vectors a and b, where a is parallel to their sum, prove that the y-component of b is -3. -/
theorem parallel_vectors_condition (a b : ℝ × ℝ) : 
  a = (-1, 1) → 
  b.1 = 3 → 
  ∃ (k : ℝ), k • a = (a + b) → 
  b.2 = -3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_condition_l2450_245051


namespace NUMINAMATH_CALUDE_handshake_count_l2450_245058

def number_of_couples : ℕ := 15

def number_of_people : ℕ := 2 * number_of_couples

def handshakes_among_men : ℕ := (number_of_couples - 1) * (number_of_couples - 2) / 2

def handshakes_between_men_and_women : ℕ := number_of_couples * (number_of_couples - 1)

def total_handshakes : ℕ := handshakes_among_men + handshakes_between_men_and_women

theorem handshake_count : total_handshakes = 301 := by
  sorry

end NUMINAMATH_CALUDE_handshake_count_l2450_245058


namespace NUMINAMATH_CALUDE_min_lamps_l2450_245030

theorem min_lamps (n p : ℕ) (h1 : p > 0) : 
  (∃ (p : ℕ), p > 0 ∧ 
    (p + 10*n - 30) - p = 100 ∧ 
    (∀ m : ℕ, m < n → ¬(∃ (q : ℕ), q > 0 ∧ (q + 10*m - 30) - q = 100))) → 
  n = 13 := by
sorry

end NUMINAMATH_CALUDE_min_lamps_l2450_245030


namespace NUMINAMATH_CALUDE_apple_basket_solution_l2450_245069

def basket_problem (x : ℕ) : Prop :=
  let first_sale := x / 4 + 6
  let remaining_after_first := x - first_sale
  let second_sale := remaining_after_first / 3 + 4
  let remaining_after_second := remaining_after_first - second_sale
  let third_sale := remaining_after_second / 2 + 3
  let final_remaining := remaining_after_second - third_sale
  final_remaining = 4

theorem apple_basket_solution :
  ∃ x : ℕ, basket_problem x ∧ x = 28 :=
sorry

end NUMINAMATH_CALUDE_apple_basket_solution_l2450_245069


namespace NUMINAMATH_CALUDE_ratio_a_to_b_l2450_245099

/-- Given that 0.5% of a is 85 paise and 0.75% of b is 150 paise, 
    prove that the ratio of a to b is 17:20 -/
theorem ratio_a_to_b (a b : ℚ) 
  (ha : (5 / 1000) * a = 85 / 100) 
  (hb : (75 / 10000) * b = 150 / 100) : 
  a / b = 17 / 20 := by
  sorry

end NUMINAMATH_CALUDE_ratio_a_to_b_l2450_245099


namespace NUMINAMATH_CALUDE_tangent_line_and_min_value_l2450_245079

/-- The function f(x) = -x^3 + 3x^2 + 9x + a -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := -3*x^2 + 6*x + 9

theorem tangent_line_and_min_value (a : ℝ) :
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f a x ≤ 22) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f a x = 22) →
  (∀ y : ℝ, (9 * 2 - y + 2 = 0) ↔ (y - f (-2) 2 = f' 2 * (y - 2))) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f 0 x ≤ f 0 y ∧ f 0 x = -7) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_min_value_l2450_245079


namespace NUMINAMATH_CALUDE_prime_square_mod_30_l2450_245043

theorem prime_square_mod_30 (p : ℕ) (hp : Prime p) (h2 : p ≠ 2) (h3 : p ≠ 3) (h5 : p ≠ 5) :
  p ^ 2 % 30 = 1 ∨ p ^ 2 % 30 = 19 := by
sorry

end NUMINAMATH_CALUDE_prime_square_mod_30_l2450_245043


namespace NUMINAMATH_CALUDE_vaccine_development_probabilities_l2450_245022

/-- Success probability of Company A for developing vaccine A -/
def prob_A : ℚ := 2/3

/-- Success probability of Company B for developing vaccine A -/
def prob_B : ℚ := 1/2

/-- The theorem states that given the success probabilities of Company A and Company B,
    1) The probability that both succeed is 1/3
    2) The probability of vaccine A being successfully developed is 5/6 -/
theorem vaccine_development_probabilities :
  (prob_A * prob_B = 1/3) ∧
  (1 - (1 - prob_A) * (1 - prob_B) = 5/6) :=
sorry

end NUMINAMATH_CALUDE_vaccine_development_probabilities_l2450_245022


namespace NUMINAMATH_CALUDE_skylar_current_age_l2450_245088

/-- Represents Skylar's donation history and age calculation -/
def skylar_age (start_age : ℕ) (annual_donation : ℕ) (total_donated : ℕ) : ℕ :=
  start_age + total_donated / annual_donation

/-- Theorem stating Skylar's current age -/
theorem skylar_current_age :
  skylar_age 13 5 105 = 34 := by
  sorry

end NUMINAMATH_CALUDE_skylar_current_age_l2450_245088


namespace NUMINAMATH_CALUDE_ratio_of_sum_to_difference_l2450_245018

theorem ratio_of_sum_to_difference (x y : ℝ) : 
  x > 0 → y > 0 → x > y → x + y = 7 * (x - y) → x / y = 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sum_to_difference_l2450_245018


namespace NUMINAMATH_CALUDE_prob_different_topics_is_five_sixths_l2450_245050

/-- The number of essay topics -/
def num_topics : ℕ := 6

/-- The probability that two students select different topics -/
def prob_different_topics : ℚ := 5/6

/-- Theorem stating that the probability of two students selecting different topics
    from 6 available topics is 5/6 -/
theorem prob_different_topics_is_five_sixths :
  prob_different_topics = 5/6 := by sorry

end NUMINAMATH_CALUDE_prob_different_topics_is_five_sixths_l2450_245050


namespace NUMINAMATH_CALUDE_remaining_money_l2450_245035

def initial_amount : ℚ := 50
def shirt_cost : ℚ := 7.85
def meal_cost : ℚ := 15.49
def magazine_cost : ℚ := 6.13
def debt_payment : ℚ := 3.27
def cd_cost : ℚ := 11.75

theorem remaining_money :
  initial_amount - (shirt_cost + meal_cost + magazine_cost + debt_payment + cd_cost) = 5.51 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_l2450_245035


namespace NUMINAMATH_CALUDE_percent_relation_l2450_245082

theorem percent_relation (x y : ℝ) (h : 0.3 * (x - y) = 0.2 * (x + y)) :
  y = 0.2 * x := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l2450_245082


namespace NUMINAMATH_CALUDE_smallest_with_12_divisors_l2450_245090

/-- The number of positive integer divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- n is the smallest positive integer satisfying property P -/
def is_smallest_satisfying (n : ℕ) (P : ℕ → Prop) : Prop :=
  P n ∧ ∀ m : ℕ, 0 < m ∧ m < n → ¬P m

theorem smallest_with_12_divisors :
  is_smallest_satisfying 288 (λ n => num_divisors n = 12) := by sorry

end NUMINAMATH_CALUDE_smallest_with_12_divisors_l2450_245090


namespace NUMINAMATH_CALUDE_exists_six_digit_number_with_digit_sum_43_l2450_245003

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem exists_six_digit_number_with_digit_sum_43 :
  ∃ n : ℕ, n < 500000 ∧ n ≥ 100000 ∧ sum_of_digits n = 43 :=
by
  sorry

end NUMINAMATH_CALUDE_exists_six_digit_number_with_digit_sum_43_l2450_245003


namespace NUMINAMATH_CALUDE_divisors_ending_in_2_mod_2010_l2450_245006

-- Define the number 2010
def n : ℕ := 2010

-- Define the function that counts divisors ending in 2
def count_divisors_ending_in_2 (x : ℕ) : ℕ := sorry

-- Theorem statement
theorem divisors_ending_in_2_mod_2010 : 
  count_divisors_ending_in_2 (n^n) % n = 503 := by sorry

end NUMINAMATH_CALUDE_divisors_ending_in_2_mod_2010_l2450_245006


namespace NUMINAMATH_CALUDE_rect_to_polar_equiv_l2450_245093

/-- Proves that the point (-1, √3) in rectangular coordinates 
    is equivalent to (2, 2π/3) in polar coordinates. -/
theorem rect_to_polar_equiv : 
  let x : ℝ := -1
  let y : ℝ := Real.sqrt 3
  let r : ℝ := 2
  let θ : ℝ := 2 * Real.pi / 3
  (r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi) → 
  (x = r * Real.cos θ ∧ y = r * Real.sin θ) :=
by sorry

end NUMINAMATH_CALUDE_rect_to_polar_equiv_l2450_245093


namespace NUMINAMATH_CALUDE_distance_proof_l2450_245054

/-- The distance between three equidistant points A, B, and C. -/
def distance_between_points : ℝ := 26

/-- The speed of the cyclist traveling from A to B in km/h. -/
def cyclist_speed : ℝ := 15

/-- The speed of the tourist traveling from B to C in km/h. -/
def tourist_speed : ℝ := 5

/-- The time at which the cyclist and tourist are at their shortest distance, in hours. -/
def time_shortest_distance : ℝ := 1.4

/-- The theorem stating that the distance between the points is 26 km under the given conditions. -/
theorem distance_proof :
  ∀ (S : ℝ),
  (S > 0) →
  (S = distance_between_points) →
  (∀ (t : ℝ), 
    (t > 0) →
    (cyclist_speed * t ≤ S) →
    (tourist_speed * t ≤ S) →
    (S^2 - 35*t*S + 325*t^2 ≥ S^2 - 35*time_shortest_distance*S + 325*time_shortest_distance^2)) →
  (S = 26) :=
sorry

end NUMINAMATH_CALUDE_distance_proof_l2450_245054


namespace NUMINAMATH_CALUDE_octal_243_equals_decimal_163_l2450_245062

/-- Converts a base-8 number to base-10 --/
def octal_to_decimal (octal : ℕ) : ℕ :=
  let hundreds := octal / 100
  let tens := (octal % 100) / 10
  let ones := octal % 10
  hundreds * 8^2 + tens * 8^1 + ones * 8^0

/-- The octal number 243 is equal to 163 in decimal --/
theorem octal_243_equals_decimal_163 : octal_to_decimal 243 = 163 := by
  sorry

end NUMINAMATH_CALUDE_octal_243_equals_decimal_163_l2450_245062


namespace NUMINAMATH_CALUDE_equation_solution_l2450_245098

theorem equation_solution : ∃! x : ℝ, x + Real.sqrt (3 * x - 2) = 6 ∧ x = (15 - Real.sqrt 73) / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2450_245098


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l2450_245072

theorem diophantine_equation_solution :
  ∀ x y : ℕ+, x^4 = y^2 + 71 ↔ x = 6 ∧ y = 35 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l2450_245072


namespace NUMINAMATH_CALUDE_susan_age_in_five_years_l2450_245031

/-- Represents the ages and time relationships in the problem -/
structure AgeRelationship where
  j : ℕ  -- James' current age
  n : ℕ  -- Janet's current age
  s : ℕ  -- Susan's current age
  x : ℕ  -- Years until James turns 37

/-- The conditions given in the problem -/
def problem_conditions (ar : AgeRelationship) : Prop :=
  (ar.j - 8 = 2 * (ar.n - 8)) ∧  -- 8 years ago, James was twice Janet's age
  (ar.j + ar.x = 37) ∧           -- In x years, James will turn 37
  (ar.s = ar.n - 3)              -- Susan was born when Janet turned 3

/-- The theorem to be proved -/
theorem susan_age_in_five_years (ar : AgeRelationship) 
  (h : problem_conditions ar) : 
  ar.s + 5 = ar.n + 2 := by
  sorry


end NUMINAMATH_CALUDE_susan_age_in_five_years_l2450_245031


namespace NUMINAMATH_CALUDE_system_solvable_iff_l2450_245001

/-- The system of equations -/
def system (b a x y : ℝ) : Prop :=
  x^2 + y^2 + 2*b*(b + x + y) = 81 ∧ y = 5 / ((x - a)^2 + 1)

/-- The theorem stating the condition for the existence of a solution -/
theorem system_solvable_iff (b : ℝ) :
  (∃ a : ℝ, ∃ x y : ℝ, system b a x y) ↔ -14 < b ∧ b < 9 :=
sorry

end NUMINAMATH_CALUDE_system_solvable_iff_l2450_245001


namespace NUMINAMATH_CALUDE_wall_length_theorem_l2450_245071

/-- Calculates the length of a wall built by a different number of workers in a different time, 
    given the original wall length and worker-days. -/
def calculate_wall_length (original_workers : ℕ) (original_days : ℕ) (original_length : ℕ) 
                          (new_workers : ℕ) (new_days : ℕ) : ℚ :=
  (original_workers * original_days * original_length : ℚ) / (new_workers * new_days)

theorem wall_length_theorem (original_workers : ℕ) (original_days : ℕ) (original_length : ℕ) 
                            (new_workers : ℕ) (new_days : ℕ) :
  original_workers = 18 →
  original_days = 42 →
  original_length = 140 →
  new_workers = 30 →
  new_days = 18 →
  calculate_wall_length original_workers original_days original_length new_workers new_days = 196 := by
  sorry

#eval calculate_wall_length 18 42 140 30 18

end NUMINAMATH_CALUDE_wall_length_theorem_l2450_245071


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2450_245046

/-- Hyperbola type -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : 0 < a ∧ 0 < b

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem: Eccentricity of a specific hyperbola -/
theorem hyperbola_eccentricity (C : Hyperbola) (P F₁ F₂ Q O : Point) : 
  (P.x^2 / C.a^2 - P.y^2 / C.b^2 = 1) →  -- P is on the hyperbola
  (F₁.x < 0 ∧ F₁.y = 0 ∧ F₂.x > 0 ∧ F₂.y = 0) →  -- F₁ and F₂ are left and right foci
  ((P.x - F₂.x) * (F₁.x - F₂.x) + (P.y - F₂.y) * (F₁.y - F₂.y) = 0) →  -- PF₂ ⟂ F₁F₂
  (∃ t : ℝ, Q.x = 0 ∧ Q.y = t * P.y + (1 - t) * F₁.y) →  -- PF₁ intersects y-axis at Q
  (O.x = 0 ∧ O.y = 0) →  -- O is the origin
  (∃ M : Point, ∃ r : ℝ, 
    (M.x - O.x)^2 + (M.y - O.y)^2 = r^2 ∧
    (M.x - F₂.x)^2 + (M.y - F₂.y)^2 = r^2 ∧
    (M.x - P.x)^2 + (M.y - P.y)^2 = r^2 ∧
    (M.x - Q.x)^2 + (M.y - Q.y)^2 = r^2) →  -- OF₂PQ has an inscribed circle
  (F₂.x^2 - F₁.x^2) / C.a^2 = 4  -- Eccentricity is 2
:= by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2450_245046


namespace NUMINAMATH_CALUDE_travel_time_calculation_l2450_245037

/-- Given a speed of 65 km/hr and a distance of 195 km, the travel time is 3 hours -/
theorem travel_time_calculation (speed : ℝ) (distance : ℝ) (time : ℝ) :
  speed = 65 → distance = 195 → time = distance / speed → time = 3 := by
  sorry

end NUMINAMATH_CALUDE_travel_time_calculation_l2450_245037


namespace NUMINAMATH_CALUDE_height_difference_ruby_xavier_l2450_245002

-- Define heights as natural numbers (in centimeters)
def janet_height : ℕ := 62
def charlene_height : ℕ := 2 * janet_height
def pablo_height : ℕ := charlene_height + 70
def ruby_height : ℕ := pablo_height - 2
def xavier_height : ℕ := charlene_height + 84
def paul_height : ℕ := ruby_height + 45

-- Theorem statement
theorem height_difference_ruby_xavier : 
  xavier_height - ruby_height = 7 := by sorry

end NUMINAMATH_CALUDE_height_difference_ruby_xavier_l2450_245002


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2450_245016

-- Problem 1
theorem problem_1 : 
  (2 + 1/4)^(1/2) + (-3.8)^0 - Real.sqrt 3 * (3/2)^(1/3) * (12^(1/6)) = -1/2 := by sorry

-- Problem 2
theorem problem_2 : 
  2 * (Real.log 2 / Real.log 3) - Real.log (32/9) / Real.log 3 + Real.log 8 / Real.log 3 - 
  (Real.log 9 / Real.log 2) * (Real.log 2 / Real.log 3) = 2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2450_245016
