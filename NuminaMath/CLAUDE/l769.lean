import Mathlib

namespace NUMINAMATH_CALUDE_a_minus_b_equals_thirteen_l769_76983

theorem a_minus_b_equals_thirteen (a b : ℝ) 
  (ha : |a| = 8)
  (hb : |b| = 5)
  (ha_pos : a > 0)
  (hb_neg : b < 0) : 
  a - b = 13 := by
sorry

end NUMINAMATH_CALUDE_a_minus_b_equals_thirteen_l769_76983


namespace NUMINAMATH_CALUDE_basketball_tryouts_l769_76977

theorem basketball_tryouts (girls boys called_back : ℕ) 
  (h1 : girls = 39)
  (h2 : boys = 4)
  (h3 : called_back = 26) :
  girls + boys - called_back = 17 := by
sorry

end NUMINAMATH_CALUDE_basketball_tryouts_l769_76977


namespace NUMINAMATH_CALUDE_remainder_calculation_l769_76924

theorem remainder_calculation (a b r : ℕ) 
  (h1 : a - b = 1200)
  (h2 : a = 1495)
  (h3 : a = 5 * b + r)
  (h4 : r < b) : 
  r = 20 := by
sorry

end NUMINAMATH_CALUDE_remainder_calculation_l769_76924


namespace NUMINAMATH_CALUDE_pizza_toppings_l769_76992

theorem pizza_toppings (total_slices : ℕ) (pepperoni_slices : ℕ) (mushroom_slices : ℕ) 
  (h1 : total_slices = 24)
  (h2 : pepperoni_slices = 15)
  (h3 : mushroom_slices = 20)
  (h4 : ∀ slice, slice ≤ total_slices → (slice ≤ pepperoni_slices ∨ slice ≤ mushroom_slices)) :
  ∃ both_toppings : ℕ, 
    both_toppings = pepperoni_slices + mushroom_slices - total_slices ∧
    both_toppings = 11 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_l769_76992


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l769_76903

theorem triangle_area_theorem (x : ℝ) (h1 : x > 0) :
  (1/2) * x * (3*x) = 96 → x = 8 := by sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l769_76903


namespace NUMINAMATH_CALUDE_line_and_points_l769_76978

-- Define the line equation
def line_equation (x y : ℝ) : Prop := y = -2

-- Define the points
def point_A : ℝ × ℝ := (2, -2)
def point_B : ℝ × ℝ := (3, 2)

-- Theorem statement
theorem line_and_points :
  (∀ x y : ℝ, line_equation x y → y = -2) ∧  -- Line equation is y = -2
  (∀ x : ℝ, line_equation x (-2))            -- Line is parallel to x-axis
  ∧ line_equation point_A.1 point_A.2        -- Point A lies on the line
  ∧ ¬line_equation point_B.1 point_B.2 :=    -- Point B does not lie on the line
by sorry

end NUMINAMATH_CALUDE_line_and_points_l769_76978


namespace NUMINAMATH_CALUDE_store_profit_theorem_l769_76970

/-- Represents the selling price and number of items sold -/
structure SaleInfo where
  price : ℝ
  quantity : ℝ

/-- The profit function given the cost, price, and quantity -/
def profit (cost : ℝ) (info : SaleInfo) : ℝ :=
  (info.price - cost) * info.quantity

/-- The demand function given the base price, base quantity, and price sensitivity -/
def demand (basePrice baseQuantity priceSensitivity : ℝ) (price : ℝ) : ℝ :=
  baseQuantity - priceSensitivity * (price - basePrice)

theorem store_profit_theorem (cost basePrice baseQuantity priceSensitivity targetProfit : ℝ) :
  cost = 40 ∧
  basePrice = 50 ∧
  baseQuantity = 150 ∧
  priceSensitivity = 5 ∧
  targetProfit = 1500 →
  ∃ (info1 info2 : SaleInfo),
    info1.price = 50 ∧
    info1.quantity = 150 ∧
    info2.price = 70 ∧
    info2.quantity = 50 ∧
    profit cost info1 = targetProfit ∧
    profit cost info2 = targetProfit ∧
    info1.quantity = demand basePrice baseQuantity priceSensitivity info1.price ∧
    info2.quantity = demand basePrice baseQuantity priceSensitivity info2.price ∧
    ∀ (info : SaleInfo),
      profit cost info = targetProfit ∧
      info.quantity = demand basePrice baseQuantity priceSensitivity info.price →
      (info = info1 ∨ info = info2) := by
  sorry


end NUMINAMATH_CALUDE_store_profit_theorem_l769_76970


namespace NUMINAMATH_CALUDE_simplify_expression_l769_76917

theorem simplify_expression (a b : ℝ) (h1 : 2*b - a < 3) (h2 : 2*a - b < 5) :
  -|2*b - a - 7| - |b - 2*a + 8| + |a + b - 9| = -6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l769_76917


namespace NUMINAMATH_CALUDE_plane_relations_theorem_l769_76947

-- Define a type for planes
def Plane : Type := Unit

-- Define the relations between planes
def perpendicular (p q : Plane) : Prop := sorry
def parallel (p q : Plane) : Prop := sorry

-- Define a predicate for three non-collinear points on a plane being equidistant from another plane
def three_points_equidistant (p q : Plane) : Prop := sorry

-- The theorem to be proven
theorem plane_relations_theorem (α β γ : Plane) : 
  ¬((perpendicular α β ∧ perpendicular β γ → parallel α γ) ∨ 
    (three_points_equidistant α β → parallel α β)) := by sorry

end NUMINAMATH_CALUDE_plane_relations_theorem_l769_76947


namespace NUMINAMATH_CALUDE_parabola_through_origin_l769_76951

/-- A parabola is defined by the equation y = ax^2 + bx + c, where a, b, and c are real numbers. -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point on a 2D plane is represented by its x and y coordinates. -/
structure Point where
  x : ℝ
  y : ℝ

/-- The origin is the point (0, 0) on a 2D plane. -/
def origin : Point := ⟨0, 0⟩

/-- A point lies on a parabola if its coordinates satisfy the parabola's equation. -/
def lies_on (p : Point) (para : Parabola) : Prop :=
  p.y = para.a * p.x^2 + para.b * p.x + para.c

/-- Theorem: A parabola passes through the origin if and only if its c coefficient is zero. -/
theorem parabola_through_origin (para : Parabola) :
  lies_on origin para ↔ para.c = 0 := by
  sorry

end NUMINAMATH_CALUDE_parabola_through_origin_l769_76951


namespace NUMINAMATH_CALUDE_system_solution_l769_76932

theorem system_solution (x y : ℝ) : 
  (x^2 + x*y + y^2 = 23) ∧ (x^4 + x^2*y^2 + y^4 = 253) →
  ((x = Real.sqrt 29 ∧ y = Real.sqrt 5) ∨ 
   (x = Real.sqrt 29 ∧ y = -Real.sqrt 5) ∨
   (x = -Real.sqrt 29 ∧ y = Real.sqrt 5) ∨
   (x = -Real.sqrt 29 ∧ y = -Real.sqrt 5)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l769_76932


namespace NUMINAMATH_CALUDE_paving_cost_calculation_l769_76979

/-- The cost of paving a rectangular floor -/
def paving_cost (length width rate : ℝ) : ℝ :=
  length * width * rate

/-- Theorem: The cost of paving a rectangular floor with given dimensions and rate -/
theorem paving_cost_calculation (length width rate : ℝ) 
  (h1 : length = 5.5)
  (h2 : width = 3.75)
  (h3 : rate = 1200) :
  paving_cost length width rate = 24750 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_calculation_l769_76979


namespace NUMINAMATH_CALUDE_system_solution_l769_76993

theorem system_solution :
  ∃ (k m : ℚ),
    (3 * k - 4) / (k + 7) = 2/5 ∧
    2 * m + 5 * k = 14 ∧
    k = 34/13 ∧
    m = 6/13 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l769_76993


namespace NUMINAMATH_CALUDE_percent_application_l769_76930

theorem percent_application (x : ℝ) : x * 0.0002 = 2.4712 → x = 12356 := by sorry

end NUMINAMATH_CALUDE_percent_application_l769_76930


namespace NUMINAMATH_CALUDE_orange_cost_l769_76911

theorem orange_cost (num_bananas : ℕ) (num_oranges : ℕ) (banana_cost : ℚ) (total_cost : ℚ) :
  num_bananas = 5 →
  num_oranges = 10 →
  banana_cost = 2 →
  total_cost = 25 →
  (total_cost - num_bananas * banana_cost) / num_oranges = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_orange_cost_l769_76911


namespace NUMINAMATH_CALUDE_toy_selling_price_l769_76938

/-- Calculates the total selling price of toys given the number of toys sold,
    the number of toys whose cost price was gained, and the cost price per toy. -/
def totalSellingPrice (numToysSold : ℕ) (numToysGained : ℕ) (costPrice : ℕ) : ℕ :=
  numToysSold * costPrice + numToysGained * costPrice

/-- Theorem stating that for the given conditions, the total selling price is 27300. -/
theorem toy_selling_price :
  totalSellingPrice 18 3 1300 = 27300 := by
  sorry

end NUMINAMATH_CALUDE_toy_selling_price_l769_76938


namespace NUMINAMATH_CALUDE_expression_equals_75_l769_76918

-- Define the expression
def expression : ℚ := 150 / (10 / 5)

-- State the theorem
theorem expression_equals_75 : expression = 75 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_75_l769_76918


namespace NUMINAMATH_CALUDE_problem_solution_l769_76981

theorem problem_solution (y : ℝ) (h : y + Real.sqrt (y^2 - 4) + 1 / (y - Real.sqrt (y^2 - 4)) = 24) :
  y^2 + Real.sqrt (y^4 - 4) + 1 / (y^2 + Real.sqrt (y^4 - 4)) = 1369/36 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l769_76981


namespace NUMINAMATH_CALUDE_fourth_separation_at_136pm_l769_76987

-- Define the distance between cities
def distance_between_cities : ℝ := 300

-- Define the start time
def start_time : ℝ := 6

-- Define the time of first 50 km separation
def first_separation_time : ℝ := 8

-- Define the distance of separation
def separation_distance : ℝ := 50

-- Define the function to calculate the fourth separation time
def fourth_separation_time : ℝ := start_time + 7.6

-- Theorem statement
theorem fourth_separation_at_136pm 
  (h1 : distance_between_cities = 300)
  (h2 : start_time = 6)
  (h3 : first_separation_time = 8)
  (h4 : separation_distance = 50) :
  fourth_separation_time = 13.6 := by sorry

end NUMINAMATH_CALUDE_fourth_separation_at_136pm_l769_76987


namespace NUMINAMATH_CALUDE_articles_bought_l769_76900

/-- The number of articles bought at the cost price -/
def X : ℝ := sorry

/-- The cost price of each article -/
def C : ℝ := sorry

/-- The selling price of each article -/
def S : ℝ := sorry

/-- The gain percent -/
def gain_percent : ℝ := 8.695652173913043

theorem articles_bought (h1 : X * C = 46 * S) 
                        (h2 : gain_percent = ((S - C) / C) * 100) : 
  X = 50 := by sorry

end NUMINAMATH_CALUDE_articles_bought_l769_76900


namespace NUMINAMATH_CALUDE_lamp_cost_ratio_l769_76910

/-- The ratio of the cost of the most expensive lamp to the cheapest lamp -/
theorem lamp_cost_ratio 
  (cheapest_lamp : ℕ) 
  (frank_money : ℕ) 
  (remaining_money : ℕ) 
  (h1 : cheapest_lamp = 20)
  (h2 : frank_money = 90)
  (h3 : remaining_money = 30) :
  (frank_money - remaining_money) / cheapest_lamp = 3 := by
sorry

end NUMINAMATH_CALUDE_lamp_cost_ratio_l769_76910


namespace NUMINAMATH_CALUDE_min_b_value_l769_76967

noncomputable def f (x : ℝ) : ℝ := Real.log x - (1/4) * x + 3/(4*x) - 1

def g (b : ℝ) (x : ℝ) : ℝ := x^2 - 2*b*x + 4

theorem min_b_value (b : ℝ) :
  (∀ x₁ ∈ Set.Ioo 0 2, ∃ x₂ ∈ Set.Icc 1 2, f x₁ ≥ g b x₂) →
  b ≥ 17/8 := by
  sorry

end NUMINAMATH_CALUDE_min_b_value_l769_76967


namespace NUMINAMATH_CALUDE_polynomial_expansion_l769_76913

theorem polynomial_expansion (x : ℝ) : 
  (2*x^2 + 3*x + 7)*(x - 2) - (x - 2)*(x^2 - 4*x + 9) + (4*x^2 - 3*x + 1)*(x - 2)*(x - 5) = 
  5*x^3 - 26*x^2 + 35*x - 6 := by
sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l769_76913


namespace NUMINAMATH_CALUDE_max_distance_circle_centers_l769_76909

/-- The maximum distance between the centers of two circles with 8-inch diameters
    placed within a 16-inch by 20-inch rectangle is 4√13 inches. -/
theorem max_distance_circle_centers (rect_width rect_height circle_diameter : ℝ)
  (hw : rect_width = 20)
  (hh : rect_height = 16)
  (hd : circle_diameter = 8)
  (h_nonneg : rect_width > 0 ∧ rect_height > 0 ∧ circle_diameter > 0) :
  Real.sqrt ((rect_width - circle_diameter) ^ 2 + (rect_height - circle_diameter) ^ 2) = 4 * Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_max_distance_circle_centers_l769_76909


namespace NUMINAMATH_CALUDE_exactly_one_even_negation_l769_76961

/-- Represents the property of a natural number being even -/
def IsEven (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

/-- Represents the property of a natural number being odd -/
def IsOdd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

/-- States that exactly one of three natural numbers is even -/
def ExactlyOneEven (a b c : ℕ) : Prop :=
  (IsEven a ∧ IsOdd b ∧ IsOdd c) ∨
  (IsOdd a ∧ IsEven b ∧ IsOdd c) ∨
  (IsOdd a ∧ IsOdd b ∧ IsEven c)

/-- States that at least two of three natural numbers are even or all are odd -/
def AtLeastTwoEvenOrAllOdd (a b c : ℕ) : Prop :=
  (IsEven a ∧ IsEven b) ∨
  (IsEven a ∧ IsEven c) ∨
  (IsEven b ∧ IsEven c) ∨
  (IsOdd a ∧ IsOdd b ∧ IsOdd c)

theorem exactly_one_even_negation (a b c : ℕ) :
  ¬(ExactlyOneEven a b c) ↔ AtLeastTwoEvenOrAllOdd a b c :=
sorry

end NUMINAMATH_CALUDE_exactly_one_even_negation_l769_76961


namespace NUMINAMATH_CALUDE_find_B_over_A_l769_76946

-- Define the equation
def equation (A B x : ℝ) : Prop :=
  A / (x + 6) + B / (x^2 - 5*x) = (x^3 - 3*x^2 + 12) / (x^3 + x^2 - 30*x)

-- Theorem statement
theorem find_B_over_A (A B : ℤ) :
  (∀ x : ℝ, x ≠ -6 ∧ x ≠ 0 ∧ x ≠ 5 → equation A B x) →
  (B : ℝ) / (A : ℝ) = 2.2 := by
  sorry

end NUMINAMATH_CALUDE_find_B_over_A_l769_76946


namespace NUMINAMATH_CALUDE_thabo_hardcover_books_l769_76949

/-- Represents the number of books Thabo owns in each category -/
structure BookCollection where
  hardcover_nonfiction : ℕ
  paperback_nonfiction : ℕ
  paperback_fiction : ℕ

/-- Thabo's book collection satisfies the given conditions -/
def is_valid_collection (bc : BookCollection) : Prop :=
  bc.hardcover_nonfiction + bc.paperback_nonfiction + bc.paperback_fiction = 500 ∧
  bc.paperback_nonfiction = bc.hardcover_nonfiction + 30 ∧
  bc.paperback_fiction = 3 * bc.paperback_nonfiction

theorem thabo_hardcover_books (bc : BookCollection) 
  (h : is_valid_collection bc) : bc.hardcover_nonfiction = 76 := by
  sorry

end NUMINAMATH_CALUDE_thabo_hardcover_books_l769_76949


namespace NUMINAMATH_CALUDE_total_marbles_l769_76940

theorem total_marbles (jar_a jar_b jar_c : ℕ) : 
  jar_a = 28 →
  jar_b = jar_a + 12 →
  jar_c = 2 * jar_b →
  jar_a + jar_b + jar_c = 148 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_l769_76940


namespace NUMINAMATH_CALUDE_race_finish_orders_l769_76920

theorem race_finish_orders (n : Nat) (h : n = 4) : Nat.factorial n = 24 := by
  sorry

end NUMINAMATH_CALUDE_race_finish_orders_l769_76920


namespace NUMINAMATH_CALUDE_matrix_power_2023_l769_76912

def A : Matrix (Fin 2) (Fin 2) ℕ := !![1, 0; 2, 1]

theorem matrix_power_2023 : 
  A ^ 2023 = !![1, 0; 4046, 1] := by sorry

end NUMINAMATH_CALUDE_matrix_power_2023_l769_76912


namespace NUMINAMATH_CALUDE_range_of_e_l769_76916

theorem range_of_e (a b c d e : ℝ) 
  (sum_condition : a + b + c + d + e = 8)
  (sum_squares_condition : a^2 + b^2 + c^2 + d^2 + e^2 = 16) :
  0 ≤ e ∧ e ≤ 16/5 := by
sorry

end NUMINAMATH_CALUDE_range_of_e_l769_76916


namespace NUMINAMATH_CALUDE_simplify_and_multiply_l769_76905

theorem simplify_and_multiply (b : ℝ) : (2 * (3 * b) * (4 * b^2) * (5 * b^3)) * 6 = 720 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_multiply_l769_76905


namespace NUMINAMATH_CALUDE_watermelon_seeds_count_l769_76901

/-- The total number of watermelon seeds Yeon, Gwi, and Bom have together -/
def total_seeds (bom gwi yeon : ℕ) : ℕ := bom + gwi + yeon

/-- Theorem stating the total number of watermelon seeds -/
theorem watermelon_seeds_count :
  ∀ (bom gwi yeon : ℕ),
  bom = 300 →
  gwi = bom + 40 →
  yeon = 3 * gwi →
  total_seeds bom gwi yeon = 1660 :=
by
  sorry

end NUMINAMATH_CALUDE_watermelon_seeds_count_l769_76901


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l769_76933

theorem geometric_sequence_fifth_term 
  (t : ℕ → ℝ) 
  (h_geometric : ∃ (a r : ℝ), ∀ n, t n = a * r^(n-1))
  (h_t1 : t 1 = 3)
  (h_t2 : t 2 = 6) :
  t 5 = 48 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l769_76933


namespace NUMINAMATH_CALUDE_bookmark_difference_l769_76907

/-- The price of a bookmark in cents -/
def bookmark_price : ℕ := sorry

/-- The number of fifth graders who bought bookmarks -/
def fifth_graders : ℕ := sorry

/-- The number of fourth graders who bought bookmarks -/
def fourth_graders : ℕ := 20

theorem bookmark_difference : 
  bookmark_price > 0 ∧ 
  bookmark_price * fifth_graders = 225 ∧ 
  bookmark_price * fourth_graders = 260 →
  fourth_graders - fifth_graders = 7 := by sorry

end NUMINAMATH_CALUDE_bookmark_difference_l769_76907


namespace NUMINAMATH_CALUDE_smallest_sum_of_three_l769_76944

def S : Finset Int := {10, 30, -12, 15, -8}

theorem smallest_sum_of_three (s : Finset Int) (h : s = S) :
  (Finset.powersetCard 3 s).toList.map (fun t => t.toList.sum)
    |>.minimum?
    |>.map (fun x => x = -10)
    |>.getD False :=
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_three_l769_76944


namespace NUMINAMATH_CALUDE_composite_5n_plus_3_l769_76906

theorem composite_5n_plus_3 (n : ℕ) (h1 : ∃ x : ℕ, 2 * n + 1 = x^2) (h2 : ∃ y : ℕ, 3 * n + 1 = y^2) : 
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 5 * n + 3 = a * b :=
by sorry

end NUMINAMATH_CALUDE_composite_5n_plus_3_l769_76906


namespace NUMINAMATH_CALUDE_number_plus_five_equals_500_l769_76954

theorem number_plus_five_equals_500 : ∃ x : ℤ, x + 5 = 500 ∧ x = 495 := by
  sorry

end NUMINAMATH_CALUDE_number_plus_five_equals_500_l769_76954


namespace NUMINAMATH_CALUDE_sequence_100th_term_l769_76935

theorem sequence_100th_term (a : ℕ → ℕ) (h1 : a 1 = 2) 
    (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 2 * n) : 
  a 100 = 9902 := by
  sorry

end NUMINAMATH_CALUDE_sequence_100th_term_l769_76935


namespace NUMINAMATH_CALUDE_grid_paths_6x5_l769_76964

/-- The number of paths from (0,0) to (m,n) on a grid, moving only right and up -/
def gridPaths (m n : ℕ) : ℕ := Nat.choose (m + n) n

/-- The dimensions of our grid -/
def gridWidth : ℕ := 6
def gridHeight : ℕ := 5

theorem grid_paths_6x5 : 
  gridPaths gridWidth gridHeight = 462 := by sorry

end NUMINAMATH_CALUDE_grid_paths_6x5_l769_76964


namespace NUMINAMATH_CALUDE_perimeter_pedal_relation_not_implies_equilateral_l769_76956

/-- A triangle with vertices A, B, C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The pedal triangle of a given triangle -/
def pedalTriangle (t : Triangle) : Triangle := sorry

/-- The perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ := sorry

/-- Predicate to check if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

/-- Theorem stating that the original statement is false -/
theorem perimeter_pedal_relation_not_implies_equilateral :
  ∃ t : Triangle, perimeter t = 2 * perimeter (pedalTriangle t) ∧ ¬isEquilateral t := by
  sorry

end NUMINAMATH_CALUDE_perimeter_pedal_relation_not_implies_equilateral_l769_76956


namespace NUMINAMATH_CALUDE_yellow_square_area_l769_76922

-- Define the cube's edge length
def cube_edge : ℝ := 15

-- Define the total amount of purple paint
def total_purple_paint : ℝ := 900

-- Define the number of faces on a cube
def num_faces : ℕ := 6

-- Theorem statement
theorem yellow_square_area :
  let total_surface_area := num_faces * (cube_edge ^ 2)
  let purple_area_per_face := total_purple_paint / num_faces
  let yellow_area_per_face := cube_edge ^ 2 - purple_area_per_face
  yellow_area_per_face = 75 := by sorry

end NUMINAMATH_CALUDE_yellow_square_area_l769_76922


namespace NUMINAMATH_CALUDE_problem_statement_l769_76994

theorem problem_statement (a b k : ℕ+) (h : (a.val^2 - 1 - b.val^2) / (a.val * b.val - 1) = k.val) : k = 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l769_76994


namespace NUMINAMATH_CALUDE_square_sum_constant_l769_76957

theorem square_sum_constant (x : ℝ) : (2*x + 3)^2 + 2*(2*x + 3)*(5 - 2*x) + (5 - 2*x)^2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_constant_l769_76957


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_10_l769_76919

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ+ → ℚ
  S : ℕ+ → ℚ
  is_arithmetic : ∀ n : ℕ+, a (n + 1) - a n = a 2 - a 1
  sum_def : ∀ n : ℕ+, S n = (n : ℚ) * (a 1 + a n) / 2

/-- The main theorem -/
theorem arithmetic_sequence_sum_10 (seq : ArithmeticSequence) 
  (h1 : seq.a 3 = 16) (h2 : seq.S 20 = 20) : seq.S 10 = 110 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_10_l769_76919


namespace NUMINAMATH_CALUDE_mistaken_subtraction_l769_76948

/-- Given a two-digit number where the units digit is 9, 
    if subtracting 57 from the number with the units digit mistaken as 6 results in 39,
    then the original number is 99. -/
theorem mistaken_subtraction (x : ℕ) : 
  x < 10 →  -- Ensure x is a single digit (tens place)
  (10 * x + 6) - 57 = 39 → 
  10 * x + 9 = 99 :=
by sorry

end NUMINAMATH_CALUDE_mistaken_subtraction_l769_76948


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l769_76999

theorem pizza_toppings_combinations (n m : ℕ) (h1 : n = 7) (h2 : m = 4) : 
  Nat.choose n m = 35 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l769_76999


namespace NUMINAMATH_CALUDE_functional_equation_solution_l769_76974

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 - y^2) = x * f x - y * f y

/-- The main theorem stating that any function satisfying the functional equation
    is of the form f(x) = x * f(1) -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∀ x : ℝ, f x = x * f 1 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l769_76974


namespace NUMINAMATH_CALUDE_cartoon_time_l769_76985

theorem cartoon_time (cartoon_ratio : ℚ) (chore_ratio : ℚ) (chore_time : ℚ) : 
  cartoon_ratio / chore_ratio = 5 / 4 →
  chore_time = 96 →
  (cartoon_ratio * chore_time) / chore_ratio / 60 = 2 := by
sorry

end NUMINAMATH_CALUDE_cartoon_time_l769_76985


namespace NUMINAMATH_CALUDE_batsman_average_is_60_l769_76926

/-- Represents a batsman's performance statistics -/
structure BatsmanStats where
  total_innings : ℕ
  highest_score : ℕ
  score_difference : ℕ
  avg_excluding_extremes : ℕ

/-- Calculates the overall batting average -/
def overall_average (stats : BatsmanStats) : ℚ :=
  let lowest_score := stats.highest_score - stats.score_difference
  let total_runs := (stats.total_innings - 2) * stats.avg_excluding_extremes + stats.highest_score + lowest_score
  total_runs / stats.total_innings

/-- Theorem stating the overall batting average is 60 runs given the specific conditions -/
theorem batsman_average_is_60 (stats : BatsmanStats) 
  (h_innings : stats.total_innings = 46)
  (h_highest : stats.highest_score = 199)
  (h_diff : stats.score_difference = 190)
  (h_avg : stats.avg_excluding_extremes = 58) :
  overall_average stats = 60 := by
  sorry


end NUMINAMATH_CALUDE_batsman_average_is_60_l769_76926


namespace NUMINAMATH_CALUDE_motorcycle_wheels_l769_76908

theorem motorcycle_wheels (total_wheels : ℕ) (num_cars : ℕ) (num_motorcycles : ℕ) 
  (wheels_per_car : ℕ) (h1 : total_wheels = 117) (h2 : num_cars = 19) 
  (h3 : num_motorcycles = 11) (h4 : wheels_per_car = 5) :
  (total_wheels - num_cars * wheels_per_car) / num_motorcycles = 2 :=
by sorry

end NUMINAMATH_CALUDE_motorcycle_wheels_l769_76908


namespace NUMINAMATH_CALUDE_second_to_third_ratio_l769_76914

/-- Given three numbers where their sum is 500, the first number is 200, and the third number is 100,
    the ratio of the second number to the third number is 2:1. -/
theorem second_to_third_ratio (a b c : ℚ) : 
  a + b + c = 500 → a = 200 → c = 100 → b / c = 2 := by
  sorry

end NUMINAMATH_CALUDE_second_to_third_ratio_l769_76914


namespace NUMINAMATH_CALUDE_abs_neg_2022_l769_76942

theorem abs_neg_2022 : |(-2022 : ℤ)| = 2022 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_2022_l769_76942


namespace NUMINAMATH_CALUDE_pig_count_l769_76939

theorem pig_count (initial_pigs additional_pigs : ℝ) 
  (h1 : initial_pigs = 2465.25)
  (h2 : additional_pigs = 5683.75) : 
  initial_pigs + additional_pigs = 8149 :=
by sorry

end NUMINAMATH_CALUDE_pig_count_l769_76939


namespace NUMINAMATH_CALUDE_marco_dad_strawberries_l769_76955

/-- The weight of additional strawberries found by Marco's dad -/
def additional_strawberries (initial_total final_marco final_dad : ℕ) : ℕ :=
  (final_marco + final_dad) - initial_total

theorem marco_dad_strawberries :
  additional_strawberries 22 36 16 = 30 := by
  sorry

end NUMINAMATH_CALUDE_marco_dad_strawberries_l769_76955


namespace NUMINAMATH_CALUDE_max_hearts_desire_desire_fulfilled_l769_76959

/-- Represents a four-digit natural number M = 1000a + 100b + 10c + d -/
structure FourDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  h1 : 1 ≤ a ∧ a ≤ 9
  h2 : 1 ≤ b ∧ b ≤ 9
  h3 : 1 ≤ c ∧ c ≤ 9
  h4 : 1 ≤ d ∧ d ≤ 9
  h5 : c > d

/-- Calculates the value of M given its digits -/
def FourDigitNumber.value (n : FourDigitNumber) : Nat :=
  1000 * n.a + 100 * n.b + 10 * n.c + n.d

/-- Checks if a number is a "heart's desire" and "desire fulfilled" number -/
def isHeartsDesireAndDesireFulfilled (n : FourDigitNumber) : Prop :=
  (10 * n.b + n.c) / (n.a + n.d) = 11

/-- Calculates F(M) -/
def F (n : FourDigitNumber) : Nat :=
  10 * (n.a + n.b) + 3 * n.c

/-- Main theorem statement -/
theorem max_hearts_desire_desire_fulfilled :
  ∃ (M : FourDigitNumber),
    isHeartsDesireAndDesireFulfilled M ∧
    F M % 7 = 0 ∧
    M.value = 5883 ∧
    (∀ (N : FourDigitNumber),
      isHeartsDesireAndDesireFulfilled N ∧
      F N % 7 = 0 →
      N.value ≤ M.value) := by
  sorry

end NUMINAMATH_CALUDE_max_hearts_desire_desire_fulfilled_l769_76959


namespace NUMINAMATH_CALUDE_dandelion_puffs_to_dog_l769_76941

/-- The number of dandelion puffs Caleb gave to his dog -/
def puffs_to_dog (total : ℕ) (to_mom : ℕ) (to_sister : ℕ) (to_grandmother : ℕ) 
                 (num_friends : ℕ) (to_each_friend : ℕ) : ℕ :=
  total - (to_mom + to_sister + to_grandmother + num_friends * to_each_friend)

/-- Theorem stating the number of dandelion puffs Caleb gave to his dog -/
theorem dandelion_puffs_to_dog : 
  puffs_to_dog 40 3 3 5 3 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_dandelion_puffs_to_dog_l769_76941


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l769_76976

open Real

def quadratic_inequality (k : ℝ) (x : ℝ) : Prop :=
  2 * k * x^2 + k * x - 3/8 < 0

theorem quadratic_inequality_range :
  ∀ k : ℝ, (∀ x : ℝ, quadratic_inequality k x) ↔ k ∈ Set.Ioo (-3/2) 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l769_76976


namespace NUMINAMATH_CALUDE_sin_A_value_area_ABC_l769_76927

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  -- Conditions
  c = Real.sqrt 2 ∧
  a = 1 ∧
  Real.cos C = 3/4

-- Theorem for sin A
theorem sin_A_value (A B C : ℝ) (a b c : ℝ) 
  (h : triangle_ABC A B C a b c) : Real.sin A = Real.sqrt 14 / 8 := by
  sorry

-- Theorem for the area of triangle ABC
theorem area_ABC (A B C : ℝ) (a b c : ℝ) 
  (h : triangle_ABC A B C a b c) : (1/2) * a * b * Real.sin C = Real.sqrt 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_A_value_area_ABC_l769_76927


namespace NUMINAMATH_CALUDE_min_students_forgot_all_items_l769_76971

theorem min_students_forgot_all_items (total : ℕ) (forgot_gloves : ℕ) (forgot_scarves : ℕ) (forgot_hats : ℕ) 
  (h1 : total = 60)
  (h2 : forgot_gloves = 55)
  (h3 : forgot_scarves = 52)
  (h4 : forgot_hats = 50) :
  total - ((total - forgot_gloves) + (total - forgot_scarves) + (total - forgot_hats)) = 37 := by
  sorry

end NUMINAMATH_CALUDE_min_students_forgot_all_items_l769_76971


namespace NUMINAMATH_CALUDE_prob_full_house_is_one_third_l769_76945

/-- Represents the outcome of rolling five six-sided dice -/
structure DiceRoll where
  pairs : Fin 6 × Fin 6
  odd : Fin 6

/-- The probability of getting a full house after rerolling the odd die -/
def prob_full_house_after_reroll (roll : DiceRoll) : ℚ :=
  2 / 6

/-- Theorem stating the probability of getting a full house after rerolling the odd die -/
theorem prob_full_house_is_one_third (roll : DiceRoll) :
  prob_full_house_after_reroll roll = 1 / 3 := by
  sorry

#check prob_full_house_is_one_third

end NUMINAMATH_CALUDE_prob_full_house_is_one_third_l769_76945


namespace NUMINAMATH_CALUDE_fraction_problem_l769_76928

theorem fraction_problem (f : ℝ) : 
  (0.5 * 100 = f * 100 - 10) → f = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l769_76928


namespace NUMINAMATH_CALUDE_triangle_area_from_perimeter_and_inradius_l769_76986

/-- The area of a triangle given its perimeter and inradius -/
theorem triangle_area_from_perimeter_and_inradius
  (perimeter : ℝ) (inradius : ℝ) (angle_smallest_sides : ℝ)
  (h_perimeter : perimeter = 36)
  (h_inradius : inradius = 2.5)
  (h_angle : angle_smallest_sides = 75) :
  inradius * (perimeter / 2) = 45 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_from_perimeter_and_inradius_l769_76986


namespace NUMINAMATH_CALUDE_abs_eq_sum_implies_zero_l769_76997

theorem abs_eq_sum_implies_zero (x y : ℝ) :
  |x - y^2| = x + y^2 → x = 0 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_abs_eq_sum_implies_zero_l769_76997


namespace NUMINAMATH_CALUDE_unique_digit_solution_l769_76929

theorem unique_digit_solution :
  ∃! (square boxplus boxtimes boxminus : ℕ),
    square < 10 ∧ boxplus < 10 ∧ boxtimes < 10 ∧ boxminus < 10 ∧
    square = 423 / 47 ∧
    1448 = 282 * boxminus + square * boxtimes ∧
    423 * boxplus = 282 * 3 ∧
    square = 9 ∧ boxplus = 2 ∧ boxtimes = 8 ∧ boxminus = 5 :=
by sorry

end NUMINAMATH_CALUDE_unique_digit_solution_l769_76929


namespace NUMINAMATH_CALUDE_sundae_price_l769_76968

/-- Given a caterer's order of ice-cream bars and sundaes, calculate the price of each sundae. -/
theorem sundae_price
  (ice_cream_bars : ℕ)
  (sundaes : ℕ)
  (total_price : ℚ)
  (ice_cream_bar_price : ℚ)
  (h1 : ice_cream_bars = 125)
  (h2 : sundaes = 125)
  (h3 : total_price = 250)
  (h4 : ice_cream_bar_price = 0.6) :
  (total_price - ice_cream_bars * ice_cream_bar_price) / sundaes = 1.4 := by
  sorry

#check sundae_price

end NUMINAMATH_CALUDE_sundae_price_l769_76968


namespace NUMINAMATH_CALUDE_solve_lawn_mowing_problem_l769_76972

/-- Edward's lawn mowing business finances -/
def lawn_mowing_problem (spring_earnings summer_earnings final_amount : ℕ) : Prop :=
  let total_earnings := spring_earnings + summer_earnings
  let supplies_cost := total_earnings - final_amount
  supplies_cost = total_earnings - final_amount

theorem solve_lawn_mowing_problem :
  lawn_mowing_problem 2 27 24 = true :=
by sorry

end NUMINAMATH_CALUDE_solve_lawn_mowing_problem_l769_76972


namespace NUMINAMATH_CALUDE_polynomial_xy_coefficient_l769_76963

theorem polynomial_xy_coefficient (k : ℝ) : 
  (∀ x y : ℝ, x^2 - 3*k*x*y - 3*y^2 + 6*x*y - 8 = x^2 + (-3*k + 6)*x*y - 3*y^2 - 8) →
  (-3*k + 6 = 0) →
  k = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_xy_coefficient_l769_76963


namespace NUMINAMATH_CALUDE_total_birds_on_fence_l769_76982

def initial_birds : ℕ := 4
def additional_birds : ℕ := 6

theorem total_birds_on_fence :
  initial_birds + additional_birds = 10 := by sorry

end NUMINAMATH_CALUDE_total_birds_on_fence_l769_76982


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_linear_l769_76973

theorem unique_solution_quadratic_linear (k : ℝ) : 
  (∃! p : ℝ × ℝ, p.2 = p.1^2 ∧ p.2 = 2*p.1 - k) ↔ k = 1 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_linear_l769_76973


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l769_76980

/-- Two points are symmetric with respect to the origin if their coordinates sum to zero -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ + x₂ = 0 ∧ y₁ + y₂ = 0

/-- Given points A(1,a) and B(b,2) are symmetric with respect to the origin,
    prove that a + b = -3 -/
theorem symmetric_points_sum (a b : ℝ) 
  (h : symmetric_wrt_origin 1 a b 2) : a + b = -3 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l769_76980


namespace NUMINAMATH_CALUDE_bessonov_tax_refund_l769_76996

def income_tax : ℝ := 156000
def education_expense : ℝ := 130000
def medical_expense : ℝ := 10000
def tax_rate : ℝ := 0.13

def total_deductible_expenses : ℝ := education_expense + medical_expense

def max_refund : ℝ := tax_rate * total_deductible_expenses

theorem bessonov_tax_refund :
  min max_refund income_tax = 18200 :=
sorry

end NUMINAMATH_CALUDE_bessonov_tax_refund_l769_76996


namespace NUMINAMATH_CALUDE_cats_in_academy_l769_76990

/-- The number of cats that can jump -/
def jump : ℕ := 40

/-- The number of cats that can climb -/
def climb : ℕ := 25

/-- The number of cats that can hunt -/
def hunt : ℕ := 30

/-- The number of cats that can jump and climb -/
def jump_and_climb : ℕ := 10

/-- The number of cats that can climb and hunt -/
def climb_and_hunt : ℕ := 15

/-- The number of cats that can jump and hunt -/
def jump_and_hunt : ℕ := 12

/-- The number of cats that can do all three skills -/
def all_skills : ℕ := 5

/-- The number of cats that cannot perform any skills -/
def no_skills : ℕ := 6

/-- The total number of cats in the academy -/
def total_cats : ℕ := 69

theorem cats_in_academy :
  total_cats = jump + climb + hunt - jump_and_climb - climb_and_hunt - jump_and_hunt + all_skills + no_skills := by
  sorry

end NUMINAMATH_CALUDE_cats_in_academy_l769_76990


namespace NUMINAMATH_CALUDE_ab_difference_l769_76989

theorem ab_difference (A B C : ℕ) : 
  A > 0 → B > 0 → C > 0 →
  (1212017 * 100 * A + 1212017 * 10 * B + 1212017 * C) % 45 = 0 →
  ∃ (max_AB min_AB : ℕ),
    (∀ A' B' C' : ℕ, 
      A' > 0 → B' > 0 → C' > 0 →
      (1212017 * 100 * A' + 1212017 * 10 * B' + 1212017 * C') % 45 = 0 →
      A' * 10 + B' ≤ max_AB) ∧
    (∀ A' B' C' : ℕ, 
      A' > 0 → B' > 0 → C' > 0 →
      (1212017 * 100 * A' + 1212017 * 10 * B' + 1212017 * C') % 45 = 0 →
      A' * 10 + B' ≥ min_AB) ∧
    max_AB - min_AB = 85 :=
by sorry

end NUMINAMATH_CALUDE_ab_difference_l769_76989


namespace NUMINAMATH_CALUDE_hyperbola_equation_l769_76950

theorem hyperbola_equation (a b p x₀ : ℝ) : 
  a > 0 → b > 0 → p > 0 →
  (b / a = 2) →
  (p / 2 = 4 / 3) →
  (x₀ = 3) →
  (16 = 2 * p * x₀) →
  (9 / a^2 - 16 / b^2 = 1) →
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 5 - y^2 / 20 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l769_76950


namespace NUMINAMATH_CALUDE_prob_two_queens_or_at_least_one_jack_is_9_221_l769_76991

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_jacks : ℕ)
  (num_queens : ℕ)

/-- Calculates the probability of the given event -/
def probability_two_queens_or_at_least_one_jack (d : Deck) : ℚ :=
  sorry

/-- Theorem stating the probability of drawing either two queens or at least one jack -/
theorem prob_two_queens_or_at_least_one_jack_is_9_221 :
  let standard_deck : Deck := ⟨52, 1, 3⟩
  probability_two_queens_or_at_least_one_jack standard_deck = 9/221 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_queens_or_at_least_one_jack_is_9_221_l769_76991


namespace NUMINAMATH_CALUDE_triangle_inequality_theorem_l769_76962

/-- Given a triangle ABC with side lengths a, b, c, and a point M inside it,
    Ra, Rb, Rc are distances from M to sides BC, CA, AB respectively,
    da, db, dc are perpendicular distances from vertices A, B, C to the line through M parallel to the opposite sides. -/
def triangle_inequality (a b c Ra Rb Rc da db dc : ℝ) : Prop :=
  a * Ra + b * Rb + c * Rc ≥ 2 * (a * da + b * db + c * dc)

/-- M is the orthocenter of triangle ABC -/
def is_orthocenter (M : Point) (A B C : Point) : Prop := sorry

theorem triangle_inequality_theorem 
  (A B C M : Point) (a b c Ra Rb Rc da db dc : ℝ) :
  triangle_inequality a b c Ra Rb Rc da db dc ∧ 
  (triangle_inequality a b c Ra Rb Rc da db dc = (a * Ra + b * Rb + c * Rc = 2 * (a * da + b * db + c * dc)) ↔ 
   is_orthocenter M A B C) := by sorry

end NUMINAMATH_CALUDE_triangle_inequality_theorem_l769_76962


namespace NUMINAMATH_CALUDE_min_value_rational_function_l769_76925

theorem min_value_rational_function (x : ℝ) (h : x > 6) :
  (x^2 + 12*x) / (x - 6) ≥ 30 ∧
  ((x^2 + 12*x) / (x - 6) = 30 ↔ x = 12) :=
by sorry

end NUMINAMATH_CALUDE_min_value_rational_function_l769_76925


namespace NUMINAMATH_CALUDE_goat_average_price_l769_76998

/-- The average price of a goat given the total cost of cows and goats, and the average price of a cow -/
theorem goat_average_price
  (total_cost : ℕ)
  (num_cows : ℕ)
  (num_goats : ℕ)
  (cow_avg_price : ℕ)
  (h1 : total_cost = 1400)
  (h2 : num_cows = 2)
  (h3 : num_goats = 8)
  (h4 : cow_avg_price = 460) :
  (total_cost - num_cows * cow_avg_price) / num_goats = 60 := by
  sorry

end NUMINAMATH_CALUDE_goat_average_price_l769_76998


namespace NUMINAMATH_CALUDE_largest_multiple_of_11_under_100_l769_76995

theorem largest_multiple_of_11_under_100 : ∃ n : ℕ, n * 11 = 99 ∧ n * 11 < 100 ∧ ∀ m : ℕ, m * 11 < 100 → m * 11 ≤ 99 := by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_11_under_100_l769_76995


namespace NUMINAMATH_CALUDE_circumscribed_sphere_radius_is_four_l769_76952

/-- Represents a triangular pyramid with specific dimensions -/
structure TriangularPyramid where
  base_side_length : ℝ
  perpendicular_edge_length : ℝ

/-- Calculates the radius of the circumscribed sphere around a triangular pyramid -/
def circumscribed_sphere_radius (pyramid : TriangularPyramid) : ℝ :=
  sorry

/-- Theorem stating that the radius of the circumscribed sphere is 4 for the given pyramid -/
theorem circumscribed_sphere_radius_is_four :
  let pyramid : TriangularPyramid := { base_side_length := 6, perpendicular_edge_length := 4 }
  circumscribed_sphere_radius pyramid = 4 := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_radius_is_four_l769_76952


namespace NUMINAMATH_CALUDE_ratio_sum_theorem_l769_76958

theorem ratio_sum_theorem (a b c : ℕ+) 
  (h1 : (a : ℚ) / b = 3 / 4)
  (h2 : (b : ℚ) / c = 5 / 6)
  (h3 : a + b + c = 1680) :
  a = 426 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_theorem_l769_76958


namespace NUMINAMATH_CALUDE_at_least_two_inequalities_false_l769_76966

theorem at_least_two_inequalities_false (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  ¬(((x + y > 0) ∧ (y + z > 0) ∧ (z + x > 0) ∧ (x + 2*y < 0) ∧ (y + 2*z < 0)) ∨
    ((x + y > 0) ∧ (y + z > 0) ∧ (z + x > 0) ∧ (x + 2*y < 0) ∧ (z + 2*x < 0)) ∨
    ((x + y > 0) ∧ (y + z > 0) ∧ (z + x > 0) ∧ (y + 2*z < 0) ∧ (z + 2*x < 0)) ∨
    ((x + y > 0) ∧ (y + z > 0) ∧ (x + 2*y < 0) ∧ (y + 2*z < 0) ∧ (z + 2*x < 0)) ∨
    ((x + y > 0) ∧ (z + x > 0) ∧ (x + 2*y < 0) ∧ (y + 2*z < 0) ∧ (z + 2*x < 0)) ∨
    ((y + z > 0) ∧ (z + x > 0) ∧ (x + 2*y < 0) ∧ (y + 2*z < 0) ∧ (z + 2*x < 0))) :=
by
  sorry


end NUMINAMATH_CALUDE_at_least_two_inequalities_false_l769_76966


namespace NUMINAMATH_CALUDE_tv_sales_effect_l769_76953

theorem tv_sales_effect (price_reduction : Real) (sales_increase : Real) :
  price_reduction = 0.18 →
  sales_increase = 0.72 →
  let new_price_factor := 1 - price_reduction
  let new_sales_factor := 1 + sales_increase
  let net_effect := new_price_factor * new_sales_factor - 1
  net_effect * 100 = 41.04 := by
  sorry

end NUMINAMATH_CALUDE_tv_sales_effect_l769_76953


namespace NUMINAMATH_CALUDE_line_slope_angle_l769_76923

theorem line_slope_angle (a : ℝ) : 
  (∃ (x y : ℝ), a * x - y - 1 = 0) → -- Line equation
  (Real.tan (π / 3) = a) →           -- Slope angle condition
  a = Real.sqrt 3 :=                 -- Conclusion
by sorry

end NUMINAMATH_CALUDE_line_slope_angle_l769_76923


namespace NUMINAMATH_CALUDE_expression_value_l769_76975

theorem expression_value : 
  |1 - Real.sqrt 3| - 2 * Real.sin (π / 3) + (π - 2023) ^ 0 = 0 := by sorry

end NUMINAMATH_CALUDE_expression_value_l769_76975


namespace NUMINAMATH_CALUDE_function_increasing_iff_a_geq_neg_three_l769_76937

/-- The function f(x) = x^2 + 2(a-1)x + 2 is increasing on [4, +∞) if and only if a ≥ -3 -/
theorem function_increasing_iff_a_geq_neg_three (a : ℝ) :
  (∀ x ≥ 4, Monotone (fun x => x^2 + 2*(a-1)*x + 2)) ↔ a ≥ -3 := by
  sorry

end NUMINAMATH_CALUDE_function_increasing_iff_a_geq_neg_three_l769_76937


namespace NUMINAMATH_CALUDE_shirts_made_today_proof_l769_76984

/-- Calculates the number of shirts made today given the production rate,
    yesterday's working time, and the total number of shirts made. -/
def shirts_made_today (rate : ℕ) (yesterday_time : ℕ) (total_shirts : ℕ) : ℕ :=
  total_shirts - (rate * yesterday_time)

/-- Proves that the number of shirts made today is 84 given the specified conditions. -/
theorem shirts_made_today_proof :
  shirts_made_today 6 12 156 = 84 := by
  sorry

end NUMINAMATH_CALUDE_shirts_made_today_proof_l769_76984


namespace NUMINAMATH_CALUDE_at_least_one_correct_l769_76915

theorem at_least_one_correct (pA pB : ℝ) (hA : pA = 0.8) (hB : pB = 0.9) :
  1 - (1 - pA) * (1 - pB) = 0.98 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_correct_l769_76915


namespace NUMINAMATH_CALUDE_quadratic_equation_q_value_l769_76965

theorem quadratic_equation_q_value 
  (p q : ℝ) 
  (h : ∃ x : ℂ, 3 * x^2 + p * x + q = 0 ∧ x = 4 + 3*I) : 
  q = 75 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_q_value_l769_76965


namespace NUMINAMATH_CALUDE_halloween_goodie_bags_l769_76943

/-- Calculates the minimum cost for buying a given number of items,
    where packs of 5 cost $3 and individual items cost $1 each. -/
def minCost (n : ℕ) : ℕ :=
  (n / 5) * 3 + (n % 5)

/-- The Halloween goodie bag problem -/
theorem halloween_goodie_bags :
  let vampireBags := 11
  let pumpkinBags := 14
  let totalBags := vampireBags + pumpkinBags
  totalBags = 25 →
  minCost vampireBags + minCost pumpkinBags = 17 := by
sorry

end NUMINAMATH_CALUDE_halloween_goodie_bags_l769_76943


namespace NUMINAMATH_CALUDE_paramon_solomon_meeting_time_l769_76902

/- Define the total distance between A and B -/
variable (S : ℝ) (S_pos : S > 0)

/- Define the speeds of Paramon, Solomon, and Agafon -/
variable (x y z : ℝ) (x_pos : x > 0) (y_pos : y > 0) (z_pos : z > 0)

/- Define the time when Paramon and Solomon meet -/
def meeting_time : ℝ := 1

/- Theorem stating that Paramon and Solomon meet at 13:00 (1 hour after 12:00) -/
theorem paramon_solomon_meeting_time :
  (S / (2 * x) = 1) ∧                   /- Paramon travels half the distance in 1 hour -/
  (2 * z = S / 2 + 2 * x) ∧             /- Agafon catches up with Paramon at 14:00 -/
  (4 / 3 * (y + z) = S) ∧               /- Agafon meets Solomon at 13:20 -/
  (S / 2 + x * meeting_time = y * meeting_time) /- Paramon and Solomon meet -/
  → meeting_time = 1 := by sorry

end NUMINAMATH_CALUDE_paramon_solomon_meeting_time_l769_76902


namespace NUMINAMATH_CALUDE_wire_length_problem_l769_76969

theorem wire_length_problem (shorter_piece longer_piece total_length : ℝ) : 
  shorter_piece = 20 →
  longer_piece = 2/4 * shorter_piece →
  total_length = shorter_piece + longer_piece →
  total_length = 60 := by
sorry

end NUMINAMATH_CALUDE_wire_length_problem_l769_76969


namespace NUMINAMATH_CALUDE_consecutive_odd_sum_2005_2006_l769_76904

theorem consecutive_odd_sum_2005_2006 :
  (∃ (n k : ℕ), n ≥ 2 ∧ 2005 = n * (2 * k + n)) ∧
  (¬ ∃ (n k : ℕ), n ≥ 2 ∧ 2006 = n * (2 * k + n)) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_sum_2005_2006_l769_76904


namespace NUMINAMATH_CALUDE_truncated_cone_radius_l769_76934

/-- Represents a cone with its base radius -/
structure Cone :=
  (radius : ℝ)

/-- Represents a truncated cone with its smaller base radius -/
structure TruncatedCone :=
  (smallerRadius : ℝ)

/-- 
  Given three touching cones and a truncated cone sharing a common generatrix with each,
  the radius of the smaller base of the truncated cone is 6.
-/
theorem truncated_cone_radius 
  (cone1 cone2 cone3 : Cone) 
  (truncCone : TruncatedCone) 
  (h1 : cone1.radius = 23) 
  (h2 : cone2.radius = 46) 
  (h3 : cone3.radius = 69) 
  (h4 : ∃ (x y : ℝ), 
    (x^2 + y^2 = (cone1.radius + truncCone.smallerRadius)^2) ∧ 
    ((x - (cone1.radius + cone2.radius))^2 + y^2 = (cone2.radius + truncCone.smallerRadius)^2) ∧
    (x^2 + (y - (cone1.radius + cone3.radius))^2 = (cone3.radius + truncCone.smallerRadius)^2)) :
  truncCone.smallerRadius = 6 := by
  sorry

end NUMINAMATH_CALUDE_truncated_cone_radius_l769_76934


namespace NUMINAMATH_CALUDE_isabel_savings_l769_76936

def initial_amount : ℚ := 204
def toy_fraction : ℚ := 1/2
def book_fraction : ℚ := 1/2

theorem isabel_savings : 
  initial_amount * (1 - toy_fraction) * (1 - book_fraction) = 51 := by
  sorry

end NUMINAMATH_CALUDE_isabel_savings_l769_76936


namespace NUMINAMATH_CALUDE_product_expansion_sum_l769_76931

theorem product_expansion_sum (a b c d : ℝ) : 
  (∀ x, (5*x^2 - 3*x + 2)*(9 - 3*x) = a*x^3 + b*x^2 + c*x + d) →
  27*a + 9*b + 3*c + d = 0 := by
sorry

end NUMINAMATH_CALUDE_product_expansion_sum_l769_76931


namespace NUMINAMATH_CALUDE_point_on_x_axis_l769_76960

/-- A point in the 2D coordinate plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The x-axis in the 2D coordinate plane -/
def xAxis : Set Point2D := {p : Point2D | p.y = 0}

/-- Theorem: A point P(x,0) lies on the x-axis -/
theorem point_on_x_axis (x : ℝ) : 
  Point2D.mk x 0 ∈ xAxis := by sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l769_76960


namespace NUMINAMATH_CALUDE_at_least_one_positive_l769_76921

theorem at_least_one_positive (x y z : ℝ) : 
  (x^2 - 2*y + Real.pi/2 > 0) ∨ 
  (y^2 - 2*z + Real.pi/3 > 0) ∨ 
  (z^2 - 2*x + Real.pi/6 > 0) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_positive_l769_76921


namespace NUMINAMATH_CALUDE_inequality_solution_equivalence_l769_76988

-- Define the types for our variables
variable (x : ℝ)
variable (a b : ℝ)

-- Define the original inequality and its solution set
def original_inequality (x a b : ℝ) : Prop := a * (x + b) * (x + 5 / a) > 0
def original_solution_set (x : ℝ) : Prop := x < -1 ∨ x > 3

-- Define the new inequality we want to solve
def new_inequality (x a b : ℝ) : Prop := x^2 + b*x - 2*a < 0

-- Define the solution set we want to prove
def target_solution_set (x : ℝ) : Prop := x > -2 ∧ x < 5

-- State the theorem
theorem inequality_solution_equivalence :
  (∀ x, original_inequality x a b ↔ original_solution_set x) →
  (∀ x, new_inequality x a b ↔ target_solution_set x) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_equivalence_l769_76988
