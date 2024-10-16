import Mathlib

namespace NUMINAMATH_CALUDE_mitzel_allowance_percentage_l2422_242230

theorem mitzel_allowance_percentage (spent : ℝ) (left : ℝ) : 
  spent = 14 → left = 26 → (spent / (spent + left)) * 100 = 35 := by
  sorry

end NUMINAMATH_CALUDE_mitzel_allowance_percentage_l2422_242230


namespace NUMINAMATH_CALUDE_original_room_population_l2422_242261

theorem original_room_population (x : ℚ) : 
  (x / 4 : ℚ) - (x / 12 : ℚ) = 15 → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_original_room_population_l2422_242261


namespace NUMINAMATH_CALUDE_ellipse_satisfies_equation_l2422_242224

/-- An ellipse with given properties -/
structure Ellipse where
  -- Foci coordinates
  f1 : ℝ × ℝ
  f2 : ℝ × ℝ
  -- Line passing through f2 perpendicular to x-axis
  line : Set (ℝ × ℝ)
  -- Intersection points of the line with the ellipse
  a : ℝ × ℝ
  b : ℝ × ℝ
  -- Distance between intersection points
  ab_distance : ℝ
  -- Properties
  f1_def : f1 = (-1, 0)
  f2_def : f2 = (1, 0)
  line_def : line = {p : ℝ × ℝ | p.1 = 1}
  ab_on_line : a ∈ line ∧ b ∈ line
  ab_distance_def : ab_distance = 3
  
/-- The equation of the ellipse -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

/-- Theorem stating that the given ellipse satisfies the equation -/
theorem ellipse_satisfies_equation (e : Ellipse) :
  ∀ x y, (x, y) ∈ {p : ℝ × ℝ | ellipse_equation e p.1 p.2} ↔ 
    (∃ t, 0 ≤ t ∧ t ≤ 1 ∧ 
      (x - e.f1.1)^2 + (y - e.f1.2)^2 + 
      (x - e.f2.1)^2 + (y - e.f2.2)^2 = 
      (2 * Real.sqrt ((x - e.f1.1)^2 + (y - e.f1.2)^2 + (x - e.f2.1)^2 + (y - e.f2.2)^2))^2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_satisfies_equation_l2422_242224


namespace NUMINAMATH_CALUDE_noah_sales_this_month_l2422_242248

/-- Represents Noah's painting sales --/
structure NoahSales where
  large_price : ℕ
  small_price : ℕ
  last_month_large : ℕ
  last_month_small : ℕ

/-- Calculates Noah's sales for this month --/
def this_month_sales (s : NoahSales) : ℕ :=
  2 * (s.large_price * s.last_month_large + s.small_price * s.last_month_small)

/-- Theorem: Noah's sales for this month equal $1200 --/
theorem noah_sales_this_month (s : NoahSales) 
  (h1 : s.large_price = 60)
  (h2 : s.small_price = 30)
  (h3 : s.last_month_large = 8)
  (h4 : s.last_month_small = 4) :
  this_month_sales s = 1200 := by
  sorry

end NUMINAMATH_CALUDE_noah_sales_this_month_l2422_242248


namespace NUMINAMATH_CALUDE_fraction_of_decimals_cubed_and_squared_l2422_242298

theorem fraction_of_decimals_cubed_and_squared :
  (0.3 ^ 3) / (0.03 ^ 2) = 30 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_decimals_cubed_and_squared_l2422_242298


namespace NUMINAMATH_CALUDE_quadratic_inequality_has_solution_l2422_242209

theorem quadratic_inequality_has_solution : ∃ x : ℝ, x^2 + 2*x - 3 < 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_has_solution_l2422_242209


namespace NUMINAMATH_CALUDE_sequential_search_comparisons_l2422_242243

/-- Represents a sequential search on an unordered array. -/
structure SequentialSearch where
  array_size : Nat
  element_not_present : Bool
  unordered : Bool

/-- The number of comparisons needed for a sequential search. -/
def comparisons_needed (search : SequentialSearch) : Nat :=
  search.array_size

/-- Theorem: The number of comparisons for a sequential search on an unordered array
    of 100 elements, where the element is not present, is 100. -/
theorem sequential_search_comparisons :
  ∀ (search : SequentialSearch),
    search.array_size = 100 →
    search.element_not_present = true →
    search.unordered = true →
    comparisons_needed search = 100 := by
  sorry

end NUMINAMATH_CALUDE_sequential_search_comparisons_l2422_242243


namespace NUMINAMATH_CALUDE_window_side_length_l2422_242210

/-- Represents the dimensions of a window pane -/
structure Pane where
  width : ℝ
  height : ℝ
  ratio : height = 3 * width

/-- Represents the dimensions of a square window -/
structure SquareWindow where
  pane : Pane
  borderWidth : ℝ
  sideLength : ℝ
  paneCount : ℕ
  isSquare : sideLength = 3 * pane.width + 4 * borderWidth
  hasPanes : paneCount = 9

/-- Theorem: The side length of the square window is 20 inches -/
theorem window_side_length (w : SquareWindow) (h : w.borderWidth = 2) : w.sideLength = 20 :=
by sorry

end NUMINAMATH_CALUDE_window_side_length_l2422_242210


namespace NUMINAMATH_CALUDE_ab_length_l2422_242283

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the conditions
axiom collinear : ∃ (t : ℝ), B = A + t • (D - A) ∧ C = A + t • (D - A)
axiom ab_eq_cd : dist A B = dist C D
axiom bc_length : dist B C = 15
axiom e_not_on_line : ¬∃ (t : ℝ), E = A + t • (D - A)
axiom be_eq_ce : dist B E = dist C E ∧ dist B E = 13

-- Define the perimeter function
def perimeter (X Y Z : ℝ × ℝ) : ℝ := dist X Y + dist Y Z + dist Z X

-- State the theorem
theorem ab_length :
  perimeter A E D = 1.5 * perimeter B E C →
  dist A B = 6.04 := by sorry

end NUMINAMATH_CALUDE_ab_length_l2422_242283


namespace NUMINAMATH_CALUDE_a_power_m_plus_2n_l2422_242214

theorem a_power_m_plus_2n (a : ℝ) (m n : ℤ) (h1 : a^m = 2) (h2 : a^n = 3) :
  a^(m + 2*n) = 18 := by
  sorry

end NUMINAMATH_CALUDE_a_power_m_plus_2n_l2422_242214


namespace NUMINAMATH_CALUDE_line_symmetry_l2422_242273

-- Define the original line
def original_line (x y : ℝ) : Prop := x * y + 1 = 0

-- Define the axis of symmetry
def symmetry_axis (x : ℝ) : Prop := x = 1

-- Define the symmetric line
def symmetric_line (x y : ℝ) : Prop := x + 2 * y - 3 = 0

-- Theorem statement
theorem line_symmetry :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    original_line x₁ y₁ →
    symmetry_axis ((x₁ + x₂) / 2) →
    symmetric_line x₂ y₂ →
    y₁ = y₂ ∧ x₁ + x₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_symmetry_l2422_242273


namespace NUMINAMATH_CALUDE_irrationality_of_sqrt_two_and_rationality_of_others_l2422_242274

theorem irrationality_of_sqrt_two_and_rationality_of_others : 
  (∃ (a b : ℤ), (a : ℝ) / (b : ℝ) = Real.sqrt 2) ∧ 
  (∃ (c d : ℤ), (c : ℝ) / (d : ℝ) = 3.14) ∧
  (∃ (e f : ℤ), (e : ℝ) / (f : ℝ) = -2) ∧
  (∃ (g h : ℤ), (g : ℝ) / (h : ℝ) = 1/3) ∧
  (¬∃ (i j : ℤ), (i : ℝ) / (j : ℝ) = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_irrationality_of_sqrt_two_and_rationality_of_others_l2422_242274


namespace NUMINAMATH_CALUDE_quadratic_sum_l2422_242203

/-- A quadratic function with vertex at (2, 5) passing through (3, 2) -/
def QuadraticFunction (d e f : ℝ) : ℝ → ℝ :=
  fun x ↦ d * x^2 + e * x + f

theorem quadratic_sum (d e f : ℝ) :
  (QuadraticFunction d e f 2 = 5) →
  (QuadraticFunction d e f 3 = 2) →
  d + e + 2*f = -5 := by
    sorry

end NUMINAMATH_CALUDE_quadratic_sum_l2422_242203


namespace NUMINAMATH_CALUDE_icosikaipentagon_diagonals_l2422_242218

/-- The number of diagonals that can be drawn from a single vertex of an n-sided polygon -/
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

theorem icosikaipentagon_diagonals :
  diagonals_from_vertex 25 = 22 :=
by sorry

end NUMINAMATH_CALUDE_icosikaipentagon_diagonals_l2422_242218


namespace NUMINAMATH_CALUDE_parabola_focus_line_intersection_l2422_242244

/-- Represents a parabola with equation y² = 2px where p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Represents a line passing through the focus of a parabola -/
structure FocusLine where
  angle : ℝ
  h_angle_eq : angle = π / 4

/-- Represents the intersection points of a line with a parabola -/
structure Intersection where
  A : ℝ × ℝ
  B : ℝ × ℝ

/-- The main theorem -/
theorem parabola_focus_line_intersection
  (para : Parabola) (line : FocusLine) (inter : Intersection) :
  let midpoint_x := (inter.A.1 + inter.B.1) / 2
  let axis_distance := midpoint_x - para.p / 2
  axis_distance = 4 → para.p = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_line_intersection_l2422_242244


namespace NUMINAMATH_CALUDE_coefficient_x3y0_l2422_242262

/-- The coefficient of x^m * y^n in the expansion of (1+x)^6 * (1+y)^4 -/
def f (m n : ℕ) : ℕ :=
  (Nat.choose 6 m) * (Nat.choose 4 n)

theorem coefficient_x3y0 : f 3 0 = 20 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x3y0_l2422_242262


namespace NUMINAMATH_CALUDE_infinite_representable_theorem_l2422_242215

-- Define an increasing sequence of positive integers
def IncreasingSequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

-- Define the property we want to prove
def InfinitelyRepresentable (a : ℕ → ℕ) : Prop :=
  ∀ i : ℕ, ∀ k : ℕ, ∃ n > k, ∃ j > i, ∃ r s : ℕ+, a n = r * a i + s * a j

-- State the theorem
theorem infinite_representable_theorem (a : ℕ → ℕ) (h : IncreasingSequence a) :
  InfinitelyRepresentable a := by
  sorry

end NUMINAMATH_CALUDE_infinite_representable_theorem_l2422_242215


namespace NUMINAMATH_CALUDE_alcohol_concentration_second_vessel_l2422_242216

/-- Proves that the initial concentration of alcohol in the second vessel is 60% --/
theorem alcohol_concentration_second_vessel :
  let vessel1_capacity : ℝ := 2
  let vessel1_alcohol_percentage : ℝ := 40
  let vessel2_capacity : ℝ := 6
  let total_liquid : ℝ := 8
  let final_vessel_capacity : ℝ := 10
  let final_mixture_percentage : ℝ := 44
  let vessel2_alcohol_percentage : ℝ := 
    (final_mixture_percentage * final_vessel_capacity - vessel1_alcohol_percentage * vessel1_capacity) / vessel2_capacity
  vessel2_alcohol_percentage = 60 := by
sorry

end NUMINAMATH_CALUDE_alcohol_concentration_second_vessel_l2422_242216


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l2422_242292

theorem geometric_series_ratio (a r : ℝ) (h1 : |r| < 1) : 
  (a / (1 - r) = 15) → 
  (a / (1 - r^2) = 6) → 
  r = 2/3 := by sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l2422_242292


namespace NUMINAMATH_CALUDE_shirt_cost_l2422_242296

theorem shirt_cost (initial_amount : ℕ) (socks_cost : ℕ) (amount_left : ℕ) :
  initial_amount = 100 →
  socks_cost = 11 →
  amount_left = 65 →
  initial_amount - amount_left - socks_cost = 24 := by
sorry

end NUMINAMATH_CALUDE_shirt_cost_l2422_242296


namespace NUMINAMATH_CALUDE_deal_or_no_deal_probability_l2422_242220

def box_values : List ℕ := [10, 50, 100, 500, 1000, 5000, 50000, 75000, 200000, 400000, 500000, 1000000]

def total_boxes : ℕ := 16

def high_value_boxes : ℕ := (box_values.filter (λ x => x ≥ 500000)).length

theorem deal_or_no_deal_probability (boxes_to_eliminate : ℕ) :
  boxes_to_eliminate = 10 ↔ 
  (high_value_boxes : ℚ) / (total_boxes - boxes_to_eliminate : ℚ) ≥ 1/2 ∧
  ∀ n : ℕ, n < boxes_to_eliminate → 
    (high_value_boxes : ℚ) / (total_boxes - n : ℚ) < 1/2 :=
sorry

end NUMINAMATH_CALUDE_deal_or_no_deal_probability_l2422_242220


namespace NUMINAMATH_CALUDE_all_heads_possible_l2422_242222

/-- Represents the state of a coin (heads or tails) -/
inductive CoinState
| Heads
| Tails

/-- Represents the state of all coins in a row -/
def CoinRow := Vector CoinState 100

/-- An operation that flips 7 equally spaced coins -/
def FlipOperation := Fin 100 → Fin 7 → Bool

/-- Applies a flip operation to a coin row -/
def applyFlip (row : CoinRow) (op : FlipOperation) : CoinRow :=
  sorry

/-- The theorem stating that any initial coin configuration can be transformed to all heads -/
theorem all_heads_possible (initial : CoinRow) : 
  ∃ (ops : List FlipOperation), 
    let final := ops.foldl applyFlip initial
    ∀ i, final.get i = CoinState.Heads :=
  sorry

end NUMINAMATH_CALUDE_all_heads_possible_l2422_242222


namespace NUMINAMATH_CALUDE_inequality_proof_l2422_242278

theorem inequality_proof (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (habc : a * b * c = 1) : 
  (a - 1) / c + (c - 1) / b + (b - 1) / a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2422_242278


namespace NUMINAMATH_CALUDE_remaining_cakes_l2422_242231

theorem remaining_cakes (initial_cakes sold_cakes : ℝ) 
  (h1 : initial_cakes = 167.3)
  (h2 : sold_cakes = 108.2) :
  initial_cakes - sold_cakes = 59.1 := by
  sorry

end NUMINAMATH_CALUDE_remaining_cakes_l2422_242231


namespace NUMINAMATH_CALUDE_max_k_logarithmic_inequality_l2422_242271

theorem max_k_logarithmic_inequality (x₀ x₁ x₂ x₃ : ℝ) (h₀ : x₀ > x₁) (h₁ : x₁ > x₂) (h₂ : x₂ > x₃) (h₃ : x₃ > 0) :
  ∃ k : ℝ, k = 9 ∧ 
  ∀ k' : ℝ, k' > k → 
  ∃ x₀' x₁' x₂' x₃' : ℝ, x₀' > x₁' ∧ x₁' > x₂' ∧ x₂' > x₃' ∧ x₃' > 0 ∧
  (Real.log (x₀' / x₁') / Real.log (x₀ / x₁) + 
   Real.log (x₁' / x₂') / Real.log (x₁ / x₂) + 
   Real.log (x₂' / x₃') / Real.log (x₂ / x₃) ≤ 
   k' * Real.log (x₀' / x₃') / Real.log (x₀ / x₃)) :=
by sorry

end NUMINAMATH_CALUDE_max_k_logarithmic_inequality_l2422_242271


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2422_242279

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y + y * f x) = f x + f y + x * f y) →
  (∀ x : ℝ, f x = 0 ∨ f x = x) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2422_242279


namespace NUMINAMATH_CALUDE_candy_distribution_l2422_242206

/-- Given 27.5 candy bars divided among 8.3 people, each person receives approximately 3.313 candy bars -/
theorem candy_distribution (total_candy : ℝ) (num_people : ℝ) (candy_per_person : ℝ) 
  (h1 : total_candy = 27.5)
  (h2 : num_people = 8.3)
  (h3 : candy_per_person = total_candy / num_people) :
  ∃ ε > 0, |candy_per_person - 3.313| < ε :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_l2422_242206


namespace NUMINAMATH_CALUDE_unique_solution_to_equation_l2422_242299

theorem unique_solution_to_equation :
  ∀ x y z : ℝ,
  x^2 + 2*x + y^2 + 4*y + z^2 + 6*z = -14 →
  x = -1 ∧ y = -2 ∧ z = -3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_to_equation_l2422_242299


namespace NUMINAMATH_CALUDE_is_arithmetic_sequence_pn_plus_q_l2422_242257

/-- A sequence is arithmetic if the difference between consecutive terms is constant. -/
def IsArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The general term of the sequence. -/
def a (n : ℕ) (p q : ℝ) : ℝ := p * n + q

/-- Theorem: A sequence with general term a_n = pn + q is an arithmetic sequence. -/
theorem is_arithmetic_sequence_pn_plus_q (p q : ℝ) :
  IsArithmeticSequence (a · p q) := by
  sorry

end NUMINAMATH_CALUDE_is_arithmetic_sequence_pn_plus_q_l2422_242257


namespace NUMINAMATH_CALUDE_specific_ellipse_foci_distance_l2422_242290

/-- An ellipse with axes parallel to the coordinate axes -/
structure ParallelAxisEllipse where
  /-- The point where the ellipse is tangent to the x-axis -/
  x_tangent : ℝ × ℝ
  /-- The point where the ellipse is tangent to the y-axis -/
  y_tangent : ℝ × ℝ

/-- The distance between the foci of an ellipse -/
def foci_distance (e : ParallelAxisEllipse) : ℝ := sorry

/-- Theorem stating the distance between foci for a specific ellipse -/
theorem specific_ellipse_foci_distance :
  ∃ (e : ParallelAxisEllipse),
    e.x_tangent = (3, 0) ∧
    e.y_tangent = (0, 2) ∧
    foci_distance e = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_specific_ellipse_foci_distance_l2422_242290


namespace NUMINAMATH_CALUDE_norma_cards_l2422_242259

/-- Given that Norma has 88.0 cards initially and finds 70.0 more cards,
    prove that she will have 158.0 cards in total. -/
theorem norma_cards (initial_cards : Float) (found_cards : Float)
    (h1 : initial_cards = 88.0)
    (h2 : found_cards = 70.0) :
  initial_cards + found_cards = 158.0 := by
  sorry

end NUMINAMATH_CALUDE_norma_cards_l2422_242259


namespace NUMINAMATH_CALUDE_nineteen_customers_without_fish_l2422_242276

/-- Represents the fish market scenario --/
structure FishMarket where
  total_customers : ℕ
  tuna_count : ℕ
  tuna_weight : ℕ
  regular_customer_request : ℕ
  special_customer_30lb : ℕ
  special_customer_20lb : ℕ
  max_cuts_per_tuna : ℕ

/-- Calculates the number of customers who will go home without fish --/
def customers_without_fish (market : FishMarket) : ℕ :=
  let total_weight := market.tuna_count * market.tuna_weight
  let weight_for_30lb := market.special_customer_30lb * 30
  let weight_for_20lb := market.special_customer_20lb * 20
  let remaining_weight := total_weight - weight_for_30lb - weight_for_20lb
  let remaining_customers := remaining_weight / market.regular_customer_request
  let total_served := market.special_customer_30lb + market.special_customer_20lb + remaining_customers
  market.total_customers - total_served

/-- Theorem stating that 19 customers will go home without fish --/
theorem nineteen_customers_without_fish (market : FishMarket) 
  (h1 : market.total_customers = 100)
  (h2 : market.tuna_count = 10)
  (h3 : market.tuna_weight = 200)
  (h4 : market.regular_customer_request = 25)
  (h5 : market.special_customer_30lb = 10)
  (h6 : market.special_customer_20lb = 15)
  (h7 : market.max_cuts_per_tuna = 8) :
  customers_without_fish market = 19 := by
  sorry

end NUMINAMATH_CALUDE_nineteen_customers_without_fish_l2422_242276


namespace NUMINAMATH_CALUDE_kevin_watermelon_weight_l2422_242211

theorem kevin_watermelon_weight :
  let first_watermelon : ℝ := 9.91
  let second_watermelon : ℝ := 4.11
  let total_weight : ℝ := first_watermelon + second_watermelon
  total_weight = 14.02 := by sorry

end NUMINAMATH_CALUDE_kevin_watermelon_weight_l2422_242211


namespace NUMINAMATH_CALUDE_collinear_points_k_value_l2422_242207

/-- Given three points on a line, prove the value of k --/
theorem collinear_points_k_value (k : ℚ) : 
  (∃ (m b : ℚ), 8 = m * 2 + b ∧ k = m * 10 + b ∧ 2 = m * 16 + b) → k = 32/7 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_k_value_l2422_242207


namespace NUMINAMATH_CALUDE_f_max_value_l2422_242208

noncomputable def f (x : ℝ) : ℝ := x^6 / (x^12 + 3*x^9 - 6*x^6 + 12*x^3 + 27)

theorem f_max_value :
  (∀ x : ℝ, f x ≤ 1 / (12 * Real.sqrt 3)) ∧
  (∃ x : ℝ, f x = 1 / (12 * Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_f_max_value_l2422_242208


namespace NUMINAMATH_CALUDE_average_pages_of_books_l2422_242267

theorem average_pages_of_books (books : List ℕ) (h : books = [120, 150, 180, 210, 240]) : 
  (books.sum / books.length : ℚ) = 180 := by
  sorry

end NUMINAMATH_CALUDE_average_pages_of_books_l2422_242267


namespace NUMINAMATH_CALUDE_max_sum_product_sqrt_l2422_242217

theorem max_sum_product_sqrt (x₁ x₂ x₃ x₄ : ℝ) 
  (non_neg : x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₃ ≥ 0 ∧ x₄ ≥ 0) 
  (sum_one : x₁ + x₂ + x₃ + x₄ = 1) :
  (x₁ + x₂) * Real.sqrt (x₁ * x₂) +
  (x₁ + x₃) * Real.sqrt (x₁ * x₃) +
  (x₁ + x₄) * Real.sqrt (x₁ * x₄) +
  (x₂ + x₃) * Real.sqrt (x₂ * x₃) +
  (x₂ + x₄) * Real.sqrt (x₂ * x₄) +
  (x₃ + x₄) * Real.sqrt (x₃ * x₄) ≤ 3/4 ∧
  (x₁ = 1/4 ∧ x₂ = 1/4 ∧ x₃ = 1/4 ∧ x₄ = 1/4 →
    (x₁ + x₂) * Real.sqrt (x₁ * x₂) +
    (x₁ + x₃) * Real.sqrt (x₁ * x₃) +
    (x₁ + x₄) * Real.sqrt (x₁ * x₄) +
    (x₂ + x₃) * Real.sqrt (x₂ * x₃) +
    (x₂ + x₄) * Real.sqrt (x₂ * x₄) +
    (x₃ + x₄) * Real.sqrt (x₃ * x₄) = 3/4) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_product_sqrt_l2422_242217


namespace NUMINAMATH_CALUDE_correct_observation_value_l2422_242284

theorem correct_observation_value 
  (n : ℕ) 
  (original_mean : ℚ) 
  (incorrect_value : ℚ) 
  (corrected_mean : ℚ) 
  (h1 : n = 50) 
  (h2 : original_mean = 30) 
  (h3 : incorrect_value = 23) 
  (h4 : corrected_mean = 30.5) : 
  (n : ℚ) * corrected_mean - ((n : ℚ) * original_mean - incorrect_value) = 48 :=
by sorry

end NUMINAMATH_CALUDE_correct_observation_value_l2422_242284


namespace NUMINAMATH_CALUDE_binary_1111_equals_15_l2422_242260

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binaryToDecimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of the number 15 -/
def binaryFifteen : List Bool := [true, true, true, true]

/-- Theorem stating that the binary representation "1111" is equal to 15 in decimal -/
theorem binary_1111_equals_15 : binaryToDecimal binaryFifteen = 15 := by
  sorry

end NUMINAMATH_CALUDE_binary_1111_equals_15_l2422_242260


namespace NUMINAMATH_CALUDE_square_to_rectangle_area_increase_l2422_242239

theorem square_to_rectangle_area_increase (a : ℝ) (h : a > 0) :
  let square_area := a * a
  let rectangle_length := a * 1.4
  let rectangle_breadth := a * 1.3
  let rectangle_area := rectangle_length * rectangle_breadth
  rectangle_area - square_area = 0.82 * square_area := by
sorry

end NUMINAMATH_CALUDE_square_to_rectangle_area_increase_l2422_242239


namespace NUMINAMATH_CALUDE_huang_yan_id_sum_l2422_242238

/-- Calculates the sum of digits in a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- Represents a student ID with a given year -/
def student_id (year : ℕ) : ℕ := year * 10000 + 1262

theorem huang_yan_id_sum (year : ℕ) (h : year ≥ 1000 ∧ year < 10000) : 
  sum_of_digits (student_id year) = 22 := by
sorry

end NUMINAMATH_CALUDE_huang_yan_id_sum_l2422_242238


namespace NUMINAMATH_CALUDE_max_hawthorns_l2422_242219

theorem max_hawthorns (x : ℕ) : 
  x > 100 ∧
  x % 3 = 1 ∧
  x % 4 = 2 ∧
  x % 5 = 3 ∧
  x % 6 = 4 →
  x ≤ 178 ∧ 
  ∃ y : ℕ, y > 100 ∧ 
    y % 3 = 1 ∧ 
    y % 4 = 2 ∧ 
    y % 5 = 3 ∧ 
    y % 6 = 4 ∧ 
    y = 178 :=
by sorry

end NUMINAMATH_CALUDE_max_hawthorns_l2422_242219


namespace NUMINAMATH_CALUDE_scaled_box_capacity_l2422_242282

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  height : ℝ
  width : ℝ
  length : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.height * d.width * d.length

/-- Theorem: A box with 3 times the height, 2 times the width, and 1/2 times the length of a box
    that can hold 60 grams of clay can hold 180 grams of clay -/
theorem scaled_box_capacity
  (first_box : BoxDimensions)
  (first_box_capacity : ℝ)
  (h_first_box_capacity : first_box_capacity = 60)
  (second_box : BoxDimensions)
  (h_second_box_height : second_box.height = 3 * first_box.height)
  (h_second_box_width : second_box.width = 2 * first_box.width)
  (h_second_box_length : second_box.length = 1/2 * first_box.length) :
  (boxVolume second_box / boxVolume first_box) * first_box_capacity = 180 := by
  sorry

end NUMINAMATH_CALUDE_scaled_box_capacity_l2422_242282


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l2422_242227

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1 / 3) :
  (1 / x + 1 / y) ≥ 9 + 6 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l2422_242227


namespace NUMINAMATH_CALUDE_recurring_decimal_to_fraction_l2422_242256

/-- Given that 0.overline{02} = 2/99, prove that 2.overline{06} = 68/33 -/
theorem recurring_decimal_to_fraction :
  (∃ (x : ℚ), x = 2 / 99 ∧ (∀ n : ℕ, (x * 10^(3*n) - ⌊x * 10^(3*n)⌋ = 0.02))) →
  (∃ (y : ℚ), y = 68 / 33 ∧ (∀ n : ℕ, (y - 2 - ⌊y - 2⌋ = 0.06))) :=
by sorry

end NUMINAMATH_CALUDE_recurring_decimal_to_fraction_l2422_242256


namespace NUMINAMATH_CALUDE_existence_of_a_sequence_l2422_242226

theorem existence_of_a_sequence (n : ℕ) (x : Fin n → Fin n → ℝ) 
  (h : ∀ (i j k : Fin n), x i j + x j k + x k i = 0) :
  ∃ (a : Fin n → ℝ), ∀ (i j : Fin n), x i j = a i - a j := by
  sorry

end NUMINAMATH_CALUDE_existence_of_a_sequence_l2422_242226


namespace NUMINAMATH_CALUDE_solve_equation_l2422_242236

theorem solve_equation (x : ℚ) : 5 * (x - 10) = 3 * (3 - 3 * x) + 9 → x = 34 / 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2422_242236


namespace NUMINAMATH_CALUDE_inverse_direct_variation_l2422_242275

theorem inverse_direct_variation (k c : ℝ) (x y z : ℝ) : 
  (5 * y = k / (x ^ 2)) →
  (3 * z = c * x) →
  (5 * 25 = k / (2 ^ 2)) →
  (x = 4) →
  (z = 6) →
  (y = 6.25) := by
  sorry

end NUMINAMATH_CALUDE_inverse_direct_variation_l2422_242275


namespace NUMINAMATH_CALUDE_cinnamon_tradition_duration_l2422_242266

/-- Represents the cinnamon ball tradition setup -/
structure CinnamonTradition where
  totalSocks : Nat
  extraSocks : Nat
  regularBalls : Nat
  extraBalls : Nat
  totalBalls : Nat

/-- Calculates the maximum number of full days the tradition can continue -/
def maxDays (ct : CinnamonTradition) : Nat :=
  ct.totalBalls / (ct.regularBalls * (ct.totalSocks - ct.extraSocks) + ct.extraBalls * ct.extraSocks)

/-- Theorem stating that for the given conditions, the tradition lasts 3 days -/
theorem cinnamon_tradition_duration :
  ∀ (ct : CinnamonTradition),
  ct.totalSocks = 9 →
  ct.extraSocks = 3 →
  ct.regularBalls = 2 →
  ct.extraBalls = 3 →
  ct.totalBalls = 75 →
  maxDays ct = 3 := by
  sorry

#eval maxDays { totalSocks := 9, extraSocks := 3, regularBalls := 2, extraBalls := 3, totalBalls := 75 }

end NUMINAMATH_CALUDE_cinnamon_tradition_duration_l2422_242266


namespace NUMINAMATH_CALUDE_correct_scientific_notation_l2422_242229

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Check if a ScientificNotation represents a given number -/
def represents (sn : ScientificNotation) (n : ℝ) : Prop :=
  sn.coefficient * (10 : ℝ) ^ sn.exponent = n

/-- The number we want to represent (5.81 million) -/
def target_number : ℝ := 5.81e6

/-- The proposed scientific notation representation -/
def proposed_notation : ScientificNotation :=
  { coefficient := 5.81
    exponent := 6
    coeff_range := by sorry }

theorem correct_scientific_notation :
  represents proposed_notation target_number :=
sorry

end NUMINAMATH_CALUDE_correct_scientific_notation_l2422_242229


namespace NUMINAMATH_CALUDE_marias_stamps_l2422_242225

theorem marias_stamps (S : ℕ) : 
  S > 1 ∧ 
  S % 9 = 1 ∧ 
  S % 10 = 1 ∧ 
  S % 11 = 1 ∧
  (∀ T : ℕ, T > 1 ∧ T % 9 = 1 ∧ T % 10 = 1 ∧ T % 11 = 1 → S ≤ T) → 
  S = 991 := by
sorry

end NUMINAMATH_CALUDE_marias_stamps_l2422_242225


namespace NUMINAMATH_CALUDE_min_x_prime_factorization_sum_l2422_242212

theorem min_x_prime_factorization_sum (x y p q : ℕ+) (e f : ℕ) : 
  (∀ x' y' : ℕ+, 13 * x'^7 = 19 * y'^17 → x ≤ x') →
  13 * x^7 = 19 * y^17 →
  x = p^e * q^f →
  p.val.Prime ∧ q.val.Prime →
  p + q + e + f = 44 := by
  sorry

end NUMINAMATH_CALUDE_min_x_prime_factorization_sum_l2422_242212


namespace NUMINAMATH_CALUDE_ceiling_sqrt_200_l2422_242289

theorem ceiling_sqrt_200 : ⌈Real.sqrt 200⌉ = 15 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_200_l2422_242289


namespace NUMINAMATH_CALUDE_max_value_of_f_l2422_242234

def f (x : ℝ) : ℝ := -(x + 1)^2 + 5

theorem max_value_of_f :
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x₀, f x₀ = M) ∧ M = 5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2422_242234


namespace NUMINAMATH_CALUDE_unique_whole_number_between_l2422_242254

theorem unique_whole_number_between (N : ℤ) : 
  (5.5 < (N : ℚ) / 4 ∧ (N : ℚ) / 4 < 6) ↔ N = 23 := by
sorry

end NUMINAMATH_CALUDE_unique_whole_number_between_l2422_242254


namespace NUMINAMATH_CALUDE_tangent_line_cubic_l2422_242255

/-- The tangent line(s) to the curve y = x^3 passing through the point (1, 1) -/
theorem tangent_line_cubic (x y : ℝ) : 
  (y = x^3 ∧ (x - 1)^2 + (y - 1)^2 = 0) ∨ 
  (y = x^3 ∧ (3*x - y - 2 = 0 ∨ 3*x - 4*y + 1 = 0) ∧ x ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_cubic_l2422_242255


namespace NUMINAMATH_CALUDE_bonus_distribution_solution_l2422_242240

/-- Represents the bonus distribution problem -/
def BonusDistribution (total : ℚ) (ac_sum : ℚ) (common_ratio : ℚ) (d_bonus : ℚ) : Prop :=
  let a := d_bonus / (common_ratio^3)
  let b := d_bonus / (common_ratio^2)
  let c := d_bonus / common_ratio
  (a + b + c + d_bonus = total) ∧ 
  (a + c = ac_sum) ∧
  (0 < common_ratio) ∧ 
  (common_ratio < 1)

/-- The theorem stating the correct solution to the bonus distribution problem -/
theorem bonus_distribution_solution :
  BonusDistribution 68780 36200 (9/10) 14580 := by
  sorry

#check bonus_distribution_solution

end NUMINAMATH_CALUDE_bonus_distribution_solution_l2422_242240


namespace NUMINAMATH_CALUDE_other_endpoint_of_line_segment_l2422_242288

/-- Given a line segment with midpoint (2, 3) and one endpoint (-1, 7),
    prove that the other endpoint is (5, -1). -/
theorem other_endpoint_of_line_segment (A B M : ℝ × ℝ) : 
  M = (2, 3) → A = (-1, 7) → M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) → B = (5, -1) := by
  sorry

end NUMINAMATH_CALUDE_other_endpoint_of_line_segment_l2422_242288


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l2422_242265

theorem arithmetic_mean_problem (x : ℚ) : 
  (x + 10 + 20 + 3*x + 18 + (3*x + 6)) / 5 = 30 → x = 96/7 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l2422_242265


namespace NUMINAMATH_CALUDE_prop_1_prop_2_prop_3_l2422_242202

-- Define the function f(x)
def f (x b c : ℝ) : ℝ := x * |x| + b * x + c

-- Theorem for proposition ①
theorem prop_1 (b : ℝ) : 
  ∀ x, f x b 0 = -f (-x) b 0 := by sorry

-- Theorem for proposition ②
theorem prop_2 (c : ℝ) (h : c > 0) :
  ∃! x, f x 0 c = 0 := by sorry

-- Theorem for proposition ③
theorem prop_3 (b c : ℝ) :
  ∀ x, f x b c = f (-x) b c + 2 * c := by sorry

end NUMINAMATH_CALUDE_prop_1_prop_2_prop_3_l2422_242202


namespace NUMINAMATH_CALUDE_constant_m_value_l2422_242253

theorem constant_m_value (x y z m : ℝ) :
  (5^2 / (x + y) = m / (x + 2*z)) ∧ (m / (x + 2*z) = 7^2 / (y - 2*z)) →
  m = 74 := by
  sorry

end NUMINAMATH_CALUDE_constant_m_value_l2422_242253


namespace NUMINAMATH_CALUDE_weight_of_b_l2422_242205

/-- Given three weights a, b, and c, prove that b = 31 when:
    1. The average of a, b, and c is 45
    2. The average of a and b is 40
    3. The average of b and c is 43 -/
theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 45)
  (h2 : (a + b) / 2 = 40)
  (h3 : (b + c) / 2 = 43) :
  b = 31 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_b_l2422_242205


namespace NUMINAMATH_CALUDE_least_possible_third_side_l2422_242237

theorem least_possible_third_side (a b : ℝ) (ha : a = 7) (hb : b = 24) :
  let c := Real.sqrt (b^2 - a^2)
  c = Real.sqrt 527 ∧ c ≤ a ∧ c ≤ b := by sorry

end NUMINAMATH_CALUDE_least_possible_third_side_l2422_242237


namespace NUMINAMATH_CALUDE_x_interval_l2422_242246

theorem x_interval (x : ℝ) (h1 : 1/x < 3) (h2 : 1/x > -4) (h3 : 2*x - 1 > 0) : x > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_x_interval_l2422_242246


namespace NUMINAMATH_CALUDE_unfair_die_expected_value_l2422_242247

/-- Represents an unfair eight-sided die -/
structure UnfairDie where
  prob_8 : ℚ
  prob_others : ℚ
  sum_to_one : prob_8 + 7 * prob_others = 1
  prob_8_is_3_8 : prob_8 = 3/8

/-- Expected value of rolling the unfair die -/
def expected_value (d : UnfairDie) : ℚ :=
  d.prob_others * (1 + 2 + 3 + 4 + 5 + 6 + 7) + d.prob_8 * 8

/-- Theorem stating the expected value of the unfair die is 77/14 -/
theorem unfair_die_expected_value :
  ∀ (d : UnfairDie), expected_value d = 77/14 := by
  sorry

end NUMINAMATH_CALUDE_unfair_die_expected_value_l2422_242247


namespace NUMINAMATH_CALUDE_complex_number_magnitude_l2422_242287

theorem complex_number_magnitude (z : ℂ) :
  (1 - z) / (1 + z) = Complex.I ^ 2018 + Complex.I ^ 2019 →
  Complex.abs (2 + z) = 5 * Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_complex_number_magnitude_l2422_242287


namespace NUMINAMATH_CALUDE_canada_population_density_l2422_242223

def population : ℕ := 38005238
def area_sq_miles : ℕ := 3855103
def sq_feet_per_sq_mile : ℕ := 5280 * 5280

def total_sq_feet : ℕ := area_sq_miles * sq_feet_per_sq_mile
def avg_sq_feet_per_person : ℚ := total_sq_feet / population

theorem canada_population_density :
  (2700000 : ℚ) < avg_sq_feet_per_person ∧ avg_sq_feet_per_person < (2900000 : ℚ) :=
sorry

end NUMINAMATH_CALUDE_canada_population_density_l2422_242223


namespace NUMINAMATH_CALUDE_average_distance_is_600_l2422_242280

/-- The length of one lap around the block in meters -/
def block_length : ℕ := 200

/-- The number of times Johnny runs around the block -/
def johnny_laps : ℕ := 4

/-- The number of times Mickey runs around the block -/
def mickey_laps : ℕ := johnny_laps / 2

/-- The total distance run by Johnny in meters -/
def johnny_distance : ℕ := johnny_laps * block_length

/-- The total distance run by Mickey in meters -/
def mickey_distance : ℕ := mickey_laps * block_length

/-- The average distance run by Johnny and Mickey in meters -/
def average_distance : ℕ := (johnny_distance + mickey_distance) / 2

theorem average_distance_is_600 : average_distance = 600 := by
  sorry

end NUMINAMATH_CALUDE_average_distance_is_600_l2422_242280


namespace NUMINAMATH_CALUDE_unique_solution_l2422_242235

theorem unique_solution : ∃! x : ℝ, 
  (3 * x^2) / (x - 2) - (3 * x + 9) / 4 + (6 - 9 * x) / (x - 2) + 2 = 0 ∧ 
  x^3 ≠ 3 * x + 1 ∧
  x = 13 / 9 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2422_242235


namespace NUMINAMATH_CALUDE_solve_for_B_l2422_242201

theorem solve_for_B : ∃ B : ℝ, (4 * B + 4 - 3 = 29) ∧ (B = 7) := by
  sorry

end NUMINAMATH_CALUDE_solve_for_B_l2422_242201


namespace NUMINAMATH_CALUDE_range_of_omega_l2422_242269

/-- Given vectors a and b, and a function f, prove the range of ω -/
theorem range_of_omega (ω : ℝ) (x : ℝ) : 
  ω > 0 →
  let a := (Real.sin (ω/2 * x), Real.sin (ω * x))
  let b := (Real.sin (ω/2 * x), (1/2 : ℝ))
  let f := λ x => (a.1 * b.1 + a.2 * b.2) - 1/2
  (∀ x ∈ Set.Ioo π (2*π), f x ≠ 0) →
  ω ∈ Set.Ioc 0 (1/8) ∪ Set.Icc (1/4) (5/8) :=
sorry

end NUMINAMATH_CALUDE_range_of_omega_l2422_242269


namespace NUMINAMATH_CALUDE_quadratic_real_root_l2422_242232

/-- A quadratic equation x^2 + bx + 25 = 0 has at least one real root if and only if b ∈ (-∞, -10] ∪ [10, ∞) -/
theorem quadratic_real_root (b : ℝ) : 
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_root_l2422_242232


namespace NUMINAMATH_CALUDE_rectangle_side_difference_l2422_242250

theorem rectangle_side_difference (A d : ℝ) (h_A : A > 0) (h_d : d > 0) :
  ∃ x y : ℝ, x > y ∧ x * y = A ∧ x^2 + y^2 = d^2 ∧ x - y = Real.sqrt (d^2 - 4 * A) :=
sorry

end NUMINAMATH_CALUDE_rectangle_side_difference_l2422_242250


namespace NUMINAMATH_CALUDE_constant_term_expansion_l2422_242242

theorem constant_term_expansion (x : ℝ) : ∃ c : ℝ, c = 24 ∧ 
  (∃ f : ℝ → ℝ, (λ x => (2*x + 1/x)^4) = f + λ _ => c) := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l2422_242242


namespace NUMINAMATH_CALUDE_sqrt_problem_quadratic_equation_problem_l2422_242270

-- Problem 1
theorem sqrt_problem :
  Real.sqrt 12 * Real.sqrt 75 - Real.sqrt 8 + Real.sqrt 2 = 30 - Real.sqrt 2 := by
  sorry

-- Problem 2
theorem quadratic_equation_problem (x : ℝ) :
  (1 / 9 : ℝ) * (3 * x - 2)^2 - 4 = 0 ↔ x = 8/3 ∨ x = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_problem_quadratic_equation_problem_l2422_242270


namespace NUMINAMATH_CALUDE_line_equation_through_two_points_l2422_242251

/-- The line passing through points A(-2, 4) and B(-1, 3) has the equation y = -x + 2 -/
theorem line_equation_through_two_points :
  let A : ℝ × ℝ := (-2, 4)
  let B : ℝ × ℝ := (-1, 3)
  let line_eq : ℝ → ℝ := λ x => -x + 2
  (line_eq A.1 = A.2) ∧ (line_eq B.1 = B.2) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_through_two_points_l2422_242251


namespace NUMINAMATH_CALUDE_complex_product_theorem_l2422_242268

theorem complex_product_theorem (z₁ z₂ : ℂ) : 
  (z₁.re = 1 ∧ z₁.im = 1) → (z₂.re = 1 ∧ z₂.im = -1) → z₁ * z₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_theorem_l2422_242268


namespace NUMINAMATH_CALUDE_knights_round_table_l2422_242241

theorem knights_round_table (n : ℕ) 
  (h1 : ∃ (f e : ℕ), f = e ∧ f + e = n) : 
  4 ∣ n := by
sorry

end NUMINAMATH_CALUDE_knights_round_table_l2422_242241


namespace NUMINAMATH_CALUDE_lower_limit_proof_l2422_242200

theorem lower_limit_proof (x : ℤ) (y : ℝ) 
  (h1 : 0 < x ∧ x < 7)
  (h2 : 0 < x ∧ x < 15)
  (h3 : y < x ∧ x < 5)
  (h4 : 0 < x ∧ x < 3)
  (h5 : x + 2 < 4)
  (h6 : x = 1) :
  y < 1 := by
sorry

end NUMINAMATH_CALUDE_lower_limit_proof_l2422_242200


namespace NUMINAMATH_CALUDE_range_of_a_l2422_242291

-- Define the sets N and M
def N (a : ℝ) : Set ℝ := {x | (x - a) * (x + a - 2) < 0}
def M : Set ℝ := {x | -1/2 ≤ x ∧ x < 2}

-- State the theorem
theorem range_of_a :
  (∀ x, x ∈ M → x ∈ N a) → (a ≤ -1/2 ∨ a ≥ 5/2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2422_242291


namespace NUMINAMATH_CALUDE_certain_number_subtraction_l2422_242295

theorem certain_number_subtraction (x : ℤ) : x + 468 = 954 → x - 3 = 483 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_subtraction_l2422_242295


namespace NUMINAMATH_CALUDE_carey_gumballs_difference_l2422_242281

/-- The number of gumballs Carolyn bought -/
def carolyn_gumballs : ℕ := 17

/-- The number of gumballs Lew bought -/
def lew_gumballs : ℕ := 12

/-- The average number of gumballs bought by the three people -/
def average_gumballs : ℚ → ℚ := λ c => (carolyn_gumballs + lew_gumballs + c) / 3

/-- The theorem stating the difference between max and min gumballs Carey could have bought -/
theorem carey_gumballs_difference :
  ∃ (min_c max_c : ℕ),
    (∀ c : ℚ, 19 ≤ average_gumballs c → average_gumballs c ≤ 25 → ↑min_c ≤ c ∧ c ≤ ↑max_c) ∧
    max_c - min_c = 18 := by
  sorry

end NUMINAMATH_CALUDE_carey_gumballs_difference_l2422_242281


namespace NUMINAMATH_CALUDE_harmonic_sum_terms_added_l2422_242258

theorem harmonic_sum_terms_added (k : ℕ) (h : k > 1) :
  (Finset.range (2^(k+1) - 1)).card - (Finset.range (2^k - 1)).card = 2^k := by
  sorry

end NUMINAMATH_CALUDE_harmonic_sum_terms_added_l2422_242258


namespace NUMINAMATH_CALUDE_linear_increase_l2422_242272

/-- A linear function f(x) = 5x - 3 -/
def f (x : ℝ) : ℝ := 5 * x - 3

/-- Theorem: For a linear function f(x) = 5x - 3, 
    if x₁ < x₂, then f(x₁) < f(x₂) -/
theorem linear_increase (x₁ x₂ : ℝ) (h : x₁ < x₂) : f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_linear_increase_l2422_242272


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2422_242263

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = 
  (x^2 + 6*x + 4) * (x^2 + 6*x + 11) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2422_242263


namespace NUMINAMATH_CALUDE_perpendicular_parallel_implies_parallel_l2422_242233

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

end NUMINAMATH_CALUDE_perpendicular_parallel_implies_parallel_l2422_242233


namespace NUMINAMATH_CALUDE_inequality_of_reciprocals_l2422_242228

theorem inequality_of_reciprocals (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1/(2*a) + 1/(2*b) + 1/(2*c) ≥ 1/(a+b) + 1/(b+c) + 1/(c+a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_reciprocals_l2422_242228


namespace NUMINAMATH_CALUDE_office_age_problem_l2422_242213

theorem office_age_problem (total_people : Nat) (group1_people : Nat) (group2_people : Nat)
  (total_avg_age : ℝ) (group1_avg_age : ℝ) (group2_avg_age : ℝ)
  (h1 : total_people = 16)
  (h2 : group1_people = 5)
  (h3 : group2_people = 9)
  (h4 : total_avg_age = 15)
  (h5 : group1_avg_age = 14)
  (h6 : group2_avg_age = 16) :
  (total_people : ℝ) * total_avg_age - 
  (group1_people : ℝ) * group1_avg_age - 
  (group2_people : ℝ) * group2_avg_age = 52 := by
sorry

end NUMINAMATH_CALUDE_office_age_problem_l2422_242213


namespace NUMINAMATH_CALUDE_arbitrary_across_classes_most_representative_l2422_242249

/-- Represents a sampling method for a student survey --/
inductive SamplingMethod
  | GradeSpecific
  | GenderSpecific
  | ActivitySpecific
  | ArbitraryAcrossClasses

/-- Determines if a sampling method is representative of the entire student population --/
def is_representative (method : SamplingMethod) : Prop :=
  match method with
  | SamplingMethod.ArbitraryAcrossClasses => true
  | _ => false

/-- Theorem stating that the arbitrary across classes method is the most representative --/
theorem arbitrary_across_classes_most_representative :
  ∀ (method : SamplingMethod),
    is_representative method →
    method = SamplingMethod.ArbitraryAcrossClasses :=
by
  sorry

#check arbitrary_across_classes_most_representative

end NUMINAMATH_CALUDE_arbitrary_across_classes_most_representative_l2422_242249


namespace NUMINAMATH_CALUDE_m_range_characterization_l2422_242221

theorem m_range_characterization (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - (3 - m) * x + 1 > 0 ∨ m * x > 0) ↔ 1/9 < m ∧ m < 1 := by
sorry

end NUMINAMATH_CALUDE_m_range_characterization_l2422_242221


namespace NUMINAMATH_CALUDE_not_necessarily_true_squared_l2422_242245

theorem not_necessarily_true_squared (x y : ℝ) (h : x > y) : 
  ¬ (∀ x y : ℝ, x > y → x^2 > y^2) :=
sorry

end NUMINAMATH_CALUDE_not_necessarily_true_squared_l2422_242245


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l2422_242204

/-- Given a man's speed with the current and the speed of the current, 
    calculate the man's speed against the current. -/
def speed_against_current (speed_with_current speed_of_current : ℝ) : ℝ :=
  speed_with_current - 2 * speed_of_current

/-- Theorem: Given the specified conditions, the man's speed against the current is 10 km/hr. -/
theorem mans_speed_against_current :
  speed_against_current 15 2.5 = 10 := by
  sorry

#eval speed_against_current 15 2.5

end NUMINAMATH_CALUDE_mans_speed_against_current_l2422_242204


namespace NUMINAMATH_CALUDE_max_min_f_on_interval_l2422_242286

def f (x : ℝ) : ℝ := 3 * x^4 + 4 * x^3 + 34

theorem max_min_f_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-2 : ℝ) 1, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-2 : ℝ) 1, f x = max) ∧
    (∀ x ∈ Set.Icc (-2 : ℝ) 1, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-2 : ℝ) 1, f x = min) ∧
    max = 50 ∧ min = 33 :=
by sorry

end NUMINAMATH_CALUDE_max_min_f_on_interval_l2422_242286


namespace NUMINAMATH_CALUDE_largest_perfect_square_factor_of_1800_l2422_242293

theorem largest_perfect_square_factor_of_1800 : 
  ∃ (n : ℕ), n^2 = 900 ∧ n^2 ∣ 1800 ∧ ∀ (m : ℕ), m^2 ∣ 1800 → m^2 ≤ 900 := by
  sorry

end NUMINAMATH_CALUDE_largest_perfect_square_factor_of_1800_l2422_242293


namespace NUMINAMATH_CALUDE_angle_equality_l2422_242285

theorem angle_equality (angle1 angle2 angle3 : ℝ) : 
  (angle1 + angle2 = 90) →  -- angle1 and angle2 are complementary
  (angle2 + angle3 = 90) →  -- angle2 and angle3 are complementary
  (angle1 = 40) →           -- angle1 is 40 degrees
  (angle3 = 40) :=          -- conclusion: angle3 is 40 degrees
by
  sorry

#check angle_equality

end NUMINAMATH_CALUDE_angle_equality_l2422_242285


namespace NUMINAMATH_CALUDE_roots_expression_equals_one_l2422_242297

theorem roots_expression_equals_one (α β γ δ : ℝ) : 
  (α^2 - 2*α + 1 = 0) → 
  (β^2 - 2*β + 1 = 0) → 
  (γ^2 - 3*γ + 1 = 0) → 
  (δ^2 - 3*δ + 1 = 0) → 
  (α - γ)^2 * (β - δ)^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_roots_expression_equals_one_l2422_242297


namespace NUMINAMATH_CALUDE_integral_equals_ln5_over_8_l2422_242294

/-- The definite integral of the given function from 0 to 1 is equal to (1/8) * ln(5) -/
theorem integral_equals_ln5_over_8 :
  ∫ x in (0 : ℝ)..1, (4 * Real.sqrt (1 - x) - Real.sqrt (x + 1)) /
    ((Real.sqrt (x + 1) + 4 * Real.sqrt (1 - x)) * (x + 1)^2) = (1/8) * Real.log 5 := by
  sorry

end NUMINAMATH_CALUDE_integral_equals_ln5_over_8_l2422_242294


namespace NUMINAMATH_CALUDE_rectangle_area_proof_l2422_242252

theorem rectangle_area_proof : 
  let card1 : ℝ := 15
  let card2 : ℝ := card1 * 0.9
  let area : ℝ := card1 * card2
  area = 202.5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_proof_l2422_242252


namespace NUMINAMATH_CALUDE_exp_2_unique_l2422_242264

def is_exp_2 (f : ℕ+ → ℝ) : Prop :=
  f 1 = 2 ∧
  (∀ n : ℕ+, f n > 0) ∧
  (∀ n₁ n₂ : ℕ+, f (n₁ + n₂) = f n₁ * f n₂)

theorem exp_2_unique (f : ℕ+ → ℝ) (hf : is_exp_2 f) :
  ∀ n : ℕ+, f n = 2^(n : ℝ) :=
sorry

end NUMINAMATH_CALUDE_exp_2_unique_l2422_242264


namespace NUMINAMATH_CALUDE_recurring_decimal_fraction_sum_l2422_242277

theorem recurring_decimal_fraction_sum (a b : ℕ+) : 
  (a : ℚ) / (b : ℚ) = 56 / 99 → 
  Nat.gcd a b = 1 → 
  (a : ℕ) + b = 155 := by
  sorry

end NUMINAMATH_CALUDE_recurring_decimal_fraction_sum_l2422_242277
