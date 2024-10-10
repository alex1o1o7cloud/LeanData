import Mathlib

namespace divides_power_sum_l894_89438

theorem divides_power_sum (a b c : ℤ) (h : (a + b + c) ∣ (a^2 + b^2 + c^2)) :
  ∀ k : ℕ, (a + b + c) ∣ (a^(2^k) + b^(2^k) + c^(2^k)) :=
sorry

end divides_power_sum_l894_89438


namespace arithmetic_sequence_eighth_term_l894_89487

/-- Given an arithmetic sequence where the 4th term is 23 and the 6th term is 47,
    prove that the 8th term is 71. -/
theorem arithmetic_sequence_eighth_term
  (a : ℕ → ℤ)  -- The arithmetic sequence
  (h1 : a 4 = 23)  -- The 4th term is 23
  (h2 : a 6 = 47)  -- The 6th term is 47
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- The sequence is arithmetic
  : a 8 = 71 := by
  sorry

end arithmetic_sequence_eighth_term_l894_89487


namespace missing_figure_proof_l894_89424

theorem missing_figure_proof : ∃ x : ℝ, (0.25 / 100) * x = 0.04 ∧ x = 16 := by sorry

end missing_figure_proof_l894_89424


namespace binomial_probability_problem_l894_89445

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution (n : ℕ) where
  p : ℝ
  h1 : 0 < p
  h2 : p < 1

/-- The probability mass function for a binomial distribution -/
def binomialPMF (n : ℕ) (X : BinomialDistribution n) (k : ℕ) : ℝ :=
  (n.choose k) * X.p^k * (1 - X.p)^(n - k)

theorem binomial_probability_problem (X : BinomialDistribution 4) 
  (h3 : X.p < 1/2) 
  (h4 : binomialPMF 4 X 2 = 8/27) : 
  binomialPMF 4 X 1 = 32/81 := by
  sorry

end binomial_probability_problem_l894_89445


namespace time_after_3339_minutes_l894_89439

/-- Represents a time of day -/
structure TimeOfDay where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Represents a date -/
structure Date where
  year : Nat
  month : Nat
  day : Nat
  deriving Repr

/-- Represents a datetime -/
structure DateTime where
  date : Date
  time : TimeOfDay
  deriving Repr

def minutesToDateTime (startDateTime : DateTime) (elapsedMinutes : Nat) : DateTime :=
  sorry

theorem time_after_3339_minutes :
  let startDateTime := DateTime.mk (Date.mk 2020 12 31) (TimeOfDay.mk 0 0)
  let endDateTime := minutesToDateTime startDateTime 3339
  endDateTime = DateTime.mk (Date.mk 2021 1 2) (TimeOfDay.mk 7 39) := by
  sorry

end time_after_3339_minutes_l894_89439


namespace equation_solution_l894_89492

theorem equation_solution (x : ℝ) : x ≠ 2 → (-x^2 = (4*x + 2) / (x - 2)) ↔ x = -2 := by
  sorry

end equation_solution_l894_89492


namespace arrangement_pattern_sixtieth_number_is_eighteen_l894_89490

/-- Represents the value in a specific position of the arrangement -/
def arrangementValue (position : ℕ) : ℕ :=
  let rowNum := (position - 1) / 3 + 1
  3 * rowNum

/-- The arrangement follows the specified pattern -/
theorem arrangement_pattern (n : ℕ) :
  ∀ k, k ≤ 3 * n → arrangementValue (3 * (n - 1) + k) = 3 * n :=
  sorry

/-- The 60th number in the arrangement is 18 -/
theorem sixtieth_number_is_eighteen :
  arrangementValue 60 = 18 :=
  sorry

end arrangement_pattern_sixtieth_number_is_eighteen_l894_89490


namespace real_root_of_cubic_l894_89460

/-- Given a cubic polynomial with real coefficients c and d, 
    if -3 + 2i is a root, then 53/5 is the real root. -/
theorem real_root_of_cubic (c d : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (fun x : ℂ => c * x ^ 3 - x ^ 2 + d * x + 30) (-3 + 2 * Complex.I) = 0 →
  (fun x : ℝ => c * x ^ 3 - x ^ 2 + d * x + 30) (53 / 5) = 0 :=
by sorry

end real_root_of_cubic_l894_89460


namespace trajectory_of_center_l894_89447

-- Define the circles F1 and F2
def circle_F1 (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def circle_F2 (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 25

-- Define the property of being externally tangent
def externally_tangent (C_x C_y R : ℝ) : Prop :=
  (C_x + 1)^2 + C_y^2 = (R + 1)^2

-- Define the property of being internally tangent
def internally_tangent (C_x C_y R : ℝ) : Prop :=
  (C_x - 1)^2 + C_y^2 = (5 - R)^2

-- Theorem stating the trajectory of the center C
theorem trajectory_of_center :
  ∀ C_x C_y R : ℝ,
  externally_tangent C_x C_y R →
  internally_tangent C_x C_y R →
  C_x^2 / 9 + C_y^2 / 8 = 1 :=
sorry

end trajectory_of_center_l894_89447


namespace probability_no_shaded_square_l894_89436

/-- Represents a rectangle in the grid --/
structure Rectangle where
  left : Nat
  right : Nat
  top : Nat
  bottom : Nat

/-- The grid configuration --/
def grid_width : Nat := 201
def grid_height : Nat := 3
def shaded_column : Nat := grid_width / 2 + 1

/-- Checks if a rectangle contains a shaded square --/
def contains_shaded (r : Rectangle) : Bool :=
  r.left ≤ shaded_column && shaded_column ≤ r.right

/-- Counts the total number of possible rectangles --/
def total_rectangles : Nat :=
  (grid_width.choose 2) * (grid_height.choose 2)

/-- Counts the number of rectangles that contain a shaded square --/
def shaded_rectangles : Nat :=
  grid_height * (shaded_column - 1) * (grid_width - shaded_column)

/-- The main theorem --/
theorem probability_no_shaded_square :
  (total_rectangles - shaded_rectangles) / total_rectangles = 100 / 201 := by
  sorry


end probability_no_shaded_square_l894_89436


namespace stock_value_order_l894_89412

def initial_investment : ℝ := 200

def omega_year1_change : ℝ := 1.15
def bravo_year1_change : ℝ := 0.70
def zeta_year1_change : ℝ := 1.00

def omega_year2_change : ℝ := 0.90
def bravo_year2_change : ℝ := 1.30
def zeta_year2_change : ℝ := 1.00

def omega_final : ℝ := initial_investment * omega_year1_change * omega_year2_change
def bravo_final : ℝ := initial_investment * bravo_year1_change * bravo_year2_change
def zeta_final : ℝ := initial_investment * zeta_year1_change * zeta_year2_change

theorem stock_value_order : bravo_final < zeta_final ∧ zeta_final < omega_final :=
by sorry

end stock_value_order_l894_89412


namespace diophantine_equation_solvable_l894_89469

theorem diophantine_equation_solvable (p : ℕ) (hp : Nat.Prime p) : 
  ∃ (x y z : ℤ), x^2 + y^2 + p * z = 2003 := by
  sorry

end diophantine_equation_solvable_l894_89469


namespace water_amount_equals_sugar_amount_l894_89446

/-- Represents the ratio of ingredients in a recipe -/
structure RecipeRatio where
  flour : ℚ
  water : ℚ
  sugar : ℚ

/-- The original recipe ratio -/
def original_ratio : RecipeRatio := ⟨10, 6, 3⟩

/-- The new recipe ratio -/
def new_ratio : RecipeRatio := 
  let flour_water_doubled := original_ratio.flour / original_ratio.water * 2
  let flour_sugar_halved := original_ratio.flour / original_ratio.sugar / 2
  ⟨
    flour_water_doubled * original_ratio.water,
    original_ratio.water,
    flour_sugar_halved * original_ratio.sugar
  ⟩

/-- Amount of sugar in the new recipe -/
def sugar_amount : ℚ := 4

theorem water_amount_equals_sugar_amount : 
  (new_ratio.water / new_ratio.sugar) * sugar_amount = sugar_amount := by
  sorry

end water_amount_equals_sugar_amount_l894_89446


namespace geometric_sequence_sum_relation_l894_89472

/-- A geometric sequence with specific partial sums -/
structure GeometricSequence where
  S : ℝ  -- Sum of first 2 terms
  T : ℝ  -- Sum of first 4 terms
  R : ℝ  -- Sum of first 6 terms

/-- Theorem stating the relation between partial sums of a geometric sequence -/
theorem geometric_sequence_sum_relation (seq : GeometricSequence) :
  seq.S^2 + seq.T^2 = seq.S * (seq.T + seq.R) := by
  sorry

end geometric_sequence_sum_relation_l894_89472


namespace largest_divisor_five_consecutive_integers_l894_89465

theorem largest_divisor_five_consecutive_integers :
  ∀ n : ℕ, ∃ m : ℕ,
    (120 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) ∧
    (∀ k : ℕ, k > 120 → ∃ p : ℕ, ¬(k ∣ (p * (p + 1) * (p + 2) * (p + 3) * (p + 4)))) :=
by sorry

end largest_divisor_five_consecutive_integers_l894_89465


namespace largest_reciprocal_l894_89427

theorem largest_reciprocal (a b c d e : ℚ) 
  (ha : a = 5/6) (hb : b = 1/2) (hc : c = 3) (hd : d = 8/3) (he : e = 240) :
  (1 / b > 1 / a) ∧ (1 / b > 1 / c) ∧ (1 / b > 1 / d) ∧ (1 / b > 1 / e) :=
by sorry

end largest_reciprocal_l894_89427


namespace apartment_complex_flashlights_joas_apartment_complex_flashlights_l894_89484

/-- Calculates the total number of emergency flashlights in an apartment complex -/
theorem apartment_complex_flashlights (total_buildings : ℕ) 
  (stories_per_building : ℕ) (families_per_floor_type1 : ℕ) 
  (families_per_floor_type2 : ℕ) (flashlights_per_family : ℕ) : ℕ :=
  let half_buildings := total_buildings / 2
  let families_type1 := half_buildings * stories_per_building * families_per_floor_type1
  let families_type2 := half_buildings * stories_per_building * families_per_floor_type2
  let total_families := families_type1 + families_type2
  total_families * flashlights_per_family

/-- The number of emergency flashlights in Joa's apartment complex -/
theorem joas_apartment_complex_flashlights : 
  apartment_complex_flashlights 8 15 4 5 2 = 1080 := by
  sorry

end apartment_complex_flashlights_joas_apartment_complex_flashlights_l894_89484


namespace prob_at_least_two_of_six_l894_89457

/-- The number of questions randomly guessed -/
def n : ℕ := 6

/-- The number of choices for each question -/
def k : ℕ := 5

/-- The probability of getting a single question correct -/
def p : ℚ := 1 / k

/-- The probability of getting a single question incorrect -/
def q : ℚ := 1 - p

/-- The probability of getting at least two questions correct out of n questions -/
def prob_at_least_two (n : ℕ) (p : ℚ) : ℚ :=
  1 - (q ^ n + n * p * q ^ (n - 1))

theorem prob_at_least_two_of_six :
  prob_at_least_two n p = 5385 / 15625 := by
  sorry

end prob_at_least_two_of_six_l894_89457


namespace finite_seq_nat_countable_l894_89448

-- Define the type for finite sequences of natural numbers
def FiniteSeqNat := List Nat

-- Statement of the theorem
theorem finite_seq_nat_countable : 
  ∃ f : FiniteSeqNat → Nat, Function.Bijective f :=
sorry

end finite_seq_nat_countable_l894_89448


namespace manuscript_review_theorem_l894_89417

/-- Represents the review process for a manuscript --/
structure ManuscriptReview where
  initial_pass_prob : ℝ
  third_expert_pass_prob : ℝ

/-- Calculates the probability of a manuscript being accepted --/
def acceptance_probability (review : ManuscriptReview) : ℝ :=
  review.initial_pass_prob ^ 2 + 
  2 * review.initial_pass_prob * (1 - review.initial_pass_prob) * review.third_expert_pass_prob

/-- Represents the distribution of accepted manuscripts --/
def manuscript_distribution (n : ℕ) (p : ℝ) : List (ℕ × ℝ) :=
  sorry

/-- Theorem stating the probability of acceptance and the distribution of accepted manuscripts --/
theorem manuscript_review_theorem (review : ManuscriptReview) 
    (h1 : review.initial_pass_prob = 0.5)
    (h2 : review.third_expert_pass_prob = 0.3) :
  acceptance_probability review = 0.4 ∧ 
  manuscript_distribution 4 (acceptance_probability review) = 
    [(0, 0.1296), (1, 0.3456), (2, 0.3456), (3, 0.1536), (4, 0.0256)] :=
  sorry

end manuscript_review_theorem_l894_89417


namespace nth_equation_l894_89486

theorem nth_equation (n : ℕ) : 
  1 / (n + 1 : ℚ) + 1 / (n * (n + 1) : ℚ) = 1 / n :=
by sorry

end nth_equation_l894_89486


namespace point_in_region_l894_89482

theorem point_in_region (m : ℝ) :
  m^2 - 3*m + 2 > 0 ↔ m < 1 ∨ m > 2 := by sorry

end point_in_region_l894_89482


namespace joshua_oranges_expenditure_l894_89425

/-- The amount Joshua spent on buying oranges -/
def joshua_spent (num_oranges : ℕ) (selling_price profit : ℚ) : ℚ :=
  (num_oranges : ℚ) * (selling_price - profit)

/-- Theorem stating the amount Joshua spent on oranges -/
theorem joshua_oranges_expenditure :
  joshua_spent 25 0.60 0.10 = 12.50 := by
  sorry

end joshua_oranges_expenditure_l894_89425


namespace sports_books_count_l894_89454

/-- Given the total number of books and the number of school books,
    prove that the number of sports books is 39. -/
theorem sports_books_count (total_books school_books : ℕ)
    (h1 : total_books = 58)
    (h2 : school_books = 19) :
    total_books - school_books = 39 := by
  sorry

end sports_books_count_l894_89454


namespace tangent_coincidence_implies_a_range_l894_89463

/-- Piecewise function f(x) defined as x^2 + x + a for x < 0, and -1/x for x > 0 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then x^2 + x + a else -1/x

/-- The derivative of f(x) -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then 2*x + 1 else 1/x^2

theorem tangent_coincidence_implies_a_range :
  ∀ a : ℝ,
  (∃ x₁ x₂ : ℝ, x₁ < 0 ∧ 0 < x₂ ∧ 
   f_derivative a x₁ = f_derivative a x₂ ∧
   f a x₁ - (f_derivative a x₁ * x₁) = f a x₂ - (f_derivative a x₂ * x₂)) →
  -2 < a ∧ a < 1/4 :=
by sorry

end tangent_coincidence_implies_a_range_l894_89463


namespace graph_shift_l894_89405

/-- Given a function f: ℝ → ℝ, prove that f(x - 2) + 1 is equivalent to
    shifting the graph of f(x) right by 2 units and up by 1 unit. -/
theorem graph_shift (f : ℝ → ℝ) (x : ℝ) :
  f (x - 2) + 1 = (fun y ↦ f (y - 2)) (x + 2) := by
  sorry

end graph_shift_l894_89405


namespace bakery_flour_usage_l894_89408

theorem bakery_flour_usage (wheat_flour : Real) (total_flour : Real) (white_flour : Real) :
  wheat_flour = 0.2 →
  total_flour = 0.3 →
  white_flour = total_flour - wheat_flour →
  white_flour = 0.1 := by
  sorry

end bakery_flour_usage_l894_89408


namespace quadrilaterals_on_circle_l894_89416

/-- The number of distinct convex quadrilaterals that can be formed by selecting 4 vertices
    from 12 distinct points on the circumference of a circle. -/
def num_quadrilaterals : ℕ := 495

/-- The number of ways to choose 4 items from a set of 12 items. -/
def choose_4_from_12 : ℕ := Nat.choose 12 4

theorem quadrilaterals_on_circle :
  num_quadrilaterals = choose_4_from_12 :=
by sorry

end quadrilaterals_on_circle_l894_89416


namespace algebraic_identities_l894_89435

theorem algebraic_identities (a b : ℝ) : 
  ((-a)^2 * (a^2)^2 / a^3 = a^3) ∧ 
  ((a + b) * (a - b) - (a - b)^2 = 2*a*b - 2*b^2) := by
  sorry

end algebraic_identities_l894_89435


namespace fraction_simplification_l894_89429

theorem fraction_simplification : 
  (2015^2 : ℚ) / (2014^2 + 2016^2 - 2) = 1/2 := by sorry

end fraction_simplification_l894_89429


namespace min_point_of_translated_abs_function_l894_89443

-- Define the function
def f (x : ℝ) : ℝ := |x - 3| - 10

-- State the theorem
theorem min_point_of_translated_abs_function :
  ∃ (x₀ : ℝ), (∀ (x : ℝ), f x₀ ≤ f x) ∧ (f x₀ = -10) ∧ (x₀ = -3) := by
  sorry

end min_point_of_translated_abs_function_l894_89443


namespace unique_solution_implies_a_equals_two_l894_89495

theorem unique_solution_implies_a_equals_two (a : ℝ) : 
  (∃! x : ℝ, 0 ≤ x^2 - a*x + a ∧ x^2 - a*x + a ≤ 1) → a = 2 :=
by sorry

end unique_solution_implies_a_equals_two_l894_89495


namespace bus_network_routes_count_l894_89464

/-- A bus network in a city. -/
structure BusNetwork where
  /-- The set of bus stops. -/
  stops : Type
  /-- The set of bus routes. -/
  routes : Type
  /-- Predicate indicating if a stop is on a route. -/
  on_route : stops → routes → Prop

/-- Properties of a valid bus network. -/
class ValidBusNetwork (bn : BusNetwork) where
  /-- From any stop to any other stop, you can get there without a transfer. -/
  no_transfer : ∀ (s₁ s₂ : bn.stops), ∃ (r : bn.routes), bn.on_route s₁ r ∧ bn.on_route s₂ r
  /-- For any pair of routes, there is exactly one stop where you can transfer from one route to the other. -/
  unique_transfer : ∀ (r₁ r₂ : bn.routes), ∃! (s : bn.stops), bn.on_route s r₁ ∧ bn.on_route s r₂
  /-- Each route has exactly three stops. -/
  three_stops : ∀ (r : bn.routes), ∃! (s₁ s₂ s₃ : bn.stops), 
    bn.on_route s₁ r ∧ bn.on_route s₂ r ∧ bn.on_route s₃ r ∧ 
    s₁ ≠ s₂ ∧ s₁ ≠ s₃ ∧ s₂ ≠ s₃

/-- The theorem stating the relationship between the number of routes and stops. -/
theorem bus_network_routes_count {bn : BusNetwork} [ValidBusNetwork bn] [Fintype bn.stops] [Fintype bn.routes] : 
  Fintype.card bn.routes = Fintype.card bn.stops * (Fintype.card bn.stops - 1) + 1 :=
sorry

end bus_network_routes_count_l894_89464


namespace inequality_not_always_correct_l894_89497

theorem inequality_not_always_correct
  (x y z w : ℝ)
  (hx : x > 0)
  (hy : y > 0)
  (hxy : x > y)
  (hz : z ≠ 0)
  (hw : w ≠ 0) :
  ∃ (x' y' z' w' : ℝ),
    x' > 0 ∧ y' > 0 ∧ x' > y' ∧ z' ≠ 0 ∧ w' ≠ 0 ∧
    x' * z' ≤ y' * w' * z' :=
sorry

end inequality_not_always_correct_l894_89497


namespace feuerbach_theorem_l894_89470

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The midpoint circle of a triangle -/
def midpointCircle (t : Triangle) : Circle := sorry

/-- The incircle of a triangle -/
def incircle (t : Triangle) : Circle := sorry

/-- The excircles of a triangle -/
def excircles (t : Triangle) : Fin 3 → Circle := sorry

/-- Two circles are tangent -/
def areTangent (c1 c2 : Circle) : Prop := sorry

/-- Feuerbach's theorem -/
theorem feuerbach_theorem (t : Triangle) : 
  (areTangent (midpointCircle t) (incircle t)) ∧ 
  (∀ i : Fin 3, areTangent (midpointCircle t) (excircles t i)) := by
  sorry

end feuerbach_theorem_l894_89470


namespace bullet_hole_displacement_l894_89479

/-- The displacement of the second hole relative to the first hole when a bullet is fired perpendicular to a moving train -/
theorem bullet_hole_displacement 
  (c : Real) -- speed of the train in km/h
  (c_prime : Real) -- speed of the bullet in m/s
  (a : Real) -- width of the train car in meters
  (h1 : c = 60) -- train speed is 60 km/h
  (h2 : c_prime = 40) -- bullet speed is 40 m/s
  (h3 : a = 4) -- train car width is 4 meters
  : (a * c * 1000 / 3600) / c_prime = 1.667 := by sorry

end bullet_hole_displacement_l894_89479


namespace third_number_in_set_l894_89421

theorem third_number_in_set (x : ℝ) : 
  (20 + 40 + x) / 3 = (10 + 70 + 16) / 3 + 8 → x = 60 := by
sorry

end third_number_in_set_l894_89421


namespace sheep_per_herd_l894_89431

theorem sheep_per_herd (total_sheep : ℕ) (num_herds : ℕ) (h1 : total_sheep = 60) (h2 : num_herds = 3) :
  total_sheep / num_herds = 20 := by
  sorry

end sheep_per_herd_l894_89431


namespace music_store_sales_calculation_l894_89450

/-- Represents the sales data for a mall with two stores -/
structure MallSales where
  num_cars : ℕ
  customers_per_car : ℕ
  sports_store_sales : ℕ

/-- Calculates the number of sales made by the music store -/
def music_store_sales (mall : MallSales) : ℕ :=
  mall.num_cars * mall.customers_per_car - mall.sports_store_sales

/-- Theorem: The music store sales is equal to the total customers minus sports store sales -/
theorem music_store_sales_calculation (mall : MallSales) 
  (h1 : mall.num_cars = 10)
  (h2 : mall.customers_per_car = 5)
  (h3 : mall.sports_store_sales = 20) :
  music_store_sales mall = 30 := by
  sorry

end music_store_sales_calculation_l894_89450


namespace min_value_theorem_l894_89461

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) :
  3 * x + y ≥ 1 + 8 * Real.sqrt 3 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 
    1 / (x₀ + 3) + 1 / (y₀ + 3) = 1 / 4 ∧
    3 * x₀ + y₀ = 1 + 8 * Real.sqrt 3 :=
sorry

end min_value_theorem_l894_89461


namespace puzzle_completion_time_l894_89426

/-- Calculates the time to complete puzzles given the number of puzzles, pieces per puzzle, and completion rate. -/
def time_to_complete_puzzles (num_puzzles : ℕ) (pieces_per_puzzle : ℕ) (pieces_per_interval : ℕ) (interval_minutes : ℕ) : ℕ :=
  let total_pieces := num_puzzles * pieces_per_puzzle
  let pieces_per_minute := pieces_per_interval / interval_minutes
  total_pieces / pieces_per_minute

/-- Proves that completing 2 puzzles of 2000 pieces each at a rate of 100 pieces per 10 minutes takes 400 minutes. -/
theorem puzzle_completion_time :
  time_to_complete_puzzles 2 2000 100 10 = 400 := by
  sorry

end puzzle_completion_time_l894_89426


namespace shadow_point_theorem_l894_89477

-- Define shadow point
def isShadowPoint (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ y > x, f y > f x

-- State the theorem
theorem shadow_point_theorem (f : ℝ → ℝ) (a b : ℝ) 
  (hf : Continuous f) 
  (hab : a < b)
  (h_shadow : ∀ x ∈ Set.Ioo a b, isShadowPoint f x)
  (ha_not_shadow : ¬ isShadowPoint f a)
  (hb_not_shadow : ¬ isShadowPoint f b) :
  (∀ x ∈ Set.Ioo a b, f x ≤ f b) ∧ f a = f b :=
sorry

end shadow_point_theorem_l894_89477


namespace fraction_decimal_digits_l894_89468

/-- The number of digits to the right of the decimal point when a positive rational number is expressed as a decimal. -/
def decimal_digits (q : ℚ) : ℕ :=
  sorry

/-- The fraction in question -/
def fraction : ℚ := (4^7) / (8^5 * 1250)

/-- Theorem stating that the number of digits to the right of the decimal point
    in the decimal representation of the given fraction is 3 -/
theorem fraction_decimal_digits :
  decimal_digits fraction = 3 := by sorry

end fraction_decimal_digits_l894_89468


namespace initial_pizzas_count_l894_89483

/-- The number of pizzas returned by customers. -/
def returned_pizzas : ℕ := 6

/-- The number of pizzas successfully served to customers. -/
def served_pizzas : ℕ := 3

/-- The total number of pizzas initially served by the restaurant. -/
def total_pizzas : ℕ := returned_pizzas + served_pizzas

theorem initial_pizzas_count : total_pizzas = 9 := by
  sorry

end initial_pizzas_count_l894_89483


namespace complex_product_equality_complex_sum_equality_l894_89452

-- Define the complex number i
def i : ℂ := Complex.I

-- Part 1
theorem complex_product_equality : 
  (1 : ℂ) * (1 - i) * (-1/2 + (Real.sqrt 3)/2 * i) * (1 + i) = -1 + Real.sqrt 3 * i := by sorry

-- Part 2
theorem complex_sum_equality :
  (2 + 2*i) / (1 - i)^2 + (Real.sqrt 2 / (1 + i))^2010 = -1 := by sorry

end complex_product_equality_complex_sum_equality_l894_89452


namespace equation_solution_l894_89433

theorem equation_solution : 
  ∃! x : ℚ, (3 * x - 17) / 4 = (x + 12) / 5 ∧ x = 133 / 11 := by sorry

end equation_solution_l894_89433


namespace smallest_n_boxes_l894_89415

theorem smallest_n_boxes : ∃ n : ℕ, n > 0 ∧ 
  (∀ m : ℕ, m > 0 → m < n → ¬(17 * m - 3) % 11 = 0) ∧ 
  (17 * n - 3) % 11 = 0 ∧ 
  n = 6 := by
  sorry

end smallest_n_boxes_l894_89415


namespace triangle_base_length_l894_89403

theorem triangle_base_length (area height : ℝ) (h1 : area = 16) (h2 : height = 4) :
  (2 * area) / height = 8 := by
  sorry

end triangle_base_length_l894_89403


namespace six_circle_arrangement_possible_l894_89441

/-- A configuration of 6 circles in a plane -/
structure CircleConfiguration where
  positions : Fin 6 → ℝ × ℝ

/-- Predicate to check if a configuration allows a 7th circle to touch all 6 -/
def ValidConfiguration (config : CircleConfiguration) : Prop :=
  ∃ (center : ℝ × ℝ), ∀ i : Fin 6, 
    let (x, y) := config.positions i
    let (cx, cy) := center
    (x - cx)^2 + (y - cy)^2 = 4  -- Assuming unit radius for simplicity

/-- Predicate to check if a configuration can be achieved without measurements or lifting -/
def AchievableWithoutMeasurement (config : CircleConfiguration) : Prop :=
  sorry  -- This would require a formal definition of "without measurement"

theorem six_circle_arrangement_possible :
  ∃ (config : CircleConfiguration), 
    ValidConfiguration config ∧ AchievableWithoutMeasurement config :=
sorry

end six_circle_arrangement_possible_l894_89441


namespace inequality_solution_set_l894_89414

theorem inequality_solution_set : 
  {x : ℝ | x^2 - 2*x - 5 > 2*x} = {x : ℝ | x > 5 ∨ x < -1} := by sorry

end inequality_solution_set_l894_89414


namespace linear_equation_m_value_l894_89411

/-- If (m+1)x + 3y^m = 5 is a linear equation in x and y, then m = 1 -/
theorem linear_equation_m_value (m : ℝ) : 
  (∀ x y : ℝ, ∃ a b c : ℝ, (m + 1) * x + 3 * y^m = a * x + b * y + c) → m = 1 := by
  sorry

end linear_equation_m_value_l894_89411


namespace basketball_court_measurements_l894_89423

theorem basketball_court_measurements :
  ∃! (A B C D E F : ℝ),
    A - B = C ∧
    D = 2 * (A + B) ∧
    E = A * B ∧
    F = 3 ∧
    A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0 ∧ E > 0 ∧ F > 0 ∧
    ({A, B, C, D, E, F} : Set ℝ) = {86, 13, 420, 15, 28, 3} ∧
    A = 28 ∧ B = 15 ∧ C = 13 ∧ D = 86 ∧ E = 420 ∧ F = 3 :=
by sorry

end basketball_court_measurements_l894_89423


namespace trig_simplification_l894_89480

theorem trig_simplification :
  (Real.cos (20 * π / 180) * Real.sqrt (1 - Real.cos (40 * π / 180))) / Real.cos (50 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end trig_simplification_l894_89480


namespace unique_solution_quadratic_l894_89493

theorem unique_solution_quadratic (m : ℝ) : 
  (∃! x : ℝ, 3 * x^2 + m * x + 36 = 0) ↔ m = 12 * Real.sqrt 3 :=
sorry

end unique_solution_quadratic_l894_89493


namespace root_equation_and_product_l894_89430

theorem root_equation_and_product (a : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + (2*a - 1)*x + a^2 = 0 ↔ x = x₁ ∨ x = x₂) →
  (x₁ + 2) * (x₂ + 2) = 11 →
  a = -1 := by
sorry

end root_equation_and_product_l894_89430


namespace sqrt_eight_div_sqrt_two_eq_two_l894_89413

theorem sqrt_eight_div_sqrt_two_eq_two :
  Real.sqrt 8 / Real.sqrt 2 = 2 := by
  sorry

end sqrt_eight_div_sqrt_two_eq_two_l894_89413


namespace prime_divisor_fourth_power_l894_89432

theorem prime_divisor_fourth_power (n : ℕ+) 
  (h : ∀ d : ℕ+, d ∣ n → ¬(n^2 ≤ d^4 ∧ d^4 ≤ n^3)) : 
  ∃ p : ℕ, p.Prime ∧ p ∣ n ∧ p^4 > n^3 := by
  sorry

end prime_divisor_fourth_power_l894_89432


namespace y_intercept_of_line_l894_89434

theorem y_intercept_of_line (x y : ℝ) : 
  (x + y - 1 = 0) → (0 + y - 1 = 0 → y = 1) := by
  sorry

end y_intercept_of_line_l894_89434


namespace consecutive_numbers_product_divisibility_l894_89478

theorem consecutive_numbers_product_divisibility (n : ℕ) (h : n > 1) :
  ∃ k : ℕ, ∀ p : ℕ,
    Prime p →
    (p ≤ 2*n + 1 ↔ ∃ i : ℕ, i < n ∧ p ∣ (k + i)) :=
by sorry

end consecutive_numbers_product_divisibility_l894_89478


namespace pizza_eaten_after_six_trips_l894_89458

def eat_pizza (n : ℕ) : ℚ :=
  1 - (2/3)^n

theorem pizza_eaten_after_six_trips :
  eat_pizza 6 = 665/729 := by sorry

end pizza_eaten_after_six_trips_l894_89458


namespace lights_remaining_on_l894_89491

def total_lights : ℕ := 2013

def lights_on_after_switches (n : ℕ) : ℕ :=
  n - (n / 2 + n / 3 + n / 5 - n / 6 - n / 10 - n / 15 + n / 30)

theorem lights_remaining_on :
  lights_on_after_switches total_lights = 1006 := by
  sorry

end lights_remaining_on_l894_89491


namespace binary_199_ones_minus_zeros_l894_89428

/-- The binary representation of a natural number -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinary' (n : ℕ) : List Bool :=
    if n = 0 then [] else (n % 2 = 1) :: toBinary' (n / 2)
  toBinary' n

/-- Count the number of true values in a list of booleans -/
def countTrue (l : List Bool) : ℕ :=
  l.filter id |>.length

/-- Count the number of false values in a list of booleans -/
def countFalse (l : List Bool) : ℕ :=
  l.filter not |>.length

theorem binary_199_ones_minus_zeros :
  let binary := toBinary 199
  let ones := countTrue binary
  let zeros := countFalse binary
  ones - zeros = 2 := by sorry

end binary_199_ones_minus_zeros_l894_89428


namespace multiplication_subtraction_equality_l894_89476

theorem multiplication_subtraction_equality : 75 * 1414 - 25 * 1414 = 70700 := by
  sorry

end multiplication_subtraction_equality_l894_89476


namespace alice_marbles_distinct_choices_l894_89488

/-- Represents the colors of marbles --/
inductive Color
  | Red
  | Green
  | Blue
  | Yellow

/-- Represents the marble collection --/
structure MarbleCollection where
  red : Nat
  green : Nat
  blue : Nat
  yellow : Nat

/-- Calculates the number of distinct ways to choose 2 marbles --/
def distinctChoices (collection : MarbleCollection) : Nat :=
  sorry

/-- Theorem stating that for Alice's marble collection, there are 9 distinct ways to choose 2 marbles --/
theorem alice_marbles_distinct_choices :
  let aliceCollection : MarbleCollection := ⟨3, 2, 1, 4⟩
  distinctChoices aliceCollection = 9 :=
sorry

end alice_marbles_distinct_choices_l894_89488


namespace work_completion_time_l894_89498

/-- The number of days it takes y to complete the work -/
def y_days : ℝ := 40

/-- The number of days it takes x and y together to complete the work -/
def combined_days : ℝ := 13.333333333333332

/-- The number of days it takes x to complete the work -/
def x_days : ℝ := 20

theorem work_completion_time :
  1 / x_days + 1 / y_days = 1 / combined_days :=
by sorry

end work_completion_time_l894_89498


namespace total_books_eq_sum_l894_89418

/-- The total number of different books in the 'crazy silly school' series -/
def total_books : ℕ := sorry

/-- The number of books already read from the series -/
def books_read : ℕ := 8

/-- The number of books left to read from the series -/
def books_left : ℕ := 6

/-- Theorem stating that the total number of books is equal to the sum of books read and books left to read -/
theorem total_books_eq_sum : total_books = books_read + books_left := by sorry

end total_books_eq_sum_l894_89418


namespace min_sum_squares_l894_89496

-- Define the set of possible values
def S : Finset Int := {-6, -4, -1, 0, 3, 5, 7, 10}

-- Define the theorem
theorem min_sum_squares (p q r s t u v w : Int) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
                q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
                r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
                s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
                t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
                u ≠ v ∧ u ≠ w ∧
                v ≠ w)
  (h_in_S : p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ s ∈ S ∧ t ∈ S ∧ u ∈ S ∧ v ∈ S ∧ w ∈ S) :
  (p + q + r + s)^2 + (t + u + v + w)^2 ≥ 98 :=
sorry

end min_sum_squares_l894_89496


namespace inequality_solution_set_l894_89475

-- Define the set of real numbers that satisfy the inequality
def solution_set : Set ℝ := {x | x ≠ 0 ∧ (1 / x < x)}

-- Theorem statement
theorem inequality_solution_set : 
  solution_set = {x | -1 < x ∧ x < 0} ∪ {x | x > 1} := by sorry

end inequality_solution_set_l894_89475


namespace area_not_above_x_axis_is_half_l894_89449

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four points -/
structure Parallelogram where
  p : Point
  q : Point
  r : Point
  s : Point

/-- Calculates the area of a parallelogram -/
def parallelogramArea (pg : Parallelogram) : ℝ :=
  sorry

/-- Calculates the area of the portion of a parallelogram below or on the x-axis -/
def areaNotAboveXAxis (pg : Parallelogram) : ℝ :=
  sorry

/-- The main theorem to be proved -/
theorem area_not_above_x_axis_is_half (pg : Parallelogram) :
  pg.p = ⟨4, 2⟩ ∧ pg.q = ⟨-2, -2⟩ ∧ pg.r = ⟨-6, -6⟩ ∧ pg.s = ⟨0, -2⟩ →
  areaNotAboveXAxis pg = (parallelogramArea pg) / 2 :=
sorry

end area_not_above_x_axis_is_half_l894_89449


namespace museum_travel_distance_l894_89456

/-- Calculates the total distance traveled to visit two museums on separate days -/
def totalDistanceTraveled (distance1 : ℕ) (distance2 : ℕ) : ℕ :=
  2 * distance1 + 2 * distance2

/-- Proves that visiting museums at 5 and 15 miles results in a total travel of 40 miles -/
theorem museum_travel_distance :
  totalDistanceTraveled 5 15 = 40 := by
  sorry

#eval totalDistanceTraveled 5 15

end museum_travel_distance_l894_89456


namespace park_outer_diameter_l894_89471

/-- Represents the dimensions of a circular park with concentric areas. -/
structure ParkDimensions where
  pond_diameter : ℝ
  seating_width : ℝ
  garden_width : ℝ
  path_width : ℝ

/-- Calculates the diameter of the outer boundary of a circular park. -/
def outer_diameter (park : ParkDimensions) : ℝ :=
  park.pond_diameter + 2 * (park.seating_width + park.garden_width + park.path_width)

/-- Theorem stating that for a park with given dimensions, the outer diameter is 64 feet. -/
theorem park_outer_diameter :
  let park := ParkDimensions.mk 20 4 10 8
  outer_diameter park = 64 := by
  sorry


end park_outer_diameter_l894_89471


namespace absolute_difference_l894_89455

/-- Given a set of five numbers {m, n, 9, 8, 10} with an average of 9 and a variance of 2, |m - n| = 4 -/
theorem absolute_difference (m n : ℝ) 
  (h_avg : (m + n + 9 + 8 + 10) / 5 = 9)
  (h_var : ((m - 9)^2 + (n - 9)^2 + (9 - 9)^2 + (8 - 9)^2 + (10 - 9)^2) / 5 = 2) :
  |m - n| = 4 := by
  sorry

end absolute_difference_l894_89455


namespace complex_subtraction_and_multiplication_l894_89489

theorem complex_subtraction_and_multiplication :
  (5 - 4*I : ℂ) - 2*(3 + 6*I) = -1 - 16*I :=
by sorry

end complex_subtraction_and_multiplication_l894_89489


namespace sports_club_members_l894_89462

/-- A sports club with members who play badminton, tennis, both, or neither -/
structure SportsClub where
  badminton : ℕ
  tennis : ℕ
  both : ℕ
  neither : ℕ

/-- The total number of members in the sports club -/
def total_members (club : SportsClub) : ℕ :=
  club.badminton + club.tennis - club.both + club.neither

/-- Theorem stating that the total number of members in the given sports club is 30 -/
theorem sports_club_members :
  ∃ (club : SportsClub),
    club.badminton = 17 ∧
    club.tennis = 21 ∧
    club.both = 10 ∧
    club.neither = 2 ∧
    total_members club = 30 := by
  sorry

end sports_club_members_l894_89462


namespace insufficient_info_for_production_l894_89459

structure MachineRates where
  A : ℝ
  B : ℝ
  C : ℝ

def total_production (rates : MachineRates) (hours : ℝ) : ℝ :=
  hours * (rates.A + rates.B + rates.C)

theorem insufficient_info_for_production (P : ℝ) :
  ∀ (rates : MachineRates),
    7 * rates.A + 11 * rates.B = 305 →
    8 * rates.A + 22 * rates.C = P →
    ∃ (rates' : MachineRates),
      7 * rates'.A + 11 * rates'.B = 305 ∧
      8 * rates'.A + 22 * rates'.C = P ∧
      total_production rates 8 ≠ total_production rates' 8 :=
by
  sorry

#check insufficient_info_for_production

end insufficient_info_for_production_l894_89459


namespace largest_negative_integer_l894_89409

theorem largest_negative_integer : 
  ∀ n : ℤ, n < 0 → n ≤ -1 :=
by sorry

end largest_negative_integer_l894_89409


namespace fifth_sample_number_l894_89422

/-- Systematic sampling function -/
def systematic_sample (total : ℕ) (sample_size : ℕ) (first_sample : ℕ) (group : ℕ) : ℕ :=
  first_sample + (total / sample_size) * (group - 1)

/-- Theorem: In a systematic sampling of 100 samples from 2000 items, 
    if the first sample is numbered 11, then the fifth sample will be numbered 91 -/
theorem fifth_sample_number :
  systematic_sample 2000 100 11 5 = 91 := by
  sorry

end fifth_sample_number_l894_89422


namespace total_fencing_cost_l894_89453

/-- Calculates the total fencing cost for an irregular shaped plot -/
theorem total_fencing_cost (square_area : ℝ) (rect_length rect_height : ℝ) (triangle_side : ℝ)
  (square_cost rect_cost triangle_cost : ℝ) (gate_cost : ℝ)
  (h_square_area : square_area = 289)
  (h_rect_length : rect_length = 45)
  (h_rect_height : rect_height = 15)
  (h_triangle_side : triangle_side = 20)
  (h_square_cost : square_cost = 55)
  (h_rect_cost : rect_cost = 65)
  (h_triangle_cost : triangle_cost = 70)
  (h_gate_cost : gate_cost = 750) :
  4 * Real.sqrt square_area * square_cost +
  (2 * rect_height + rect_length) * rect_cost +
  3 * triangle_side * triangle_cost +
  gate_cost = 13565 := by
  sorry


end total_fencing_cost_l894_89453


namespace only_25_is_five_times_greater_than_last_digit_l894_89420

def lastDigit (n : Nat) : Nat :=
  n % 10

theorem only_25_is_five_times_greater_than_last_digit :
  ∀ n : Nat, n > 0 → (n = 5 * lastDigit n + lastDigit n) → n = 25 := by
  sorry

end only_25_is_five_times_greater_than_last_digit_l894_89420


namespace regular_pentagons_are_similar_l894_89401

/-- A regular pentagon is a polygon with 5 sides of equal length and 5 angles of equal measure. -/
structure RegularPentagon where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- Two shapes are similar if they have the same shape but not necessarily the same size. -/
def are_similar (p1 p2 : RegularPentagon) : Prop :=
  ∃ k : ℝ, k > 0 ∧ p1.side_length = k * p2.side_length

/-- Theorem: Any two regular pentagons are similar. -/
theorem regular_pentagons_are_similar (p1 p2 : RegularPentagon) : are_similar p1 p2 := by
  sorry

end regular_pentagons_are_similar_l894_89401


namespace eugene_pencils_l894_89444

theorem eugene_pencils (initial_pencils : ℝ) (pencils_given : ℝ) :
  initial_pencils = 51.0 →
  pencils_given = 6.0 →
  initial_pencils - pencils_given = 45.0 :=
by
  sorry

end eugene_pencils_l894_89444


namespace trajectory_and_line_equations_l894_89404

-- Define the points
def A : ℝ × ℝ := (0, 3)
def O : ℝ × ℝ := (0, 0)
def N : ℝ × ℝ := (-1, 3)

-- Define the moving point M
def M : ℝ × ℝ → Prop := fun (x, y) ↦ 
  (x - A.1)^2 + (y - A.2)^2 = 4 * ((x - O.1)^2 + (y - O.2)^2)

-- Define the trajectory
def Trajectory : ℝ × ℝ → Prop := fun (x, y) ↦ 
  x^2 + (y + 1)^2 = 4

-- Define the line equations
def Line1 : ℝ × ℝ → Prop := fun (x, y) ↦ x = -1
def Line2 : ℝ × ℝ → Prop := fun (x, y) ↦ 15*x + 8*y - 9 = 0

theorem trajectory_and_line_equations :
  (∀ p, M p ↔ Trajectory p) ∧
  (∃ l, (l = Line1 ∨ l = Line2) ∧
        (l N) ∧
        (∃ p q : ℝ × ℝ, p ≠ q ∧ Trajectory p ∧ Trajectory q ∧ l p ∧ l q ∧
          (p.1 - q.1)^2 + (p.2 - q.2)^2 = 12)) :=
by sorry

end trajectory_and_line_equations_l894_89404


namespace cookie_box_cost_l894_89406

/-- Given Faye's initial money, her mother's contribution, cupcake purchases, and remaining money,
    prove that each box of cookies costs $3. -/
theorem cookie_box_cost (initial_money : ℚ) (cupcake_price : ℚ) (num_cupcakes : ℕ) 
  (num_cookie_boxes : ℕ) (money_left : ℚ) :
  initial_money = 20 →
  cupcake_price = 3/2 →
  num_cupcakes = 10 →
  num_cookie_boxes = 5 →
  money_left = 30 →
  let total_money := initial_money + 2 * initial_money
  let money_after_cupcakes := total_money - (cupcake_price * num_cupcakes)
  let cookie_boxes_cost := money_after_cupcakes - money_left
  cookie_boxes_cost / num_cookie_boxes = 3 :=
by sorry


end cookie_box_cost_l894_89406


namespace donnas_truck_weight_l894_89442

-- Define the given weights and quantities
def bridge_limit : ℕ := 20000
def empty_truck_weight : ℕ := 12000
def soda_crates : ℕ := 20
def soda_crate_weight : ℕ := 50
def dryers : ℕ := 3
def dryer_weight : ℕ := 3000

-- Define the theorem
theorem donnas_truck_weight :
  let soda_weight := soda_crates * soda_crate_weight
  let produce_weight := 2 * soda_weight
  let dryers_weight := dryers * dryer_weight
  let total_weight := empty_truck_weight + soda_weight + produce_weight + dryers_weight
  total_weight = 24000 := by
  sorry

end donnas_truck_weight_l894_89442


namespace tanya_work_days_l894_89419

/-- Given Sakshi can do a piece of work in 12 days and Tanya is 20% more efficient than Sakshi,
    prove that Tanya can complete the same piece of work in 10 days. -/
theorem tanya_work_days (sakshi_days : ℝ) (tanya_efficiency : ℝ) :
  sakshi_days = 12 →
  tanya_efficiency = 1.2 →
  (sakshi_days / tanya_efficiency) = 10 := by
sorry

end tanya_work_days_l894_89419


namespace equation_solution_l894_89440

theorem equation_solution (x : ℚ) (h1 : x ≠ 3) (h2 : x ≠ -2) :
  (x + 4) / (x - 3) = (x - 2) / (x + 2) ↔ x = -2/11 := by
  sorry

end equation_solution_l894_89440


namespace number_difference_l894_89494

theorem number_difference (x y : ℝ) : 
  x + y = 50 →
  3 * max x y - 5 * min x y = 10 →
  |x - y| = 15 := by
sorry

end number_difference_l894_89494


namespace system_solution_l894_89467

theorem system_solution (x y : ℝ) : 
  x^3 + y^3 = 1 ∧ x^4 + y^4 = 1 ↔ (x = 1 ∧ y = 0) ∨ (x = 0 ∧ y = 1) := by
  sorry

end system_solution_l894_89467


namespace partial_fraction_decomposition_l894_89402

theorem partial_fraction_decomposition (D E F : ℝ) :
  (∀ x : ℝ, x ≠ 1 ∧ x ≠ -2 ∧ x ≠ 6 →
    1 / (x^3 - 3*x^2 - 4*x + 12) = D / (x - 1) + E / (x + 2) + F / (x + 2)^2) →
  D = -1/15 := by
sorry

end partial_fraction_decomposition_l894_89402


namespace right_triangle_sets_l894_89485

/-- Checks if three numbers can form a right triangle --/
def isRightTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2)

theorem right_triangle_sets :
  ¬(isRightTriangle 4 6 8) ∧
  (isRightTriangle 5 12 13) ∧
  (isRightTriangle 6 8 10) ∧
  (isRightTriangle 7 24 25) :=
by sorry

end right_triangle_sets_l894_89485


namespace no_positive_integer_solutions_for_quadratic_l894_89437

theorem no_positive_integer_solutions_for_quadratic :
  ∀ A : ℕ, 1 ≤ A → A ≤ 9 →
    ¬∃ x : ℕ, x > 0 ∧ x^2 - (A + 1) * x + A * 2 = 0 :=
by sorry

end no_positive_integer_solutions_for_quadratic_l894_89437


namespace tailwind_speed_l894_89481

def plane_speed_with_tailwind : ℝ := 460
def plane_speed_against_tailwind : ℝ := 310

theorem tailwind_speed : ∃ (plane_speed tailwind_speed : ℝ),
  plane_speed + tailwind_speed = plane_speed_with_tailwind ∧
  plane_speed - tailwind_speed = plane_speed_against_tailwind ∧
  tailwind_speed = 75 := by
  sorry

end tailwind_speed_l894_89481


namespace b_31_mod_33_l894_89473

/-- Definition of b_n as the concatenation of integers from 1 to n --/
def b (n : ℕ) : ℕ :=
  -- This is a placeholder definition. The actual implementation would be more complex.
  sorry

/-- Theorem stating that b_31 mod 33 = 11 --/
theorem b_31_mod_33 : b 31 % 33 = 11 := by
  sorry

end b_31_mod_33_l894_89473


namespace hike_pace_proof_l894_89466

/-- Proves that given the conditions of the hike, the pace to the destination is 4 miles per hour -/
theorem hike_pace_proof (distance : ℝ) (return_pace : ℝ) (total_time : ℝ) (pace_to : ℝ) : 
  distance = 12 → 
  return_pace = 6 → 
  total_time = 5 → 
  distance / pace_to + distance / return_pace = total_time → 
  pace_to = 4 := by
sorry

end hike_pace_proof_l894_89466


namespace investment_timing_l894_89410

/-- Proves that B invested after 6 months given the conditions of the investment problem -/
theorem investment_timing (a_investment : ℕ) (b_investment : ℕ) (total_profit : ℕ) (a_profit : ℕ) :
  a_investment = 150 →
  b_investment = 200 →
  total_profit = 100 →
  a_profit = 60 →
  ∃ x : ℕ,
    x = 6 ∧
    (a_investment * 12 : ℚ) / (b_investment * (12 - x)) = (a_profit : ℚ) / (total_profit - a_profit) :=
by
  sorry


end investment_timing_l894_89410


namespace screamers_lineup_count_l894_89451

-- Define the total number of players
def total_players : ℕ := 12

-- Define the number of players in a lineup
def lineup_size : ℕ := 5

-- Define a function to calculate combinations
def combinations (n : ℕ) (r : ℕ) : ℕ := 
  Nat.choose n r

-- Theorem statement
theorem screamers_lineup_count : 
  combinations (total_players - 2) (lineup_size - 1) * 2 + 
  combinations (total_players - 2) lineup_size = 672 := by
  sorry


end screamers_lineup_count_l894_89451


namespace negation_of_existence_negation_of_proposition_l894_89474

theorem negation_of_existence (p : ℕ → Prop) : 
  (¬ ∃ x : ℕ, p x) ↔ (∀ x : ℕ, ¬ p x) := by sorry

theorem negation_of_proposition : 
  (¬ ∃ x : ℕ, x^2 ≥ x) ↔ (∀ x : ℕ, x^2 < x) := by sorry

end negation_of_existence_negation_of_proposition_l894_89474


namespace sum_of_roots_equals_36_l894_89400

theorem sum_of_roots_equals_36 : ∃ (x₁ x₂ x₃ : ℝ),
  (∀ x, (11 - x)^3 + (13 - x)^3 = (24 - 2*x)^3 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
  x₁ + x₂ + x₃ = 36 := by
  sorry

end sum_of_roots_equals_36_l894_89400


namespace papaya_tree_growth_ratio_l894_89499

/-- Papaya tree growth problem -/
theorem papaya_tree_growth_ratio : 
  ∀ (growth_1 growth_2 growth_3 growth_4 growth_5 : ℝ),
  growth_1 = 2 →
  growth_2 = growth_1 * 1.5 →
  growth_3 = growth_2 * 1.5 →
  growth_5 = growth_4 / 2 →
  growth_1 + growth_2 + growth_3 + growth_4 + growth_5 = 23 →
  growth_4 / growth_3 = 2 := by
sorry


end papaya_tree_growth_ratio_l894_89499


namespace min_value_expression_l894_89407

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 1 / (2 * y))^2 + (y + 1 / (2 * x))^2 ≥ 4 ∧
  ((x + 1 / (2 * y))^2 + (y + 1 / (2 * x))^2 = 4 ↔ x = y ∧ x = Real.sqrt 2 / 2) :=
by sorry

end min_value_expression_l894_89407
