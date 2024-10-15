import Mathlib

namespace NUMINAMATH_CALUDE_bianca_books_total_l40_4039

theorem bianca_books_total (books_per_shelf : ℕ) (mystery_shelves : ℕ) (picture_shelves : ℕ)
  (h1 : books_per_shelf = 8)
  (h2 : mystery_shelves = 5)
  (h3 : picture_shelves = 4) :
  books_per_shelf * (mystery_shelves + picture_shelves) = 72 :=
by sorry

end NUMINAMATH_CALUDE_bianca_books_total_l40_4039


namespace NUMINAMATH_CALUDE_volume_of_specific_tetrahedron_l40_4024

/-- The volume of a tetrahedron given its edge lengths -/
def tetrahedron_volume (pq pr ps qr qs rs : ℝ) : ℝ := sorry

/-- Theorem: The volume of tetrahedron PQRS with given edge lengths is 140/9 -/
theorem volume_of_specific_tetrahedron :
  let pq : ℝ := 6
  let pr : ℝ := 5
  let ps : ℝ := 4 * Real.sqrt 2
  let qr : ℝ := 3 * Real.sqrt 2
  let qs : ℝ := 5
  let rs : ℝ := 4
  tetrahedron_volume pq pr ps qr qs rs = 140 / 9 := by sorry

end NUMINAMATH_CALUDE_volume_of_specific_tetrahedron_l40_4024


namespace NUMINAMATH_CALUDE_average_study_time_difference_l40_4041

def average_difference (differences : List Int) : ℚ :=
  (differences.sum : ℚ) / differences.length

theorem average_study_time_difference 
  (differences : List Int) 
  (h1 : differences.length = 5) :
  average_difference differences = 
    (differences.sum : ℚ) / 5 := by sorry

end NUMINAMATH_CALUDE_average_study_time_difference_l40_4041


namespace NUMINAMATH_CALUDE_childrens_book_weight_l40_4082

-- Define the weight of a comic book
def comic_book_weight : ℝ := 0.8

-- Define the total weight of all books
def total_weight : ℝ := 10.98

-- Define the number of comic books
def num_comic_books : ℕ := 9

-- Define the number of children's books
def num_children_books : ℕ := 7

-- Theorem to prove
theorem childrens_book_weight :
  (total_weight - (num_comic_books : ℝ) * comic_book_weight) / num_children_books = 0.54 := by
  sorry

end NUMINAMATH_CALUDE_childrens_book_weight_l40_4082


namespace NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l40_4083

theorem smallest_number_satisfying_conditions : 
  ∃ (n : ℕ), 
    (∀ (m : ℕ), m < n → ¬((m + 3) % 5 = 0 ∧ (m - 3) % 6 = 0)) ∧
    (n + 3) % 5 = 0 ∧ 
    (n - 3) % 6 = 0 ∧
    n = 27 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l40_4083


namespace NUMINAMATH_CALUDE_evaluate_expression_l40_4087

theorem evaluate_expression : (120 : ℚ) / 6 * 2 / 3 = 40 / 3 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l40_4087


namespace NUMINAMATH_CALUDE_problem_solution_l40_4027

theorem problem_solution (m n : ℝ) (h1 : m + 1/m = -4) (h2 : n + 1/n = -4) (h3 : m ≠ n) : 
  m * (n + 1) + n = -3 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l40_4027


namespace NUMINAMATH_CALUDE_roses_count_l40_4052

theorem roses_count (vase_capacity : ℕ) (carnations : ℕ) (vases : ℕ) : 
  vase_capacity = 9 → carnations = 4 → vases = 3 → 
  vases * vase_capacity - carnations = 23 := by
  sorry

end NUMINAMATH_CALUDE_roses_count_l40_4052


namespace NUMINAMATH_CALUDE_largest_complete_graph_with_arithmetic_progression_edges_l40_4008

/-- A function that assigns non-negative integers to edges of a complete graph -/
def EdgeAssignment (n : ℕ) := Fin n → Fin n → ℕ

/-- Predicate to check if three numbers form an arithmetic progression -/
def IsArithmeticProgression (a b c : ℕ) : Prop := 2 * b = a + c

/-- Predicate to check if all edges of a triangle form an arithmetic progression -/
def TriangleIsArithmeticProgression (f : EdgeAssignment n) (i j k : Fin n) : Prop :=
  IsArithmeticProgression (f i j) (f i k) (f j k) ∧
  IsArithmeticProgression (f i j) (f j k) (f i k) ∧
  IsArithmeticProgression (f i k) (f j k) (f i j)

/-- Predicate to check if the edge assignment is valid -/
def ValidAssignment (n : ℕ) (f : EdgeAssignment n) : Prop :=
  (∀ i j : Fin n, i ≠ j → f i j = f j i) ∧ 
  (∀ i j k : Fin n, i ≠ j → j ≠ k → i ≠ k → TriangleIsArithmeticProgression f i j k) ∧
  (∀ i j k l : Fin n, i ≠ j → i ≠ k → i ≠ l → j ≠ k → j ≠ l → k ≠ l → 
    f i j ≠ f i k ∧ f i j ≠ f i l ∧ f i j ≠ f j k ∧ f i j ≠ f j l ∧ f i j ≠ f k l ∧
    f i k ≠ f i l ∧ f i k ≠ f j k ∧ f i k ≠ f j l ∧ f i k ≠ f k l ∧
    f i l ≠ f j k ∧ f i l ≠ f j l ∧ f i l ≠ f k l ∧
    f j k ≠ f j l ∧ f j k ≠ f k l ∧
    f j l ≠ f k l)

theorem largest_complete_graph_with_arithmetic_progression_edges :
  (∃ f : EdgeAssignment 4, ValidAssignment 4 f) ∧
  (∀ n : ℕ, n > 4 → ¬∃ f : EdgeAssignment n, ValidAssignment n f) :=
sorry

end NUMINAMATH_CALUDE_largest_complete_graph_with_arithmetic_progression_edges_l40_4008


namespace NUMINAMATH_CALUDE_range_of_x_inequality_l40_4034

theorem range_of_x_inequality (x : ℝ) : 
  (∀ (a b : ℝ), a ≠ 0 → |a + b| + |a - b| ≥ |a| * |x - 2|) ↔ 0 ≤ x ∧ x ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_inequality_l40_4034


namespace NUMINAMATH_CALUDE_mark_fish_problem_l40_4084

/-- Calculates the total number of young fish given the number of tanks, 
    pregnant fish per tank, and young per fish. -/
def total_young_fish (num_tanks : ℕ) (fish_per_tank : ℕ) (young_per_fish : ℕ) : ℕ :=
  num_tanks * fish_per_tank * young_per_fish

/-- Theorem stating that with 3 tanks, 4 pregnant fish per tank, and 20 young per fish, 
    the total number of young fish is 240. -/
theorem mark_fish_problem :
  total_young_fish 3 4 20 = 240 := by
  sorry

end NUMINAMATH_CALUDE_mark_fish_problem_l40_4084


namespace NUMINAMATH_CALUDE_election_votes_theorem_l40_4047

theorem election_votes_theorem (emily_votes : ℕ) (emily_fraction : ℚ) (dexter_fraction : ℚ) :
  emily_votes = 48 →
  emily_fraction = 4 / 15 →
  dexter_fraction = 1 / 3 →
  ∃ (total_votes : ℕ),
    (emily_votes : ℚ) / total_votes = emily_fraction ∧
    total_votes = 180 :=
by sorry

end NUMINAMATH_CALUDE_election_votes_theorem_l40_4047


namespace NUMINAMATH_CALUDE_hidden_numbers_average_l40_4055

/-- A card with two numbers -/
structure Card where
  visible : ℕ
  hidden : ℕ

/-- The problem setup -/
def problem_setup (cards : Fin 3 → Card) : Prop :=
  -- The sums of the numbers on each card are the same
  (∃ s : ℕ, ∀ i : Fin 3, (cards i).visible + (cards i).hidden = s) ∧
  -- Visible numbers are 81, 52, and 47
  (cards 0).visible = 81 ∧ (cards 1).visible = 52 ∧ (cards 2).visible = 47 ∧
  -- Hidden numbers are all prime
  (∀ i : Fin 3, Nat.Prime (cards i).hidden) ∧
  -- All numbers are different
  (∀ i j : Fin 3, i ≠ j → (cards i).visible ≠ (cards j).visible ∧ 
                         (cards i).hidden ≠ (cards j).hidden ∧
                         (cards i).visible ≠ (cards j).hidden)

/-- The theorem to prove -/
theorem hidden_numbers_average (cards : Fin 3 → Card) 
  (h : problem_setup cards) : 
  (cards 0).hidden + (cards 1).hidden + (cards 2).hidden = 119 := by
  sorry

#check hidden_numbers_average

end NUMINAMATH_CALUDE_hidden_numbers_average_l40_4055


namespace NUMINAMATH_CALUDE_ones_digit_of_11_to_46_l40_4096

theorem ones_digit_of_11_to_46 : (11^46 : ℕ) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_11_to_46_l40_4096


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l40_4031

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 + 8*x + 9 = 0 ↔ (x + 4)^2 = 7 :=
by sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l40_4031


namespace NUMINAMATH_CALUDE_diamond_two_three_l40_4077

-- Define the diamond operation
def diamond (a b : ℕ) : ℕ := a * b^2 - b + 1

-- Theorem statement
theorem diamond_two_three : diamond 2 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_diamond_two_three_l40_4077


namespace NUMINAMATH_CALUDE_wednesday_savings_l40_4080

/-- Represents Donny's savings throughout the week -/
structure WeekSavings where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ

/-- Calculates the total savings before Thursday -/
def total_savings (s : WeekSavings) : ℕ :=
  s.monday + s.tuesday + s.wednesday

theorem wednesday_savings (s : WeekSavings) 
  (h1 : s.monday = 15)
  (h2 : s.tuesday = 28)
  (h3 : total_savings s / 2 = 28) : 
  s.wednesday = 13 := by
sorry

end NUMINAMATH_CALUDE_wednesday_savings_l40_4080


namespace NUMINAMATH_CALUDE_strudel_price_calculation_l40_4002

/-- Calculates the final price of a strudel after two 50% increases and a 50% decrease -/
def finalPrice (initialPrice : ℝ) : ℝ :=
  initialPrice * 1.5 * 1.5 * 0.5

/-- Theorem stating that the final price of a strudel is 90 rubles -/
theorem strudel_price_calculation :
  finalPrice 80 = 90 := by
  sorry

end NUMINAMATH_CALUDE_strudel_price_calculation_l40_4002


namespace NUMINAMATH_CALUDE_only_third_proposition_true_l40_4049

theorem only_third_proposition_true :
  ∃ (a b c d : ℝ),
    (∃ c, a > b ∧ c ≠ 0 ∧ ¬(a * c > b * c)) ∧
    (∃ c, a > b ∧ ¬(a * c^2 > b * c^2)) ∧
    (∀ a b c : ℝ, a * c^2 > b * c^2 → a > b) ∧
    (∃ a b, a > b ∧ ¬(1 / a < 1 / b)) ∧
    (∃ a b c d, a > b ∧ b > 0 ∧ c > d ∧ ¬(a * c > b * d)) :=
by sorry

end NUMINAMATH_CALUDE_only_third_proposition_true_l40_4049


namespace NUMINAMATH_CALUDE_final_bacteria_count_l40_4010

def initial_count : ℕ := 30
def start_time : ℕ := 0  -- 10:00 AM represented as 0 minutes
def end_time : ℕ := 30   -- 10:30 AM represented as 30 minutes
def growth_interval : ℕ := 5  -- population triples every 5 minutes
def death_interval : ℕ := 15  -- 10% die every 15 minutes

def growth_factor : ℚ := 3
def survival_rate : ℚ := 0.9  -- 90% survival rate (10% die)

def number_of_growth_periods (t : ℕ) : ℕ := t / growth_interval

def number_of_death_periods (t : ℕ) : ℕ := t / death_interval

def bacteria_count (t : ℕ) : ℚ :=
  initial_count *
  growth_factor ^ (number_of_growth_periods t) *
  survival_rate ^ (number_of_death_periods t)

theorem final_bacteria_count :
  bacteria_count end_time = 17694 := by sorry

end NUMINAMATH_CALUDE_final_bacteria_count_l40_4010


namespace NUMINAMATH_CALUDE_base_9_8_conversion_l40_4078

/-- Represents a number in a given base -/
def BaseRepresentation (base : ℕ) (tens_digit : ℕ) (ones_digit : ℕ) : ℕ :=
  base * tens_digit + ones_digit

theorem base_9_8_conversion : 
  ∃ (n : ℕ) (C D : ℕ), 
    C < 9 ∧ D < 9 ∧ D < 8 ∧ 
    n = BaseRepresentation 9 C D ∧
    n = BaseRepresentation 8 D C ∧
    n = 71 := by
  sorry

end NUMINAMATH_CALUDE_base_9_8_conversion_l40_4078


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l40_4071

theorem sqrt_equation_solution (x : ℝ) (h : x > 6) :
  Real.sqrt (x - 6 * Real.sqrt (x - 6)) + 3 = Real.sqrt (x + 6 * Real.sqrt (x - 6)) - 3 ↔ x ≥ 18 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l40_4071


namespace NUMINAMATH_CALUDE_correct_calculation_l40_4091

theorem correct_calculation (x : ℝ) : (x / 7 = 49) → (x * 6 = 2058) := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l40_4091


namespace NUMINAMATH_CALUDE_intersection_P_Q_l40_4014

-- Define set P
def P : Set ℝ := {x | x * (x - 3) < 0}

-- Define set Q
def Q : Set ℝ := {x | |x| < 2}

-- Theorem statement
theorem intersection_P_Q : P ∩ Q = Set.Ioo 0 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l40_4014


namespace NUMINAMATH_CALUDE_incircle_tangents_concurrent_l40_4067

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a line in 2D space -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Represents a circle in 2D space -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (a : Point) (b : Point) (c : Point)

/-- Checks if three lines are concurrent -/
def are_concurrent (l1 l2 l3 : Line) : Prop := sorry

/-- Returns the tangent line to a circle at a given point -/
def tangent_line (c : Circle) (p : Point) : Line := sorry

/-- Returns the line passing through two points -/
def line_through_points (p1 p2 : Point) : Line := sorry

/-- Checks if a point lies on a line -/
def point_on_line (p : Point) (l : Line) : Prop := sorry

/-- Checks if a point lies on a circle -/
def point_on_circle (p : Point) (c : Circle) : Prop := sorry

theorem incircle_tangents_concurrent 
  (t : Triangle) 
  (incircle : Circle) 
  (m n k l : Point) :
  point_on_circle m incircle →
  point_on_circle n incircle →
  point_on_circle k incircle →
  point_on_circle l incircle →
  point_on_line m (line_through_points t.a t.b) →
  point_on_line n (line_through_points t.b t.c) →
  point_on_line k (line_through_points t.c t.a) →
  point_on_line l (line_through_points t.a t.c) →
  are_concurrent 
    (line_through_points m n)
    (line_through_points k l)
    (tangent_line incircle t.a) :=
by
  sorry

end NUMINAMATH_CALUDE_incircle_tangents_concurrent_l40_4067


namespace NUMINAMATH_CALUDE_sum_of_squared_coefficients_l40_4038

def expression (x : ℝ) : ℝ := 5 * (x^2 - 3*x + 4) - 9 * (x^3 - 2*x^2 + x - 1)

theorem sum_of_squared_coefficients :
  ∃ (a b c d : ℝ),
    (∀ x, expression x = a*x^3 + b*x^2 + c*x + d) ∧
    a^2 + b^2 + c^2 + d^2 = 2027 := by sorry

end NUMINAMATH_CALUDE_sum_of_squared_coefficients_l40_4038


namespace NUMINAMATH_CALUDE_multiple_of_nine_in_range_l40_4025

theorem multiple_of_nine_in_range (y : ℕ) :
  y > 0 ∧ 
  ∃ k : ℕ, y = 9 * k ∧ 
  y^2 > 225 ∧ 
  y < 30 →
  y = 18 ∨ y = 27 := by
sorry

end NUMINAMATH_CALUDE_multiple_of_nine_in_range_l40_4025


namespace NUMINAMATH_CALUDE_a_minus_b_pow_2014_l40_4090

theorem a_minus_b_pow_2014 (a b : ℝ) 
  (ha : a^3 - 6*a^2 + 15*a = 9) 
  (hb : b^3 - 3*b^2 + 6*b = -1) : 
  (a - b)^2014 = 1 := by sorry

end NUMINAMATH_CALUDE_a_minus_b_pow_2014_l40_4090


namespace NUMINAMATH_CALUDE_conical_funnel_area_l40_4000

/-- The area of cardboard required for a conical funnel -/
theorem conical_funnel_area (slant_height : ℝ) (base_circumference : ℝ) 
  (h1 : slant_height = 6)
  (h2 : base_circumference = 6 * Real.pi) : 
  (1 / 2 : ℝ) * base_circumference * slant_height = 18 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_conical_funnel_area_l40_4000


namespace NUMINAMATH_CALUDE_triangle_properties_l40_4097

-- Define a triangle
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

-- Define geometric elements
def angleBisector (t : Triangle) : Fin 3 → Set (ℝ × ℝ) := sorry
def median (t : Triangle) : Fin 3 → Set (ℝ × ℝ) := sorry
def altitude (t : Triangle) : Fin 3 → Set (ℝ × ℝ) := sorry

-- Define properties
def isInside (t : Triangle) (s : Set (ℝ × ℝ)) : Prop := sorry
def isRightTriangle (t : Triangle) : Prop := sorry
def isLine (s : Set (ℝ × ℝ)) : Prop := sorry
def isRay (s : Set (ℝ × ℝ)) : Prop := sorry
def isLineSegment (s : Set (ℝ × ℝ)) : Prop := sorry

-- Theorem statement
theorem triangle_properties :
  ∃ (t : Triangle),
    (¬ (∀ i : Fin 3, isInside t (angleBisector t i) ∧ isInside t (median t i) ∧ isInside t (altitude t i))) ∧
    (isRightTriangle t → ∃ i j : Fin 3, i ≠ j ∧ altitude t i ≠ altitude t j) ∧
    (∃ i : Fin 3, isInside t (altitude t i)) ∧
    (¬ (∀ i : Fin 3, isLine (altitude t i) ∧ isRay (angleBisector t i) ∧ isLineSegment (median t i))) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l40_4097


namespace NUMINAMATH_CALUDE_multiplicative_inverse_modulo_l40_4068

theorem multiplicative_inverse_modulo (A' B' : Nat) (m : Nat) (h : m = 2000000) :
  A' = 222222 →
  B' = 285714 →
  (1500000 * (A' * B')) % m = 1 :=
by sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_modulo_l40_4068


namespace NUMINAMATH_CALUDE_polar_to_cartesian_circle_l40_4003

/-- The polar equation r = 1 / (sin θ + cos θ) represents a circle in Cartesian coordinates -/
theorem polar_to_cartesian_circle :
  ∃ (h k r : ℝ), ∀ (x y : ℝ),
    (∃ (θ : ℝ), x = (1 / (Real.sin θ + Real.cos θ)) * Real.cos θ ∧
                 y = (1 / (Real.sin θ + Real.cos θ)) * Real.sin θ) →
    (x - h)^2 + (y - k)^2 = r^2 :=
by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_circle_l40_4003


namespace NUMINAMATH_CALUDE_simplified_ratio_of_stickers_l40_4043

theorem simplified_ratio_of_stickers (kate_stickers : ℕ) (jenna_stickers : ℕ) 
  (h1 : kate_stickers = 21) (h2 : jenna_stickers = 12) : 
  (kate_stickers / Nat.gcd kate_stickers jenna_stickers : ℚ) / 
  (jenna_stickers / Nat.gcd kate_stickers jenna_stickers : ℚ) = 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_simplified_ratio_of_stickers_l40_4043


namespace NUMINAMATH_CALUDE_lunch_cost_proof_l40_4093

theorem lunch_cost_proof (adam_cost rick_cost jose_cost total_cost : ℚ) : 
  adam_cost = (2 : ℚ) / (3 : ℚ) * rick_cost →
  rick_cost = jose_cost →
  jose_cost = 45 →
  total_cost = adam_cost + rick_cost + jose_cost →
  total_cost = 120 := by
sorry

end NUMINAMATH_CALUDE_lunch_cost_proof_l40_4093


namespace NUMINAMATH_CALUDE_ways_to_fifth_floor_l40_4057

/-- Represents a building with a specified number of floors and staircases between each floor. -/
structure Building where
  floors : ℕ
  staircases : ℕ

/-- Calculates the number of different ways to go from the first floor to the top floor. -/
def waysToTopFloor (b : Building) : ℕ :=
  b.staircases ^ (b.floors - 1)

/-- Theorem stating that in a 5-floor building with 2 staircases between each pair of consecutive floors,
    the number of different ways to go from the first floor to the fifth floor is 2^4. -/
theorem ways_to_fifth_floor :
  let b : Building := { floors := 5, staircases := 2 }
  waysToTopFloor b = 2^4 := by
  sorry

end NUMINAMATH_CALUDE_ways_to_fifth_floor_l40_4057


namespace NUMINAMATH_CALUDE_fiscal_revenue_scientific_notation_l40_4045

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coeff : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation := sorry

/-- Rounds a ScientificNotation to a specified number of significant figures -/
def roundToSignificantFigures (sn : ScientificNotation) (figures : ℕ) : ScientificNotation := sorry

/-- The fiscal revenue in yuan -/
def fiscalRevenue : ℝ := 1073 * 10^9

theorem fiscal_revenue_scientific_notation :
  roundToSignificantFigures (toScientificNotation fiscalRevenue) 2 =
  ScientificNotation.mk 1.07 11 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_fiscal_revenue_scientific_notation_l40_4045


namespace NUMINAMATH_CALUDE_stratified_sample_theorem_l40_4006

/-- Represents a stratified sampling scenario in a high school -/
structure StratifiedSample where
  total_students : ℕ
  liberal_arts_students : ℕ
  sample_size : ℕ

/-- Calculates the expected number of liberal arts students in the sample -/
def expected_liberal_arts_in_sample (s : StratifiedSample) : ℕ :=
  (s.liberal_arts_students * s.sample_size) / s.total_students

/-- Theorem stating the expected number of liberal arts students in the sample -/
theorem stratified_sample_theorem (s : StratifiedSample) 
  (h1 : s.total_students = 1000)
  (h2 : s.liberal_arts_students = 200)
  (h3 : s.sample_size = 100) :
  expected_liberal_arts_in_sample s = 20 := by
  sorry

#eval expected_liberal_arts_in_sample { total_students := 1000, liberal_arts_students := 200, sample_size := 100 }

end NUMINAMATH_CALUDE_stratified_sample_theorem_l40_4006


namespace NUMINAMATH_CALUDE_expression_evaluation_l40_4059

theorem expression_evaluation :
  let a : ℤ := -2
  3 * a * (2 * a^2 - 4 * a + 3) - 2 * a^2 * (3 * a + 4) = -98 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l40_4059


namespace NUMINAMATH_CALUDE_equation_solution_l40_4016

theorem equation_solution : 
  ∃ x : ℝ, 0.3 * x + (0.4 * 0.5) = 0.26 ∧ x = 0.2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l40_4016


namespace NUMINAMATH_CALUDE_probability_ratio_l40_4013

/-- The number of balls -/
def num_balls : ℕ := 24

/-- The number of bins -/
def num_bins : ℕ := 6

/-- The probability of the first distribution (6-6-3-3-3-3) -/
noncomputable def p : ℝ := sorry

/-- The probability of the second distribution (4-4-4-4-4-4) -/
noncomputable def q : ℝ := sorry

/-- Theorem stating that the ratio of probabilities p and q is 12 -/
theorem probability_ratio : p / q = 12 := by sorry

end NUMINAMATH_CALUDE_probability_ratio_l40_4013


namespace NUMINAMATH_CALUDE_child_workers_count_l40_4061

/-- Represents the number of workers of each type and their daily wages --/
structure WorkforceData where
  male_workers : ℕ
  female_workers : ℕ
  male_wage : ℕ
  female_wage : ℕ
  child_wage : ℕ
  average_wage : ℕ

/-- Calculates the number of child workers given the workforce data --/
def calculate_child_workers (data : WorkforceData) : ℕ :=
  let total_workers := data.male_workers + data.female_workers
  let total_wage := data.male_workers * data.male_wage + data.female_workers * data.female_wage
  let x := (data.average_wage * total_workers - total_wage) / (data.average_wage - data.child_wage)
  x

/-- Theorem stating that the number of child workers is 5 given the specific workforce data --/
theorem child_workers_count (data : WorkforceData) 
  (h1 : data.male_workers = 20)
  (h2 : data.female_workers = 15)
  (h3 : data.male_wage = 35)
  (h4 : data.female_wage = 20)
  (h5 : data.child_wage = 8)
  (h6 : data.average_wage = 26) :
  calculate_child_workers data = 5 := by
  sorry

end NUMINAMATH_CALUDE_child_workers_count_l40_4061


namespace NUMINAMATH_CALUDE_initial_books_count_l40_4089

def books_sold : ℕ := 26
def books_left : ℕ := 7

theorem initial_books_count :
  books_sold + books_left = 33 := by sorry

end NUMINAMATH_CALUDE_initial_books_count_l40_4089


namespace NUMINAMATH_CALUDE_regression_lines_intersection_l40_4028

-- Define the regression line type
def RegressionLine := ℝ → ℝ

-- Define the property that a regression line passes through a point
def passes_through (l : RegressionLine) (p : ℝ × ℝ) : Prop :=
  l p.1 = p.2

-- Theorem statement
theorem regression_lines_intersection
  (l₁ l₂ : RegressionLine)
  (s t : ℝ)
  (h₁ : passes_through l₁ (s, t))
  (h₂ : passes_through l₂ (s, t)) :
  ∃ p : ℝ × ℝ, p = (s, t) ∧ passes_through l₁ p ∧ passes_through l₂ p :=
sorry

end NUMINAMATH_CALUDE_regression_lines_intersection_l40_4028


namespace NUMINAMATH_CALUDE_macaroon_packing_l40_4026

/-- The number of brown bags used to pack macaroons -/
def number_of_bags : ℕ := 4

/-- The total number of macaroons -/
def total_macaroons : ℕ := 12

/-- The weight of each macaroon in ounces -/
def macaroon_weight : ℕ := 5

/-- The remaining weight of macaroons after one bag is eaten, in ounces -/
def remaining_weight : ℕ := 45

theorem macaroon_packing :
  (total_macaroons % number_of_bags = 0) ∧
  (total_macaroons / number_of_bags * macaroon_weight = 
   total_macaroons * macaroon_weight - remaining_weight) →
  number_of_bags = 4 := by
sorry

end NUMINAMATH_CALUDE_macaroon_packing_l40_4026


namespace NUMINAMATH_CALUDE_min_skilled_players_exists_tournament_with_three_skilled_l40_4019

/-- Represents a player in the tournament -/
def Player := Fin 2023

/-- Represents the result of a match between two players -/
def MatchResult := Player → Player → Prop

/-- A player is skilled if for every player who defeats them, there exists another player who defeats that player and loses to the skilled player -/
def IsSkilled (result : MatchResult) (p : Player) : Prop :=
  ∀ q, result q p → ∃ r, result p r ∧ result r q

/-- The tournament satisfies the given conditions -/
def ValidTournament (result : MatchResult) : Prop :=
  (∀ p q, p ≠ q → (result p q ∨ result q p)) ∧
  (∀ p, ¬(∀ q, p ≠ q → result p q))

/-- The main theorem: there are at least 3 skilled players in any valid tournament -/
theorem min_skilled_players (result : MatchResult) (h : ValidTournament result) :
  ∃ a b c : Player, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ IsSkilled result a ∧ IsSkilled result b ∧ IsSkilled result c :=
sorry

/-- There exists a valid tournament with exactly 3 skilled players -/
theorem exists_tournament_with_three_skilled :
  ∃ result : MatchResult, ValidTournament result ∧
  (∃ a b c : Player, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    IsSkilled result a ∧ IsSkilled result b ∧ IsSkilled result c ∧
    (∀ p, IsSkilled result p → p = a ∨ p = b ∨ p = c)) :=
sorry

end NUMINAMATH_CALUDE_min_skilled_players_exists_tournament_with_three_skilled_l40_4019


namespace NUMINAMATH_CALUDE_parallel_transitive_perpendicular_to_plane_parallel_l40_4044

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relation between lines
variable (parallel : Line → Line → Prop)

-- Define the perpendicularity relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Axiom for transitivity of parallelism
axiom parallel_trans {a b c : Line} : parallel a b → parallel b c → parallel a c

-- Axiom for perpendicular lines to a plane being parallel
axiom perpendicular_parallel {a b : Line} {γ : Plane} : 
  perpendicular a γ → perpendicular b γ → parallel a b

-- Theorem 1: If a∥b and b∥c, then a∥c
theorem parallel_transitive {a b c : Line} : 
  parallel a b → parallel b c → parallel a c :=
by sorry

-- Theorem 2: If a⊥γ and b⊥γ, then a∥b
theorem perpendicular_to_plane_parallel {a b : Line} {γ : Plane} : 
  perpendicular a γ → perpendicular b γ → parallel a b :=
by sorry

end NUMINAMATH_CALUDE_parallel_transitive_perpendicular_to_plane_parallel_l40_4044


namespace NUMINAMATH_CALUDE_remaining_distance_l40_4058

-- Define the total distance to the concert
def total_distance : ℕ := 78

-- Define the distance already driven
def distance_driven : ℕ := 32

-- Theorem to prove the remaining distance
theorem remaining_distance : total_distance - distance_driven = 46 := by
  sorry

end NUMINAMATH_CALUDE_remaining_distance_l40_4058


namespace NUMINAMATH_CALUDE_fraction_reduction_l40_4015

theorem fraction_reduction (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) :
  (4*x - 4*y) / (4*x * 4*y) = (1/4) * ((x - y) / (x * y)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_reduction_l40_4015


namespace NUMINAMATH_CALUDE_midpoint_x_coordinate_constant_l40_4037

/-- Parabola structure -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ

/-- Line structure -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- Defines a point on a parabola -/
def PointOnParabola (p : Parabola) (x y : ℝ) : Prop :=
  p.equation x y

/-- Defines the perpendicular bisector of a line segment -/
def PerpendicularBisector (A B M : ℝ × ℝ) : Prop :=
  -- Definition of perpendicular bisector passing through M
  sorry

/-- Main theorem -/
theorem midpoint_x_coordinate_constant
  (p : Parabola)
  (A B : ℝ × ℝ)
  (hA : PointOnParabola p A.1 A.2)
  (hB : PointOnParabola p B.1 B.2)
  (hAB : A.2 - B.2 ≠ 0)  -- AB not perpendicular to x-axis
  (hM : PerpendicularBisector A B (4, 0)) :
  (A.1 + B.1) / 2 = 2 :=
sorry

/-- Setup for the specific problem -/
def problem_setup : Parabola :=
  { equation := fun x y => y^2 = 4*x
  , focus := (1, 0) }

#check midpoint_x_coordinate_constant problem_setup

end NUMINAMATH_CALUDE_midpoint_x_coordinate_constant_l40_4037


namespace NUMINAMATH_CALUDE_sum_not_arithmetic_l40_4051

/-- An infinite arithmetic progression -/
def arithmetic_progression (a d : ℝ) : ℕ → ℝ := λ n => a + (n - 1) * d

/-- An infinite geometric progression -/
def geometric_progression (b q : ℝ) : ℕ → ℝ := λ n => b * q^(n - 1)

/-- The sum of an arithmetic and a geometric progression -/
def sum_progression (a d b q : ℝ) : ℕ → ℝ :=
  λ n => arithmetic_progression a d n + geometric_progression b q n

theorem sum_not_arithmetic (a d b q : ℝ) (hq : q ≠ 1) :
  ¬ ∃ (A D : ℝ), ∀ n : ℕ, sum_progression a d b q n = A + (n - 1) * D :=
sorry

end NUMINAMATH_CALUDE_sum_not_arithmetic_l40_4051


namespace NUMINAMATH_CALUDE_line_l_equation_l40_4069

-- Define the ellipse E
def ellipse (t : ℝ) (x y : ℝ) : Prop := x^2 / t + y^2 = 1

-- Define the parabola C
def parabola (x y : ℝ) : Prop := x^2 = 2 * Real.sqrt 2 * y

-- Define the point H
def H : ℝ × ℝ := (2, 0)

-- Define the condition for line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 2)

-- Define the condition for tangent lines being perpendicular
def perpendicular_tangents (x₁ x₂ : ℝ) : Prop := 
  (Real.sqrt 2 / 2 * x₁) * (Real.sqrt 2 / 2 * x₂) = -1

theorem line_l_equation :
  ∃ (t k : ℝ) (A B M N : ℝ × ℝ),
    -- Conditions
    (ellipse t A.1 A.2) ∧
    (ellipse t B.1 B.2) ∧
    (parabola A.1 A.2) ∧
    (parabola B.1 B.2) ∧
    (parabola M.1 M.2) ∧
    (parabola N.1 N.2) ∧
    (line_l k M.1 M.2) ∧
    (line_l k N.1 N.2) ∧
    (perpendicular_tangents M.1 N.1) →
    -- Conclusion
    k = -Real.sqrt 2 / 4 ∧ 
    ∀ (x y : ℝ), line_l k x y ↔ x + 2 * Real.sqrt 2 * y - 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_line_l_equation_l40_4069


namespace NUMINAMATH_CALUDE_leo_current_weight_l40_4062

/-- Leo's current weight in pounds -/
def leo_weight : ℝ := 80

/-- Kendra's current weight in pounds -/
def kendra_weight : ℝ := 140 - leo_weight

/-- The combined weight of Leo and Kendra in pounds -/
def combined_weight : ℝ := 140

theorem leo_current_weight :
  (leo_weight + 10 = 1.5 * kendra_weight) ∧
  (leo_weight + kendra_weight = combined_weight) ∧
  (leo_weight = 80) :=
by sorry

end NUMINAMATH_CALUDE_leo_current_weight_l40_4062


namespace NUMINAMATH_CALUDE_missing_number_value_l40_4094

theorem missing_number_value (a b some_number : ℕ) : 
  a = 105 → 
  b = 147 → 
  a^3 = 21 * 25 * some_number * b → 
  some_number = 3 := by
sorry

end NUMINAMATH_CALUDE_missing_number_value_l40_4094


namespace NUMINAMATH_CALUDE_gcd_lcm_equality_implies_equal_l40_4070

theorem gcd_lcm_equality_implies_equal (a b c : ℕ+) :
  (Nat.gcd a b + Nat.lcm a b = Nat.gcd a c + Nat.lcm a c) → b = c := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_equality_implies_equal_l40_4070


namespace NUMINAMATH_CALUDE_equation_describes_cone_l40_4018

/-- Spherical coordinates -/
structure SphericalCoord where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- Definition of a cone in spherical coordinates -/
def IsCone (c : ℝ) (f : SphericalCoord → Prop) : Prop :=
  c > 0 ∧ ∀ p : SphericalCoord, f p ↔ p.ρ = c * Real.sin p.φ

/-- The main theorem: the equation ρ = c sin(φ) describes a cone -/
theorem equation_describes_cone (c : ℝ) :
  IsCone c (fun p => p.ρ = c * Real.sin p.φ) := by
  sorry


end NUMINAMATH_CALUDE_equation_describes_cone_l40_4018


namespace NUMINAMATH_CALUDE_quadratic_minimum_minimum_value_l40_4017

theorem quadratic_minimum (x y : ℝ) :
  let M := 4 * x^2 - 12 * x * y + 10 * y^2 + 4 * y + 9
  ∀ a b, M ≥ (4 * a^2 - 12 * a * b + 10 * b^2 + 4 * b + 9) → a = -3 ∧ b = -2 :=
by
  sorry

theorem minimum_value :
  ∃ x y, 4 * x^2 - 12 * x * y + 10 * y^2 + 4 * y + 9 = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_minimum_value_l40_4017


namespace NUMINAMATH_CALUDE_triple_angle_bracket_ten_l40_4009

def divisor_sum (n : ℕ) : ℕ :=
  sorry

def angle_bracket (n : ℕ) : ℕ :=
  sorry

theorem triple_angle_bracket_ten : angle_bracket (angle_bracket (angle_bracket 10)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_triple_angle_bracket_ten_l40_4009


namespace NUMINAMATH_CALUDE_smallest_number_l40_4030

theorem smallest_number (A B C : ℤ) : 
  A = 18 + 38 →
  B = A - 26 →
  C = B / 3 →
  C < A ∧ C < B :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l40_4030


namespace NUMINAMATH_CALUDE_range_of_a_for_p_range_of_a_for_p_or_q_and_not_p_and_q_l40_4007

-- Define proposition p
def p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0

-- Define proposition q
def q (a : ℝ) : Prop := ∃ x : ℝ, x ≥ 1 ∧ 4^x + 2^(x+1) - 7 - a < 0

-- Theorem for part 1
theorem range_of_a_for_p :
  {a : ℝ | p a} = {a : ℝ | 0 ≤ a ∧ a < 4} :=
sorry

-- Theorem for part 2
theorem range_of_a_for_p_or_q_and_not_p_and_q :
  {a : ℝ | (p a ∨ q a) ∧ ¬(p a ∧ q a)} = {a : ℝ | (0 ≤ a ∧ a ≤ 1) ∨ a ≥ 4} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_p_range_of_a_for_p_or_q_and_not_p_and_q_l40_4007


namespace NUMINAMATH_CALUDE_fib_100_div_5_l40_4001

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- The Fibonacci sequence modulo 5 repeats every 20 terms -/
axiom fib_mod_5_period : ∀ n : ℕ, fib (n + 20) % 5 = fib n % 5

theorem fib_100_div_5 : 5 ∣ fib 100 := by
  sorry

end NUMINAMATH_CALUDE_fib_100_div_5_l40_4001


namespace NUMINAMATH_CALUDE_pigeon_problem_solution_l40_4032

/-- Represents the number of pigeons on the branches and under the tree -/
structure PigeonCount where
  onBranches : ℕ
  underTree : ℕ

/-- The conditions of the pigeon problem -/
def satisfiesPigeonConditions (p : PigeonCount) : Prop :=
  (p.underTree - 1 = (p.onBranches + 1) / 3) ∧
  (p.onBranches - 1 = p.underTree + 1)

/-- The theorem stating the solution to the pigeon problem -/
theorem pigeon_problem_solution :
  ∃ (p : PigeonCount), satisfiesPigeonConditions p ∧ p.onBranches = 7 ∧ p.underTree = 5 := by
  sorry


end NUMINAMATH_CALUDE_pigeon_problem_solution_l40_4032


namespace NUMINAMATH_CALUDE_no_integer_solution_l40_4040

theorem no_integer_solution : ¬ ∃ (a b c d : ℤ),
  (a * 19^3 + b * 19^2 + c * 19 + d = 1) ∧
  (a * 62^3 + b * 62^2 + c * 62 + d = 2) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l40_4040


namespace NUMINAMATH_CALUDE_system_solution_condition_l40_4005

-- Define the system of equations
def equation1 (x y a : ℝ) : Prop := Real.arccos ((4 + y) / 4) = Real.arccos (x - a)
def equation2 (x y b : ℝ) : Prop := x^2 + y^2 - 4*x + 8*y = b

-- Define the condition for no more than one solution
def atMostOneSolution (a : ℝ) : Prop :=
  ∀ b : ℝ, ∃! (x y : ℝ), equation1 x y a ∧ equation2 x y b

-- Theorem statement
theorem system_solution_condition (a : ℝ) :
  atMostOneSolution a ↔ a ≤ -15 ∨ a ≥ 19 := by sorry

end NUMINAMATH_CALUDE_system_solution_condition_l40_4005


namespace NUMINAMATH_CALUDE_plane_speed_with_wind_l40_4029

theorem plane_speed_with_wind (distance : ℝ) (wind_speed : ℝ) (time_with_wind : ℝ) (time_against_wind : ℝ) :
  wind_speed = 24 ∧ time_with_wind = 5.5 ∧ time_against_wind = 6 →
  ∃ (plane_speed : ℝ),
    distance / time_with_wind = plane_speed + wind_speed ∧
    distance / time_against_wind = plane_speed - wind_speed ∧
    plane_speed + wind_speed = 576 ∧
    plane_speed - wind_speed = 528 := by
  sorry

end NUMINAMATH_CALUDE_plane_speed_with_wind_l40_4029


namespace NUMINAMATH_CALUDE_smallest_base_for_100_l40_4021

theorem smallest_base_for_100 : 
  ∃ (b : ℕ), b = 5 ∧ 
  (∀ (x : ℕ), x < b → ¬(x^2 ≤ 100 ∧ 100 < x^3)) ∧
  (5^2 ≤ 100 ∧ 100 < 5^3) := by
  sorry

end NUMINAMATH_CALUDE_smallest_base_for_100_l40_4021


namespace NUMINAMATH_CALUDE_dc_length_l40_4076

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the conditions
def conditions (q : Quadrilateral) : Prop :=
  let dist := λ p₁ p₂ : ℝ × ℝ => Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)
  let sinAngle := λ p₁ p₂ p₃ : ℝ × ℝ => 
    let v1 := (p₂.1 - p₁.1, p₂.2 - p₁.2)
    let v2 := (p₃.1 - p₁.1, p₃.2 - p₁.2)
    (v1.1 * v2.2 - v1.2 * v2.1) / (dist p₁ p₂ * dist p₁ p₃)
  dist q.A q.B = 30 ∧
  (q.A.1 - q.D.1) * (q.B.1 - q.D.1) + (q.A.2 - q.D.2) * (q.B.2 - q.D.2) = 0 ∧
  sinAngle q.B q.A q.D = 4/5 ∧
  sinAngle q.B q.C q.D = 1/5

-- State the theorem
theorem dc_length (q : Quadrilateral) (h : conditions q) : 
  dist q.D q.C = 48 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_dc_length_l40_4076


namespace NUMINAMATH_CALUDE_banana_cream_pie_degrees_is_44_l40_4064

/-- The number of degrees in a pie chart slice for banana cream pie preference --/
def banana_cream_pie_degrees (total_students : ℕ) 
                              (strawberry_pref : ℕ) 
                              (pecan_pref : ℕ) 
                              (pumpkin_pref : ℕ) : ℚ :=
  let remaining_students := total_students - (strawberry_pref + pecan_pref + pumpkin_pref)
  let banana_cream_pref := remaining_students / 2
  (banana_cream_pref / total_students) * 360

/-- Theorem stating that the number of degrees for banana cream pie preference is 44 --/
theorem banana_cream_pie_degrees_is_44 : 
  banana_cream_pie_degrees 45 15 10 9 = 44 := by
  sorry

end NUMINAMATH_CALUDE_banana_cream_pie_degrees_is_44_l40_4064


namespace NUMINAMATH_CALUDE_remainder_theorem_l40_4048

-- Define the polynomial q(x)
def q (A B C x : ℝ) : ℝ := A * x^5 - B * x^3 + C * x - 2

-- Theorem statement
theorem remainder_theorem (A B C : ℝ) :
  (q A B C 2 = -6) → (q A B C (-2) = 2) := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l40_4048


namespace NUMINAMATH_CALUDE_circle_radius_from_area_l40_4050

theorem circle_radius_from_area (r : ℝ) : r > 0 → π * r^2 = 9 * π → r = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_area_l40_4050


namespace NUMINAMATH_CALUDE_triangle_theorem_l40_4054

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.a^2 + t.b^2 - t.c^2 = Real.sqrt 3 * t.a * t.b) 
  (h2 : 0 < t.A ∧ t.A ≤ 2 * Real.pi / 3) :
  t.C = Real.pi / 6 ∧ 
  let m := 2 * (Real.cos (t.A / 2))^2 - Real.sin t.B - 1
  ∀ x, m = x → -1 ≤ x ∧ x < 1/2 := by
sorry

end NUMINAMATH_CALUDE_triangle_theorem_l40_4054


namespace NUMINAMATH_CALUDE_square_sum_from_means_l40_4042

theorem square_sum_from_means (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 20) 
  (h_geometric : Real.sqrt (x * y) = Real.sqrt 110) : 
  x^2 + y^2 = 1380 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_from_means_l40_4042


namespace NUMINAMATH_CALUDE_parabola_directrix_equation_l40_4092

/-- The equation of the directrix of a parabola passing through a point on a circle -/
theorem parabola_directrix_equation (y : ℝ) (p : ℝ) : 
  (1^2 - 4*1 + y^2 = 0) →  -- Point P(1, y) lies on the circle
  (p > 0) →                -- p is positive
  (1^2 = -2*p*y) →         -- Parabola passes through P(1, y)
  (-(p : ℝ) = Real.sqrt 3 / 12) := by
  sorry

end NUMINAMATH_CALUDE_parabola_directrix_equation_l40_4092


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l40_4073

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 ≠ x) ↔ (∃ x : ℝ, x^2 = x) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l40_4073


namespace NUMINAMATH_CALUDE_smallest_angle_proof_l40_4035

def AP : ℝ := 2

noncomputable def smallest_angle (x : ℝ) : ℝ :=
  Real.arctan (Real.sqrt 2 / 4)

theorem smallest_angle_proof (x : ℝ) : 
  smallest_angle x = Real.arctan (Real.sqrt 2 / 4) :=
sorry

end NUMINAMATH_CALUDE_smallest_angle_proof_l40_4035


namespace NUMINAMATH_CALUDE_restaurant_menu_combinations_l40_4004

theorem restaurant_menu_combinations (menu_size : ℕ) (yann_order camille_order : ℕ) : 
  menu_size = 12 →
  yann_order ≠ camille_order →
  yann_order ≤ menu_size ∧ camille_order ≤ menu_size →
  (menu_size * (menu_size - 1) : ℕ) = 132 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_menu_combinations_l40_4004


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l40_4066

/-- Given two real numbers x and y that are inversely proportional,
    prove that if x + y = 20 and x - y = 4, then y = 24 when x = 4. -/
theorem inverse_proportion_problem (x y : ℝ) (h1 : ∃ k : ℝ, x * y = k) 
    (h2 : x + y = 20) (h3 : x - y = 4) : 
    x = 4 → y = 24 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l40_4066


namespace NUMINAMATH_CALUDE_distinct_values_of_combination_sum_l40_4072

theorem distinct_values_of_combination_sum :
  ∃ (S : Finset ℕ), 
    (∀ r : ℕ, r + 1 ≤ 10 ∧ 17 - r ≤ 10 → 
      (Nat.choose 10 (r + 1) + Nat.choose 10 (17 - r)) ∈ S) ∧
    Finset.card S = 2 :=
sorry

end NUMINAMATH_CALUDE_distinct_values_of_combination_sum_l40_4072


namespace NUMINAMATH_CALUDE_largest_n_sin_cos_inequality_l40_4098

theorem largest_n_sin_cos_inequality :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (x : ℝ), (Real.sin x)^n + (Real.cos x)^n ≥ 1 / (2 * Real.sqrt (n : ℝ))) ∧
  (∀ (m : ℕ), m > n → ∃ (y : ℝ), (Real.sin y)^m + (Real.cos y)^m < 1 / (2 * Real.sqrt (m : ℝ))) ∧
  n = 2 := by
  sorry

end NUMINAMATH_CALUDE_largest_n_sin_cos_inequality_l40_4098


namespace NUMINAMATH_CALUDE_earth_inhabitable_fraction_l40_4022

theorem earth_inhabitable_fraction :
  let water_fraction : ℚ := 2/3
  let land_fraction : ℚ := 1 - water_fraction
  let inhabitable_land_fraction : ℚ := 1/3
  (1 - water_fraction) * inhabitable_land_fraction = 1/9 :=
by
  sorry

end NUMINAMATH_CALUDE_earth_inhabitable_fraction_l40_4022


namespace NUMINAMATH_CALUDE_foreign_student_percentage_l40_4079

theorem foreign_student_percentage 
  (total_students : ℕ) 
  (new_foreign_students : ℕ) 
  (future_foreign_students : ℕ) :
  total_students = 1800 →
  new_foreign_students = 200 →
  future_foreign_students = 740 →
  (↑(future_foreign_students - new_foreign_students) / ↑total_students : ℚ) = 30 / 100 := by
sorry

end NUMINAMATH_CALUDE_foreign_student_percentage_l40_4079


namespace NUMINAMATH_CALUDE_removed_to_total_ratio_is_one_to_two_l40_4012

/-- Represents the number of bricks in a course -/
def bricks_per_course : ℕ := 400

/-- Represents the initial number of courses -/
def initial_courses : ℕ := 3

/-- Represents the number of courses added -/
def added_courses : ℕ := 2

/-- Represents the total number of bricks after removal -/
def total_bricks_after_removal : ℕ := 1800

/-- Theorem stating that the ratio of removed bricks to total bricks in the last course is 1:2 -/
theorem removed_to_total_ratio_is_one_to_two :
  let total_courses := initial_courses + added_courses
  let expected_total_bricks := total_courses * bricks_per_course
  let removed_bricks := expected_total_bricks - total_bricks_after_removal
  let last_course_bricks := bricks_per_course
  (removed_bricks : ℚ) / (last_course_bricks : ℚ) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_removed_to_total_ratio_is_one_to_two_l40_4012


namespace NUMINAMATH_CALUDE_multiply_by_six_l40_4060

theorem multiply_by_six (x : ℚ) (h : x / 11 = 2) : 6 * x = 132 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_six_l40_4060


namespace NUMINAMATH_CALUDE_snickers_count_l40_4046

theorem snickers_count (total : ℕ) (mars : ℕ) (butterfingers : ℕ) (snickers : ℕ) : 
  total = 12 → mars = 2 → butterfingers = 7 → total = mars + butterfingers + snickers → snickers = 3 := by
  sorry

end NUMINAMATH_CALUDE_snickers_count_l40_4046


namespace NUMINAMATH_CALUDE_mall_meal_pairs_l40_4056

/-- The number of distinct pairs of meals for two people, given the number of options for each meal component. -/
def distinct_meal_pairs (num_entrees num_drinks num_desserts : ℕ) : ℕ :=
  let total_meals := num_entrees * num_drinks * num_desserts
  total_meals * (total_meals - 1)

/-- Theorem stating that the number of distinct meal pairs is 1260 given the specific options. -/
theorem mall_meal_pairs :
  distinct_meal_pairs 4 3 3 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_mall_meal_pairs_l40_4056


namespace NUMINAMATH_CALUDE_student_average_age_l40_4081

theorem student_average_age (n : ℕ) (teacher_age : ℕ) (new_average : ℝ) :
  n = 50 ∧ teacher_age = 65 ∧ new_average = 15 →
  (n : ℝ) * (((n : ℝ) * new_average - teacher_age) / n) + teacher_age = (n + 1 : ℝ) * new_average →
  ((n : ℝ) * new_average - teacher_age) / n = 14 := by
  sorry

end NUMINAMATH_CALUDE_student_average_age_l40_4081


namespace NUMINAMATH_CALUDE_parallel_planes_normal_vectors_l40_4074

/-- Given two vectors that are normal vectors of parallel planes, prove that their specific components multiply to -3 -/
theorem parallel_planes_normal_vectors (m n : ℝ) : 
  let a : Fin 3 → ℝ := ![0, 1, m]
  let b : Fin 3 → ℝ := ![0, n, -3]
  (∃ (k : ℝ), a = k • b) →  -- Parallel planes condition
  m * n = -3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_planes_normal_vectors_l40_4074


namespace NUMINAMATH_CALUDE_calculate_expression_l40_4053

theorem calculate_expression : 24 / (-6) * (3/2) / (-4/3) = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l40_4053


namespace NUMINAMATH_CALUDE_multiplication_problem_solution_l40_4088

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the multiplication problem -/
structure MultiplicationProblem where
  A : Digit
  B : Digit
  C : Digit
  D : Digit
  distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D
  equation : A.val * 100 + B.val * 10 + C.val = C.val * 100 + C.val * 10 + A.val

theorem multiplication_problem_solution (p : MultiplicationProblem) : p.A.val + p.C.val = 10 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_problem_solution_l40_4088


namespace NUMINAMATH_CALUDE_bug_visits_24_tiles_l40_4086

/-- The number of tiles a bug visits when walking diagonally across a rectangular floor -/
def tilesVisited (width length : ℕ) : ℕ :=
  width + length - Nat.gcd width length

/-- The floor dimensions -/
def floorWidth : ℕ := 12
def floorLength : ℕ := 18

/-- Theorem: A bug walking diagonally across the given rectangular floor visits 24 tiles -/
theorem bug_visits_24_tiles :
  tilesVisited floorWidth floorLength = 24 := by
  sorry


end NUMINAMATH_CALUDE_bug_visits_24_tiles_l40_4086


namespace NUMINAMATH_CALUDE_arithmetic_progression_equality_l40_4075

theorem arithmetic_progression_equality (n : ℕ) 
  (a b : Fin n → ℕ+) 
  (h_n : n ≥ 2018)
  (h_distinct : ∀ i j : Fin n, i ≠ j → (a i ≠ a j ∧ b i ≠ b j))
  (h_bound : ∀ i : Fin n, a i ≤ 5*n ∧ b i ≤ 5*n)
  (h_arithmetic : ∃ d : ℚ, ∀ i j : Fin n, (a j : ℚ) / (b j : ℚ) - (a i : ℚ) / (b i : ℚ) = (j : ℚ) - (i : ℚ) * d) :
  ∀ i j : Fin n, (a i : ℚ) / (b i : ℚ) = (a j : ℚ) / (b j : ℚ) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_equality_l40_4075


namespace NUMINAMATH_CALUDE_bus_distance_ratio_l40_4095

theorem bus_distance_ratio (total_distance : ℝ) (foot_fraction : ℝ) (car_distance : ℝ) :
  total_distance = 40 →
  foot_fraction = 1 / 4 →
  car_distance = 10 →
  (total_distance - (foot_fraction * total_distance + car_distance)) / total_distance = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_bus_distance_ratio_l40_4095


namespace NUMINAMATH_CALUDE_elizabeth_revenue_is_900_l40_4011

/-- Represents the revenue and investment data for Mr. Banks and Ms. Elizabeth -/
structure InvestmentData where
  banks_investments : ℕ
  banks_revenue_per_investment : ℕ
  elizabeth_investments : ℕ
  elizabeth_total_revenue_difference : ℕ

/-- Calculates Ms. Elizabeth's revenue per investment given the investment data -/
def elizabeth_revenue_per_investment (data : InvestmentData) : ℕ :=
  (data.banks_investments * data.banks_revenue_per_investment + data.elizabeth_total_revenue_difference) / data.elizabeth_investments

/-- Theorem stating that Ms. Elizabeth's revenue per investment is $900 given the problem conditions -/
theorem elizabeth_revenue_is_900 (data : InvestmentData)
  (h1 : data.banks_investments = 8)
  (h2 : data.banks_revenue_per_investment = 500)
  (h3 : data.elizabeth_investments = 5)
  (h4 : data.elizabeth_total_revenue_difference = 500) :
  elizabeth_revenue_per_investment data = 900 := by
  sorry

end NUMINAMATH_CALUDE_elizabeth_revenue_is_900_l40_4011


namespace NUMINAMATH_CALUDE_megans_earnings_l40_4099

/-- Calculates the total earnings for a given number of months based on daily work hours, hourly rate, and days worked per month. -/
def total_earnings (hours_per_day : ℕ) (hourly_rate : ℚ) (days_per_month : ℕ) (months : ℕ) : ℚ :=
  hours_per_day * hourly_rate * days_per_month * months

/-- Proves that Megan's total earnings for two months of work is $2400. -/
theorem megans_earnings :
  total_earnings 8 (15/2) 20 2 = 2400 := by
  sorry

#eval total_earnings 8 (15/2) 20 2

end NUMINAMATH_CALUDE_megans_earnings_l40_4099


namespace NUMINAMATH_CALUDE_suzannes_book_pages_l40_4033

theorem suzannes_book_pages : 
  ∀ (pages_monday pages_tuesday pages_left : ℕ),
    pages_monday = 15 →
    pages_tuesday = pages_monday + 16 →
    pages_left = 18 →
    pages_monday + pages_tuesday + pages_left = 64 :=
by
  sorry

end NUMINAMATH_CALUDE_suzannes_book_pages_l40_4033


namespace NUMINAMATH_CALUDE_factorization_proof_l40_4020

theorem factorization_proof (a b : ℝ) : 4 * a^3 * b - a * b = a * b * (2 * a + 1) * (2 * a - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l40_4020


namespace NUMINAMATH_CALUDE_intercept_ratio_l40_4036

/-- Given two lines with the same y-intercept (0, b) where b ≠ 0,
    if the first line has slope 12 and x-intercept (s, 0),
    and the second line has slope 8 and x-intercept (t, 0),
    then s/t = 2/3 -/
theorem intercept_ratio (b s t : ℝ) (hb : b ≠ 0)
  (h1 : 0 = 12 * s + b) (h2 : 0 = 8 * t + b) : s / t = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_intercept_ratio_l40_4036


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_condition_l40_4023

/-- 
Given a quadratic equation (k-1)x^2 + 4x + 1 = 0, this theorem states that
for the equation to have two distinct real roots, k must be less than 5 and not equal to 1.
-/
theorem quadratic_distinct_roots_condition (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (k - 1) * x₁^2 + 4 * x₁ + 1 = 0 ∧ 
    (k - 1) * x₂^2 + 4 * x₂ + 1 = 0) ↔ 
  (k < 5 ∧ k ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_condition_l40_4023


namespace NUMINAMATH_CALUDE_cost_at_two_l40_4085

/-- The cost function for a product -/
def cost (q : ℝ) : ℝ := q^3 + q - 1

/-- Theorem: The cost is 9 when the quantity is 2 -/
theorem cost_at_two : cost 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_cost_at_two_l40_4085


namespace NUMINAMATH_CALUDE_vowel_initial_probability_theorem_l40_4063

/-- The number of students in the class -/
def total_students : ℕ := 26

/-- The number of vowels (including 'Y') -/
def vowel_count : ℕ := 6

/-- The probability of selecting a student with vowel initials -/
def vowel_initial_probability : ℚ := 3 / 13

/-- Theorem stating the probability of selecting a student with vowel initials -/
theorem vowel_initial_probability_theorem :
  (vowel_count : ℚ) / total_students = vowel_initial_probability := by
  sorry

end NUMINAMATH_CALUDE_vowel_initial_probability_theorem_l40_4063


namespace NUMINAMATH_CALUDE_quadratic_polynomial_from_sum_and_product_l40_4065

theorem quadratic_polynomial_from_sum_and_product (x y : ℝ) 
  (sum_condition : x + y = 15) 
  (product_condition : x * y = 36) : 
  (fun z : ℝ => z^2 - 15*z + 36) = (fun z : ℝ => (z - x) * (z - y)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_from_sum_and_product_l40_4065
