import Mathlib

namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1621_162195

/-- Two vectors are parallel if their components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Given vectors a and b, prove that if they are parallel, then x = 6 -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (4, 2)
  let b : ℝ × ℝ := (x, 3)
  parallel a b → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1621_162195


namespace NUMINAMATH_CALUDE_equation_solution_l1621_162128

theorem equation_solution : ∃! x : ℚ, x + 2/5 = 8/15 + 1/3 ∧ x = 7/15 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1621_162128


namespace NUMINAMATH_CALUDE_binomial_12_choose_3_l1621_162142

theorem binomial_12_choose_3 : Nat.choose 12 3 = 220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_choose_3_l1621_162142


namespace NUMINAMATH_CALUDE_omega_value_l1621_162112

/-- Given a function f(x) = sin(ωx) + cos(ωx) where ω > 0 and x ∈ ℝ,
    if f(x) is monotonically increasing on (-ω, ω) and
    the graph of y = f(x) is symmetric with respect to x = ω,
    then ω = √π / 2 -/
theorem omega_value (ω : ℝ) (h_pos : ω > 0) :
  let f : ℝ → ℝ := λ x ↦ Real.sin (ω * x) + Real.cos (ω * x)
  (∀ x ∈ Set.Ioo (-ω) ω, Monotone f) →
  (∀ x : ℝ, f (ω + x) = f (ω - x)) →
  ω = Real.sqrt π / 2 := by
  sorry

end NUMINAMATH_CALUDE_omega_value_l1621_162112


namespace NUMINAMATH_CALUDE_intersection_when_a_is_two_proper_subset_condition_l1621_162184

def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2 * a + 3}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 4}

theorem intersection_when_a_is_two :
  A 2 ∩ B = {x | 1 < x ∧ x ≤ 4} := by sorry

theorem proper_subset_condition (a : ℝ) :
  A a ⊂ B ↔ a ≤ -4 ∨ (-1 ≤ a ∧ a ≤ 1/2) := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_two_proper_subset_condition_l1621_162184


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l1621_162183

/-- The distance between the vertices of a hyperbola with equation y²/48 - x²/16 = 1 is 8√3 -/
theorem hyperbola_vertex_distance :
  let hyperbola := {(x, y) : ℝ × ℝ | y^2 / 48 - x^2 / 16 = 1}
  ∃ v₁ v₂ : ℝ × ℝ, v₁ ∈ hyperbola ∧ v₂ ∈ hyperbola ∧ 
    ∀ p ∈ hyperbola, (p.1 = v₁.1 ∨ p.1 = v₂.1) → p.2 = 0 ∧
    ‖v₁ - v₂‖ = 8 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l1621_162183


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1621_162137

def arithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℚ) 
  (h1 : arithmeticSequence a)
  (h2 : a 2 + a 9 + a 12 - a 14 + a 20 - a 7 = 8) :
  a 9 - (1/4) * a 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1621_162137


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1621_162104

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_geom : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h_a4a5 : a 4 * a 5 = 1) 
  (h_a8a9 : a 8 * a 9 = 16) : 
  q = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1621_162104


namespace NUMINAMATH_CALUDE_liquid_rise_ratio_l1621_162159

-- Define the cones and their properties
structure Cone where
  radius : ℝ
  height : ℝ
  volume : ℝ

-- Define the marble
def marbleRadius : ℝ := 1

-- Define the cones
def narrowCone : Cone := { radius := 3, height := 0, volume := 0 }
def wideCone : Cone := { radius := 6, height := 0, volume := 0 }

-- State that both cones contain the same amount of liquid
axiom equal_volume : narrowCone.volume = wideCone.volume

-- Define the rise in liquid level after dropping the marble
def liquidRise (c : Cone) : ℝ := sorry

-- Theorem to prove
theorem liquid_rise_ratio :
  liquidRise narrowCone / liquidRise wideCone = 4 := by sorry

end NUMINAMATH_CALUDE_liquid_rise_ratio_l1621_162159


namespace NUMINAMATH_CALUDE_distance_sum_between_19_and_20_l1621_162117

/-- Given points A, B, and D in a coordinate plane, prove that the sum of distances AD and BD is between 19 and 20 -/
theorem distance_sum_between_19_and_20 (A B D : ℝ × ℝ) : 
  A = (15, 0) → B = (0, 0) → D = (8, 6) → 
  19 < Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) + Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) ∧
  Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) + Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) < 20 :=
by
  sorry

end NUMINAMATH_CALUDE_distance_sum_between_19_and_20_l1621_162117


namespace NUMINAMATH_CALUDE_percentage_written_second_week_l1621_162123

/-- Proves that the percentage of remaining pages written in the second week is 30% --/
theorem percentage_written_second_week :
  ∀ (total_pages : ℕ) 
    (first_week_pages : ℕ) 
    (damaged_percentage : ℚ) 
    (final_empty_pages : ℕ),
  total_pages = 500 →
  first_week_pages = 150 →
  damaged_percentage = 20 / 100 →
  final_empty_pages = 196 →
  ∃ (second_week_percentage : ℚ),
    second_week_percentage = 30 / 100 ∧
    final_empty_pages = 
      (1 - damaged_percentage) * 
      (total_pages - first_week_pages - 
       (second_week_percentage * (total_pages - first_week_pages))) :=
by sorry

end NUMINAMATH_CALUDE_percentage_written_second_week_l1621_162123


namespace NUMINAMATH_CALUDE_locus_of_point_M_l1621_162134

/-- The locus of points M(x,y) forming triangles with fixed points A(-1,0) and B(1,0),
    where the sum of slopes of AM and BM is 2. -/
theorem locus_of_point_M (x y : ℝ) (h : x ≠ 1 ∧ x ≠ -1) :
  (y / (x + 1) + y / (x - 1) = 2) → (x^2 - x*y - 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_locus_of_point_M_l1621_162134


namespace NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l1621_162169

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n - 1)

theorem fourth_term_of_geometric_sequence :
  let a₁ : ℝ := 6
  let a₈ : ℝ := 186624
  let r : ℝ := (a₈ / a₁) ^ (1 / 7)
  geometric_sequence a₁ r 4 = 1296 := by
sorry

end NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l1621_162169


namespace NUMINAMATH_CALUDE_upstream_downstream_time_ratio_l1621_162126

/-- Proves that the ratio of the time taken to row upstream to the time taken to row downstream is 2:1 -/
theorem upstream_downstream_time_ratio 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (h1 : boat_speed = 78) 
  (h2 : stream_speed = 26) : 
  (boat_speed - stream_speed) / (boat_speed + stream_speed) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_upstream_downstream_time_ratio_l1621_162126


namespace NUMINAMATH_CALUDE_tulip_price_is_two_l1621_162163

/-- Represents the price of a tulip in dollars -/
def tulip_price : ℝ := 2

/-- Represents the price of a rose in dollars -/
def rose_price : ℝ := 3

/-- Calculates the total revenue for the three days -/
def total_revenue (tulip_price : ℝ) : ℝ :=
  -- First day
  (30 * tulip_price + 20 * rose_price) +
  -- Second day
  (60 * tulip_price + 40 * rose_price) +
  -- Third day
  (6 * tulip_price + 16 * rose_price)

theorem tulip_price_is_two :
  total_revenue tulip_price = 420 :=
by sorry

end NUMINAMATH_CALUDE_tulip_price_is_two_l1621_162163


namespace NUMINAMATH_CALUDE_monotonicity_and_extrema_l1621_162103

noncomputable def f (a x : ℝ) : ℝ := a * x^2 + (2*a - 1) * x - Real.log x

theorem monotonicity_and_extrema (a : ℝ) :
  (∀ x, x > 0 → f a x = a * x^2 + (2*a - 1) * x - Real.log x) →
  (a = 1/2 →
    (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f a x₁ > f a x₂) ∧
    (∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂) ∧
    (∀ x, x > 0 → f a x ≥ 1/2) ∧
    f a 1 = 1/2) ∧
  (a ≤ 0 →
    ∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ → f a x₁ > f a x₂) ∧
  (a > 0 →
    (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1/(2*a) → f a x₁ > f a x₂) ∧
    (∀ x₁ x₂, 1/(2*a) < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂)) :=
by sorry

end NUMINAMATH_CALUDE_monotonicity_and_extrema_l1621_162103


namespace NUMINAMATH_CALUDE_evaluate_expression_l1621_162161

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 6560 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1621_162161


namespace NUMINAMATH_CALUDE_answer_A_first_is_better_l1621_162182

-- Define the probabilities and point values
def prob_A : ℝ := 0.7
def prob_B : ℝ := 0.5
def points_A : ℝ := 40
def points_B : ℝ := 60

-- Define the expected score when answering A first
def E_A : ℝ := (1 - prob_A) * 0 + prob_A * (1 - prob_B) * points_A + prob_A * prob_B * (points_A + points_B)

-- Define the expected score when answering B first
def E_B : ℝ := (1 - prob_B) * 0 + prob_B * (1 - prob_A) * points_B + prob_B * prob_A * (points_A + points_B)

-- Theorem: Answering A first yields a higher expected score
theorem answer_A_first_is_better : E_A > E_B := by
  sorry

end NUMINAMATH_CALUDE_answer_A_first_is_better_l1621_162182


namespace NUMINAMATH_CALUDE_no_three_subset_partition_of_positive_integers_l1621_162151

theorem no_three_subset_partition_of_positive_integers :
  ¬ ∃ (A B C : Set ℕ),
    (A ∪ B ∪ C = {n : ℕ | n > 0}) ∧
    (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (C ∩ A = ∅) ∧
    (A ≠ ∅) ∧ (B ≠ ∅) ∧ (C ≠ ∅) ∧
    (∀ x y : ℕ, x > 0 → y > 0 →
      ((x ∈ A ∧ y ∈ B) ∨ (x ∈ B ∧ y ∈ A) → x^2 - x*y + y^2 ∈ C) ∧
      ((x ∈ B ∧ y ∈ C) ∨ (x ∈ C ∧ y ∈ B) → x^2 - x*y + y^2 ∈ A) ∧
      ((x ∈ C ∧ y ∈ A) ∨ (x ∈ A ∧ y ∈ C) → x^2 - x*y + y^2 ∈ B)) :=
sorry

end NUMINAMATH_CALUDE_no_three_subset_partition_of_positive_integers_l1621_162151


namespace NUMINAMATH_CALUDE_average_weight_problem_l1621_162166

/-- Given the average weight of three people and some additional information,
    prove that the average weight of two of them is 43 kg. -/
theorem average_weight_problem (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →   -- average weight of a, b, and c
  (a + b) / 2 = 40 →       -- average weight of a and b
  b = 31 →                 -- weight of b
  (b + c) / 2 = 43         -- average weight of b and c to be proved
  := by sorry

end NUMINAMATH_CALUDE_average_weight_problem_l1621_162166


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1621_162113

-- Define the inequality system
def inequality_system (x : ℝ) : Prop :=
  x + 1 < 4 ∧ 1 - 3*x ≥ -5

-- Define the solution set
def solution_set : Set ℝ :=
  {x | x ≤ 2}

-- Theorem statement
theorem inequality_system_solution :
  {x : ℝ | inequality_system x} = solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1621_162113


namespace NUMINAMATH_CALUDE_bob_corn_rows_l1621_162196

/-- Represents the number of corn stalks in each row -/
def stalks_per_row : ℕ := 80

/-- Represents the number of corn stalks needed to produce one bushel -/
def stalks_per_bushel : ℕ := 8

/-- Represents the total number of bushels Bob will harvest -/
def total_bushels : ℕ := 50

/-- Calculates the number of rows of corn Bob has -/
def number_of_rows : ℕ := (total_bushels * stalks_per_bushel) / stalks_per_row

theorem bob_corn_rows :
  number_of_rows = 5 :=
sorry

end NUMINAMATH_CALUDE_bob_corn_rows_l1621_162196


namespace NUMINAMATH_CALUDE_classroom_addition_problem_l1621_162110

theorem classroom_addition_problem (x y : ℕ) (h1 : x > y) (h2 : x - y = 6) (h3 : x * y = 45) : 
  x = 11 ∧ y = 5 := by
sorry

end NUMINAMATH_CALUDE_classroom_addition_problem_l1621_162110


namespace NUMINAMATH_CALUDE_largest_n_satisfying_equation_l1621_162187

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating that 397 is the largest positive integer satisfying the equation -/
theorem largest_n_satisfying_equation :
  ∀ n : ℕ, n > 0 → n = (sum_of_digits n)^2 + 2*(sum_of_digits n) - 2 → n ≤ 397 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_equation_l1621_162187


namespace NUMINAMATH_CALUDE_coal_cost_equilibrium_point_verify_equilibrium_point_l1621_162157

/-- Represents the cost of coal at a point on the line segment AB -/
def coal_cost (x : ℝ) (from_a : Bool) : ℝ :=
  if from_a then
    3.75 + 0.008 * x
  else
    4.25 + 0.008 * (225 - x)

/-- Theorem stating the existence and uniqueness of point C -/
theorem coal_cost_equilibrium_point :
  ∃! x : ℝ, 0 ≤ x ∧ x ≤ 225 ∧
    coal_cost x true = coal_cost x false ∧
    ∀ y : ℝ, 0 ≤ y ∧ y ≤ 225 → coal_cost y true ≤ coal_cost x true ∧ coal_cost y false ≤ coal_cost x false :=
by
  sorry

/-- The actual equilibrium point -/
def equilibrium_point : ℝ := 143.75

/-- The cost of coal at the equilibrium point -/
def equilibrium_cost : ℝ := 4.90

/-- Theorem verifying the equilibrium point and cost -/
theorem verify_equilibrium_point :
  coal_cost equilibrium_point true = equilibrium_cost ∧
  coal_cost equilibrium_point false = equilibrium_cost :=
by
  sorry

end NUMINAMATH_CALUDE_coal_cost_equilibrium_point_verify_equilibrium_point_l1621_162157


namespace NUMINAMATH_CALUDE_interest_rate_proof_l1621_162119

/-- The rate of interest given specific investment conditions -/
theorem interest_rate_proof (principal : ℝ) (time : ℝ) (simple_interest : ℝ) (compound_interest : ℝ) 
  (h1 : principal = 4000)
  (h2 : time = 2)
  (h3 : simple_interest = 400)
  (h4 : compound_interest = 410) :
  ∃ (rate : ℝ), 
    rate = 5 ∧
    simple_interest = (principal * rate * time) / 100 ∧
    compound_interest = principal * ((1 + rate / 100) ^ time - 1) := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_proof_l1621_162119


namespace NUMINAMATH_CALUDE_odd_function_sum_l1621_162140

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_sum (f : ℝ → ℝ) (h_odd : is_odd_function f) (h_domain : ∀ x ∈ [-3, 3], f x = f x) (h_value : f 3 = -2) :
  f (-3) + f 0 = 2 :=
by sorry

end NUMINAMATH_CALUDE_odd_function_sum_l1621_162140


namespace NUMINAMATH_CALUDE_roof_dimension_difference_l1621_162197

/-- Represents the dimensions of a rectangular base pyramid roof -/
structure RoofDimensions where
  width : ℝ
  length : ℝ
  height : ℝ
  area : ℝ

/-- Conditions for the roof dimensions -/
def roof_conditions (r : RoofDimensions) : Prop :=
  r.length = 4 * r.width ∧
  r.area = 1024 ∧
  r.height = 50 ∧
  r.area = r.length * r.width

/-- Theorem stating the difference between length and width -/
theorem roof_dimension_difference (r : RoofDimensions) 
  (h : roof_conditions r) : r.length - r.width = 48 := by
  sorry

#check roof_dimension_difference

end NUMINAMATH_CALUDE_roof_dimension_difference_l1621_162197


namespace NUMINAMATH_CALUDE_perfect_square_pairs_l1621_162158

theorem perfect_square_pairs (x y : ℕ) :
  (∃ a : ℕ, x^2 + 8*y = a^2) ∧ (∃ b : ℕ, y^2 - 8*x = b^2) →
  (∃ n : ℕ, x = n ∧ y = n + 2) ∨
  ((x = 7 ∧ y = 15) ∨ (x = 33 ∧ y = 17) ∨ (x = 45 ∧ y = 23)) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_pairs_l1621_162158


namespace NUMINAMATH_CALUDE_power_sum_equals_two_l1621_162191

theorem power_sum_equals_two : (1 : ℤ)^10 + (-1 : ℤ)^8 + (-1 : ℤ)^7 + (1 : ℤ)^5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equals_two_l1621_162191


namespace NUMINAMATH_CALUDE_range_of_t_l1621_162139

theorem range_of_t (x t : ℝ) : 
  (∀ x, (1 < x ∧ x ≤ 4) → |x - t| < 1) →
  (2 ≤ t ∧ t ≤ 3) := by
sorry

end NUMINAMATH_CALUDE_range_of_t_l1621_162139


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1621_162146

theorem rectangle_perimeter (h w : ℝ) : 
  h > 0 ∧ w > 0 ∧              -- positive dimensions
  h * w = 40 ∧                 -- area of rectangle is 40
  w > 2 * h ∧                  -- width more than twice the height
  h * (w - h) = 24 →           -- area of parallelogram after folding
  2 * h + 2 * w = 28 :=        -- perimeter of original rectangle
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1621_162146


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l1621_162152

/-- The number of distinct diagonals in a convex nonagon -/
def diagonals_in_nonagon : ℕ := 27

/-- A convex nonagon has 27 distinct diagonals -/
theorem nonagon_diagonals : diagonals_in_nonagon = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l1621_162152


namespace NUMINAMATH_CALUDE_fairview_population_l1621_162172

/-- The number of cities in the District of Fairview -/
def num_cities : ℕ := 25

/-- The average population of cities in the District of Fairview -/
def avg_population : ℕ := 3800

/-- The total population of the District of Fairview -/
def total_population : ℕ := num_cities * avg_population

theorem fairview_population :
  total_population = 95000 := by
  sorry

end NUMINAMATH_CALUDE_fairview_population_l1621_162172


namespace NUMINAMATH_CALUDE_system_solutions_l1621_162135

theorem system_solutions (a : ℝ) (x y : ℝ) 
  (h1 : x - 2*y = 3 - a) 
  (h2 : x + y = 2*a) 
  (h3 : -2 ≤ a ∧ a ≤ 0) : 
  (a = 0 → x = -y) ∧ 
  (a = -1 → 2*x - y = 1 - a) := by
sorry

end NUMINAMATH_CALUDE_system_solutions_l1621_162135


namespace NUMINAMATH_CALUDE_production_period_is_seven_days_l1621_162106

def computers_per_day : ℕ := 1500
def price_per_computer : ℕ := 150
def total_revenue : ℕ := 1575000

theorem production_period_is_seven_days :
  (total_revenue / price_per_computer) / computers_per_day = 7 := by
  sorry

end NUMINAMATH_CALUDE_production_period_is_seven_days_l1621_162106


namespace NUMINAMATH_CALUDE_inequality_proof_l1621_162162

theorem inequality_proof (a b c : ℝ) (h1 : a < b) (h2 : b < 0) (h3 : c > 0) :
  b / a < (c - b) / (c - a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1621_162162


namespace NUMINAMATH_CALUDE_quadratic_sum_l1621_162107

theorem quadratic_sum (a h k : ℝ) : 
  (∀ x, 5 * x^2 - 10 * x - 3 = a * (x - h)^2 + k) → 
  a + h + k = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_sum_l1621_162107


namespace NUMINAMATH_CALUDE_binomial_variance_example_l1621_162190

/-- The variance of a binomial distribution with 100 trials and 0.02 probability of success is 1.96 -/
theorem binomial_variance_example :
  let n : ℕ := 100
  let p : ℝ := 0.02
  let q : ℝ := 1 - p
  let variance : ℝ := n * p * q
  variance = 1.96 := by
  sorry

end NUMINAMATH_CALUDE_binomial_variance_example_l1621_162190


namespace NUMINAMATH_CALUDE_circle_center_l1621_162192

/-- The equation of a circle in the x-y plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 + 4*y = -4

/-- The center of a circle -/
def is_center (h k : ℝ) : Prop :=
  ∀ x y : ℝ, circle_equation x y ↔ (x - h)^2 + (y - k)^2 = 16

/-- Theorem: The center of the circle with equation x^2 - 8x + y^2 + 4y = -4 is (4, -2) -/
theorem circle_center : is_center 4 (-2) := by sorry

end NUMINAMATH_CALUDE_circle_center_l1621_162192


namespace NUMINAMATH_CALUDE_shaded_area_between_circles_l1621_162180

theorem shaded_area_between_circles (r₁ r₂ : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 5) : 
  let R := (r₁ + r₂) / 2
  π * R^2 - (π * r₁^2 + π * r₂^2) = 40 * π :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_between_circles_l1621_162180


namespace NUMINAMATH_CALUDE_original_buckets_count_l1621_162147

/-- The number of buckets needed to fill a tank with reduced capacity buckets -/
def reduced_buckets : ℕ := 105

/-- The ratio of the reduced bucket capacity to the original bucket capacity -/
def capacity_ratio : ℚ := 2 / 5

/-- The volume of the tank in terms of original bucket capacity -/
def tank_volume (original_buckets : ℕ) : ℚ := original_buckets

/-- The volume of the tank in terms of reduced bucket capacity -/
def tank_volume_reduced : ℚ := reduced_buckets * capacity_ratio

theorem original_buckets_count : 
  ∃ (original_buckets : ℕ), 
    tank_volume original_buckets = tank_volume_reduced ∧ 
    original_buckets = 42 :=
sorry

end NUMINAMATH_CALUDE_original_buckets_count_l1621_162147


namespace NUMINAMATH_CALUDE_trivia_team_distribution_l1621_162186

theorem trivia_team_distribution (total : ℕ) (not_picked : ℕ) (groups : ℕ) : 
  total = 36 → not_picked = 9 → groups = 3 → 
  (total - not_picked) / groups = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_trivia_team_distribution_l1621_162186


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1621_162129

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 2 * x + 1 > 0) ↔ a > 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1621_162129


namespace NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_9_divisible_by_11_l1621_162168

theorem smallest_positive_integer_ending_in_9_divisible_by_11 :
  ∃ (n : ℕ), n > 0 ∧ n % 10 = 9 ∧ n % 11 = 0 ∧
  ∀ (m : ℕ), m > 0 → m % 10 = 9 → m % 11 = 0 → m ≥ n :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_9_divisible_by_11_l1621_162168


namespace NUMINAMATH_CALUDE_matinee_attendance_difference_l1621_162118

theorem matinee_attendance_difference (child_price adult_price total_receipts num_children : ℚ)
  (h1 : child_price = 4.5)
  (h2 : adult_price = 6.75)
  (h3 : total_receipts = 405)
  (h4 : num_children = 48) :
  num_children - (total_receipts - num_children * child_price) / adult_price = 20 := by
  sorry

end NUMINAMATH_CALUDE_matinee_attendance_difference_l1621_162118


namespace NUMINAMATH_CALUDE_g_properties_l1621_162154

noncomputable def f (a x : ℝ) : ℝ := a * Real.sqrt (1 - x^2) + Real.sqrt (1 + x) + Real.sqrt (1 - x)

noncomputable def g (a : ℝ) : ℝ := ⨆ (x : ℝ), f a x

theorem g_properties (a : ℝ) :
  (a > -1/2 → g a = a + 2) ∧
  (-Real.sqrt 2 / 2 < a ∧ a ≤ -1/2 → g a = -a - 1/(2*a)) ∧
  (a ≤ -Real.sqrt 2 / 2 → g a = Real.sqrt 2) ∧
  (g a = g (1/a) ↔ a = 1 ∨ (-Real.sqrt 2 ≤ a ∧ a ≤ -Real.sqrt 2 / 2)) :=
sorry

end NUMINAMATH_CALUDE_g_properties_l1621_162154


namespace NUMINAMATH_CALUDE_older_brother_allowance_l1621_162145

theorem older_brother_allowance (younger_allowance older_allowance : ℕ) : 
  younger_allowance + older_allowance = 12000 →
  older_allowance = younger_allowance + 1000 →
  older_allowance = 6500 := by
sorry

end NUMINAMATH_CALUDE_older_brother_allowance_l1621_162145


namespace NUMINAMATH_CALUDE_playground_area_l1621_162132

theorem playground_area (total_posts : ℕ) (post_spacing : ℕ) 
  (h1 : total_posts = 28)
  (h2 : post_spacing = 6)
  (h3 : ∃ (short_side long_side : ℕ), 
    short_side + 1 + long_side + 1 = total_posts ∧ 
    long_side + 1 = 3 * (short_side + 1)) :
  ∃ (width length : ℕ), 
    width * length = 1188 ∧ 
    width = post_spacing * short_side ∧ 
    length = post_spacing * long_side :=
sorry

end NUMINAMATH_CALUDE_playground_area_l1621_162132


namespace NUMINAMATH_CALUDE_heart_stickers_count_l1621_162116

/-- Represents the number of stickers needed to decorate a single page -/
def stickers_per_page (total_stickers : ℕ) (num_pages : ℕ) : ℕ :=
  total_stickers / num_pages

/-- Checks if the total number of stickers can be evenly distributed among the pages -/
def can_distribute_evenly (total_stickers : ℕ) (num_pages : ℕ) : Prop :=
  total_stickers % num_pages = 0

theorem heart_stickers_count (star_stickers : ℕ) (num_pages : ℕ) (heart_stickers : ℕ) : 
  star_stickers = 27 →
  num_pages = 9 →
  can_distribute_evenly (heart_stickers + star_stickers) num_pages →
  heart_stickers = 9 := by
  sorry

end NUMINAMATH_CALUDE_heart_stickers_count_l1621_162116


namespace NUMINAMATH_CALUDE_decagon_diagonal_intersections_l1621_162173

-- Define a regular polygon
def RegularPolygon (n : ℕ) := {p : ℕ | p ≥ 3}

-- Define a decagon
def Decagon := RegularPolygon 10

-- Define the number of interior intersection points of diagonals
def InteriorIntersectionPoints (p : RegularPolygon 10) : ℕ := sorry

-- Define the number of ways to choose 4 vertices from 10
def Choose4From10 : ℕ := Nat.choose 10 4

-- Theorem statement
theorem decagon_diagonal_intersections (d : Decagon) : 
  InteriorIntersectionPoints d = Choose4From10 := by sorry

end NUMINAMATH_CALUDE_decagon_diagonal_intersections_l1621_162173


namespace NUMINAMATH_CALUDE_square_sum_101_99_l1621_162108

theorem square_sum_101_99 : 101 * 101 + 99 * 99 = 20200 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_101_99_l1621_162108


namespace NUMINAMATH_CALUDE_complex_square_simplification_l1621_162133

theorem complex_square_simplification :
  let z : ℂ := 4 - 3 * I
  z^2 = 7 - 24 * I := by sorry

end NUMINAMATH_CALUDE_complex_square_simplification_l1621_162133


namespace NUMINAMATH_CALUDE_escalator_time_theorem_l1621_162199

/-- The time taken for a person to cover the length of an escalator -/
theorem escalator_time_theorem (escalator_speed : ℝ) (person_speed : ℝ) (escalator_length : ℝ) :
  escalator_speed = 12 →
  person_speed = 8 →
  escalator_length = 160 →
  escalator_length / (escalator_speed + person_speed) = 8 := by
  sorry

end NUMINAMATH_CALUDE_escalator_time_theorem_l1621_162199


namespace NUMINAMATH_CALUDE_molecular_weight_BaBr2_l1621_162185

/-- The atomic weight of barium in g/mol -/
def atomic_weight_Ba : ℝ := 137.33

/-- The atomic weight of bromine in g/mol -/
def atomic_weight_Br : ℝ := 79.90

/-- The number of bromine atoms in a barium bromide molecule -/
def Br_count : ℕ := 2

/-- The number of moles of barium bromide -/
def moles_BaBr2 : ℕ := 4

/-- Theorem: The molecular weight of 4 moles of Barium bromide (BaBr2) is 1188.52 grams -/
theorem molecular_weight_BaBr2 : 
  moles_BaBr2 * (atomic_weight_Ba + Br_count * atomic_weight_Br) = 1188.52 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_BaBr2_l1621_162185


namespace NUMINAMATH_CALUDE_smallest_d_value_l1621_162101

theorem smallest_d_value (d : ℝ) : 
  (5 * Real.sqrt 5)^2 + (d + 4)^2 = (5 * d)^2 → d ≥ (1 + Real.sqrt 212.5) / 6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_d_value_l1621_162101


namespace NUMINAMATH_CALUDE_ellipse_intersecting_lines_l1621_162177

/-- Represents an ellipse with center at the origin and foci on the x-axis -/
structure Ellipse where
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis
  h : a > b ∧ b > 0

/-- The standard equation of an ellipse -/
def Ellipse.equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- A line in the form y = kx + m -/
structure Line where
  k : ℝ
  m : ℝ
  h : k ≠ 0

/-- Theorem about a specific ellipse and lines intersecting it -/
theorem ellipse_intersecting_lines (e : Ellipse) 
  (h1 : e.a + (e.a^2 - e.b^2).sqrt = 3)
  (h2 : e.a - (e.a^2 - e.b^2).sqrt = 1) :
  (∀ x y, e.equation x y ↔ x^2/4 + y^2/3 = 1) ∧
  (∀ l : Line, ∃ M N : ℝ × ℝ,
    M ≠ N ∧
    e.equation M.1 M.2 ∧
    e.equation N.1 N.2 ∧
    M.2 = l.k * M.1 + l.m ∧
    N.2 = l.k * N.1 + l.m ∧
    M ≠ (2, 0) ∧ N ≠ (2, 0) ∧ M ≠ (-2, 0) ∧ N ≠ (-2, 0) ∧
    (M.1 - 2) * (N.1 - 2) + M.2 * N.2 = 0 →
    l.k * (2/7) + l.m = 0) :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersecting_lines_l1621_162177


namespace NUMINAMATH_CALUDE_a_equals_two_l1621_162124

theorem a_equals_two (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x + 1 > 0) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_a_equals_two_l1621_162124


namespace NUMINAMATH_CALUDE_magazine_choice_count_l1621_162181

theorem magazine_choice_count : 
  let science_count : Nat := 4
  let digest_count : Nat := 3
  let entertainment_count : Nat := 2
  science_count + digest_count + entertainment_count = 9 :=
by sorry

end NUMINAMATH_CALUDE_magazine_choice_count_l1621_162181


namespace NUMINAMATH_CALUDE_function_lower_bound_l1621_162109

/-- Given a function f(x) = x^2 - (a+1)x + a, where a is a real number,
    if f(x) ≥ -1 for all x > 1, then a ≤ 3 -/
theorem function_lower_bound (a : ℝ) :
  (∀ x > 1, x^2 - (a + 1)*x + a ≥ -1) → a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_function_lower_bound_l1621_162109


namespace NUMINAMATH_CALUDE_student_number_problem_l1621_162178

theorem student_number_problem (x : ℝ) : 6 * x - 138 = 102 → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l1621_162178


namespace NUMINAMATH_CALUDE_tangent_line_at_origin_l1621_162136

/-- Given a real number a and a function f(x) = x^3 + ax^2 + (a - 2)x whose derivative
    is an even function, the tangent line to f(x) at the origin has equation y = -2x. -/
theorem tangent_line_at_origin (a : ℝ) :
  let f := fun x : ℝ => x^3 + a*x^2 + (a - 2)*x
  let f' := fun x : ℝ => 3*x^2 + 2*a*x + (a - 2)
  (∀ x, f' x = f' (-x)) →
  (fun x => -2*x) = fun x => (f' 0) * x :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_origin_l1621_162136


namespace NUMINAMATH_CALUDE_b_spend_percent_calculation_l1621_162170

def combined_salary : ℝ := 3000
def a_salary : ℝ := 2250
def a_spend_percent : ℝ := 0.95

theorem b_spend_percent_calculation :
  let b_salary := combined_salary - a_salary
  let a_savings := a_salary * (1 - a_spend_percent)
  let b_spend_percent := 1 - (a_savings / b_salary)
  b_spend_percent = 0.85 := by sorry

end NUMINAMATH_CALUDE_b_spend_percent_calculation_l1621_162170


namespace NUMINAMATH_CALUDE_perpendicular_planes_l1621_162130

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines
variable (perpLine : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpLinePlane : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (perpPlane : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes 
  (m n : Line) (α β : Plane) :
  perpLine m n → 
  perpLinePlane m α → 
  perpLinePlane n β → 
  perpPlane α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_l1621_162130


namespace NUMINAMATH_CALUDE_flowerbed_perimeter_l1621_162165

/-- A rectangular flowerbed with given dimensions --/
structure Flowerbed where
  width : ℝ
  length : ℝ

/-- The perimeter of a rectangular flowerbed --/
def perimeter (f : Flowerbed) : ℝ := 2 * (f.length + f.width)

/-- Theorem: The perimeter of the specific flowerbed is 22 meters --/
theorem flowerbed_perimeter :
  ∃ (f : Flowerbed), f.width = 4 ∧ f.length = 2 * f.width - 1 ∧ perimeter f = 22 := by
  sorry

end NUMINAMATH_CALUDE_flowerbed_perimeter_l1621_162165


namespace NUMINAMATH_CALUDE_fraction_irreducible_l1621_162148

theorem fraction_irreducible (n : ℕ+) : Nat.gcd (n^2 + n - 1) (n^2 + 2*n) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_irreducible_l1621_162148


namespace NUMINAMATH_CALUDE_two_thirds_cubed_l1621_162194

theorem two_thirds_cubed : (2 / 3 : ℚ) ^ 3 = 8 / 27 := by
  sorry

end NUMINAMATH_CALUDE_two_thirds_cubed_l1621_162194


namespace NUMINAMATH_CALUDE_greatest_product_prime_factorization_sum_l1621_162149

/-- The greatest product of positive integers summing to 2014 -/
def A : ℕ := 3^670 * 2^2

/-- The sum of all positive integers that produce A -/
def sum_of_factors : ℕ := 2014

/-- Function to calculate the sum of bases and exponents in prime factorization -/
def sum_bases_and_exponents (n : ℕ) : ℕ := sorry

theorem greatest_product_prime_factorization_sum :
  sum_bases_and_exponents A = 677 :=
sorry

end NUMINAMATH_CALUDE_greatest_product_prime_factorization_sum_l1621_162149


namespace NUMINAMATH_CALUDE_mary_money_l1621_162100

def quarters : ℕ := 21
def dimes : ℕ := (quarters - 7) / 2

def quarter_value : ℚ := 25 / 100
def dime_value : ℚ := 10 / 100

def total_money : ℚ := quarters * quarter_value + dimes * dime_value

theorem mary_money : total_money = 595 / 100 := by
  sorry

end NUMINAMATH_CALUDE_mary_money_l1621_162100


namespace NUMINAMATH_CALUDE_max_angle_MPN_at_x_equals_one_l1621_162102

structure Point where
  x : ℝ
  y : ℝ

def M : Point := ⟨-1, 2⟩
def N : Point := ⟨1, 4⟩

def angle_MPN (P : Point) : ℝ :=
  sorry  -- Definition of angle MPN

theorem max_angle_MPN_at_x_equals_one :
  ∃ (P : Point), P.y = 0 ∧ 
    (∀ (Q : Point), Q.y = 0 → angle_MPN P ≥ angle_MPN Q) ∧
    P.x = 1 := by
  sorry

#check max_angle_MPN_at_x_equals_one

end NUMINAMATH_CALUDE_max_angle_MPN_at_x_equals_one_l1621_162102


namespace NUMINAMATH_CALUDE_largest_five_digit_sum_20_l1621_162120

def is_five_digit (n : ℕ) : Prop := n ≥ 10000 ∧ n < 100000

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem largest_five_digit_sum_20 :
  ∀ n : ℕ, is_five_digit n → digit_sum n = 20 → n ≤ 99200 :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_sum_20_l1621_162120


namespace NUMINAMATH_CALUDE_new_person_weight_l1621_162188

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 2.5 →
  replaced_weight = 20 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 40 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l1621_162188


namespace NUMINAMATH_CALUDE_erased_number_l1621_162131

/-- Represents a quadratic polynomial ax^2 + bx + c with roots m and n -/
structure QuadraticPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ
  m : ℤ
  n : ℤ

/-- Checks if the given QuadraticPolynomial satisfies Vieta's formulas -/
def satisfiesVieta (p : QuadraticPolynomial) : Prop :=
  p.c = p.a * p.m * p.n ∧ p.b = -p.a * (p.m + p.n)

/-- Checks if four of the five numbers in the QuadraticPolynomial are 2, 3, 4, -5 -/
def hasFourOf (p : QuadraticPolynomial) : Prop :=
  (p.a = 2 ∧ p.b = 4 ∧ p.m = 3 ∧ p.n = -5) ∨
  (p.a = 2 ∧ p.b = 4 ∧ p.m = 3 ∧ p.c = -5) ∨
  (p.a = 2 ∧ p.b = 4 ∧ p.n = -5 ∧ p.c = 3) ∨
  (p.a = 2 ∧ p.m = 3 ∧ p.n = -5 ∧ p.c = 4) ∨
  (p.b = 4 ∧ p.m = 3 ∧ p.n = -5 ∧ p.c = 2)

theorem erased_number (p : QuadraticPolynomial) :
  satisfiesVieta p → hasFourOf p → 
  p.a = -30 ∨ p.b = -30 ∨ p.c = -30 ∨ p.m = -30 ∨ p.n = -30 := by
  sorry


end NUMINAMATH_CALUDE_erased_number_l1621_162131


namespace NUMINAMATH_CALUDE_egg_distribution_l1621_162171

theorem egg_distribution (a : ℚ) : a = 7 ↔
  (a / 2 - 1 / 2) / 2 - 1 / 2 - ((a / 4 - 3 / 4) / 2 + 1 / 2) = 0 :=
by sorry

end NUMINAMATH_CALUDE_egg_distribution_l1621_162171


namespace NUMINAMATH_CALUDE_expr_D_not_fraction_l1621_162111

-- Define what a fraction is
def is_fraction (expr : ℚ → ℚ) : Prop :=
  ∃ (f g : ℚ → ℚ), ∀ x, expr x = (f x) / (g x) ∧ g x ≠ 0

-- Define the expressions
def expr_A (x : ℚ) : ℚ := 1 / (x^2)
def expr_B (a b : ℚ) : ℚ := (b + 3) / a
def expr_C (x : ℚ) : ℚ := (x^2 - 1) / (x + 1)
def expr_D (a : ℚ) : ℚ := (2 / 7) * a

-- Theorem stating that expr_D is not a fraction
theorem expr_D_not_fraction : ¬ is_fraction expr_D :=
sorry

end NUMINAMATH_CALUDE_expr_D_not_fraction_l1621_162111


namespace NUMINAMATH_CALUDE_optimal_mask_pricing_l1621_162150

/-- Represents the cost and pricing model for masks during an epidemic --/
structure MaskPricing where
  costA : ℝ  -- Cost of type A masks
  costB : ℝ  -- Cost of type B masks
  sellingPrice : ℝ  -- Selling price of type B masks
  profit : ℝ  -- Daily average total profit

/-- Conditions for the mask pricing problem --/
def MaskPricingConditions (m : MaskPricing) : Prop :=
  m.costB = 2 * m.costA - 10 ∧  -- Condition 1
  6000 / m.costA = 10000 / m.costB ∧  -- Condition 2
  m.profit = (m.sellingPrice - m.costB) * (100 - 5 * (m.sellingPrice - 60))  -- Conditions 3 and 4 combined

/-- Theorem stating the optimal solution for the mask pricing problem --/
theorem optimal_mask_pricing :
  ∃ m : MaskPricing,
    MaskPricingConditions m ∧
    m.costA = 30 ∧
    m.costB = 50 ∧
    m.sellingPrice = 65 ∧
    m.profit = 1125 ∧
    ∀ m' : MaskPricing, MaskPricingConditions m' → m'.profit ≤ m.profit :=
by
  sorry

end NUMINAMATH_CALUDE_optimal_mask_pricing_l1621_162150


namespace NUMINAMATH_CALUDE_tenth_term_is_110_l1621_162138

/-- Define the sequence of small stars -/
def smallStars (n : ℕ) : ℕ := n * (n + 1)

/-- Theorem: The 10th term of the sequence is 110 -/
theorem tenth_term_is_110 : smallStars 10 = 110 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_is_110_l1621_162138


namespace NUMINAMATH_CALUDE_smallest_constant_inequality_l1621_162198

theorem smallest_constant_inequality (x y z w : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) :
  Real.sqrt (x / (y + z + w)) + Real.sqrt (y / (x + z + w)) +
  Real.sqrt (z / (x + y + w)) + Real.sqrt (w / (x + y + z)) ≤ 2 ∧
  ∀ M : ℝ, M < 2 → ∃ a b c d : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    Real.sqrt (a / (b + c + d)) + Real.sqrt (b / (a + c + d)) +
    Real.sqrt (c / (a + b + d)) + Real.sqrt (d / (a + b + c)) > M :=
by sorry

end NUMINAMATH_CALUDE_smallest_constant_inequality_l1621_162198


namespace NUMINAMATH_CALUDE_sequence_value_l1621_162160

theorem sequence_value (a : ℕ → ℕ) (h1 : a 1 = 2) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n + 2 * n) : a 100 = 9902 := by
  sorry

end NUMINAMATH_CALUDE_sequence_value_l1621_162160


namespace NUMINAMATH_CALUDE_probability_marked_standard_deck_l1621_162115

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (total_ranks : ℕ)
  (total_suits : ℕ)
  (marked_ranks : ℕ)

/-- A standard deck with 52 cards, 13 ranks, 4 suits, and 4 marked ranks -/
def standard_deck : Deck :=
  { total_cards := 52,
    total_ranks := 13,
    total_suits := 4,
    marked_ranks := 4 }

/-- The probability of drawing a card with a special symbol -/
def probability_marked (d : Deck) : ℚ :=
  (d.marked_ranks * d.total_suits) / d.total_cards

/-- Theorem: The probability of drawing a card with a special symbol from a standard deck is 4/13 -/
theorem probability_marked_standard_deck :
  probability_marked standard_deck = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_marked_standard_deck_l1621_162115


namespace NUMINAMATH_CALUDE_count_solution_pairs_l1621_162144

/-- The number of pairs of positive integers (x, y) satisfying x^2 - y^2 = 72 -/
def solution_count : Nat :=
  (Finset.filter (fun p : Nat × Nat =>
    let (x, y) := p
    x > 0 ∧ y > 0 ∧ x^2 - y^2 = 72
  ) (Finset.product (Finset.range 100) (Finset.range 100))).card

theorem count_solution_pairs : solution_count = 3 := by
  sorry

end NUMINAMATH_CALUDE_count_solution_pairs_l1621_162144


namespace NUMINAMATH_CALUDE_ice_cream_flavors_l1621_162143

/-- The number of ways to distribute n indistinguishable objects into k distinguishable boxes -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The binomial coefficient C(n,k) -/
def binomial_coefficient (n k : ℕ) : ℕ := sorry

theorem ice_cream_flavors :
  distribute 5 4 = binomial_coefficient 8 3 := by sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_l1621_162143


namespace NUMINAMATH_CALUDE_horse_division_l1621_162122

theorem horse_division (total_horses : ℕ) (eldest_share middle_share youngest_share : ℕ) : 
  total_horses = 7 →
  eldest_share = 4 →
  middle_share = 2 →
  youngest_share = 1 →
  eldest_share + middle_share + youngest_share = total_horses →
  eldest_share = (total_horses + 1) / 2 →
  middle_share = (total_horses + 1) / 4 →
  youngest_share = (total_horses + 1) / 8 :=
by sorry

end NUMINAMATH_CALUDE_horse_division_l1621_162122


namespace NUMINAMATH_CALUDE_afternoon_shells_l1621_162179

/-- Given that Lino picked up 292 shells in the morning and a total of 616 shells,
    prove that he picked up 324 shells in the afternoon. -/
theorem afternoon_shells (morning_shells : ℕ) (total_shells : ℕ) (h1 : morning_shells = 292) (h2 : total_shells = 616) :
  total_shells - morning_shells = 324 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_shells_l1621_162179


namespace NUMINAMATH_CALUDE_compute_expression_l1621_162189

theorem compute_expression : 
  20 * (150 / 3 + 50 / 6 + 16 / 25 + 2) = 90460 / 75 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l1621_162189


namespace NUMINAMATH_CALUDE_inclination_angle_expression_l1621_162125

theorem inclination_angle_expression (θ : Real) : 
  (2 : Real) * Real.tan θ = -1 → 
  (Real.sin θ + Real.cos θ) / (Real.sin θ - Real.cos θ) = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_inclination_angle_expression_l1621_162125


namespace NUMINAMATH_CALUDE_adam_final_money_l1621_162164

/-- Calculates the final amount of money Adam has after a series of transactions --/
theorem adam_final_money (initial : ℚ) (game_cost : ℚ) (snack_cost : ℚ) (found : ℚ) (allowance : ℚ) :
  initial = 5.25 →
  game_cost = 2.30 →
  snack_cost = 1.75 →
  found = 1.00 →
  allowance = 5.50 →
  initial - game_cost - snack_cost + found + allowance = 7.70 := by
  sorry

end NUMINAMATH_CALUDE_adam_final_money_l1621_162164


namespace NUMINAMATH_CALUDE_some_number_value_l1621_162156

theorem some_number_value (x y N : ℝ) 
  (eq1 : 2 * x + y = N) 
  (eq2 : x + 2 * y = 5) 
  (eq3 : (x + y) / 3 = 1) : 
  N = 4 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l1621_162156


namespace NUMINAMATH_CALUDE_call_center_team_b_fraction_l1621_162121

/-- The fraction of total calls processed by team B in a call center with two teams -/
theorem call_center_team_b_fraction (team_a team_b : ℕ) (calls_a calls_b : ℚ) :
  team_a = (5 : ℚ) / 8 * team_b →
  calls_a = (7 : ℚ) / 5 * calls_b →
  (team_b * calls_b) / (team_a * calls_a + team_b * calls_b) = (8 : ℚ) / 15 := by
sorry


end NUMINAMATH_CALUDE_call_center_team_b_fraction_l1621_162121


namespace NUMINAMATH_CALUDE_problem_solution_l1621_162167

theorem problem_solution (a b c : ℝ) (m n : ℕ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq1 : (a + b) * (a + c) = b * c + 2)
  (h_eq2 : (b + c) * (b + a) = c * a + 5)
  (h_eq3 : (c + a) * (c + b) = a * b + 9)
  (h_abc : a * b * c = m / n)
  (h_coprime : Nat.Coprime m n) : 
  100 * m + n = 4532 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1621_162167


namespace NUMINAMATH_CALUDE_rectangle_longer_side_l1621_162114

/-- Given a circle with radius 6 cm that is tangent to three sides of a rectangle,
    where the rectangle's area is four times the circle's area,
    prove that the length of the longer side of the rectangle is 12π cm. -/
theorem rectangle_longer_side (r : ℝ) (circle_area rectangle_area : ℝ) 
  (shorter_side longer_side : ℝ) :
  r = 6 →
  circle_area = Real.pi * r^2 →
  rectangle_area = 4 * circle_area →
  shorter_side = 2 * r →
  rectangle_area = shorter_side * longer_side →
  longer_side = 12 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_rectangle_longer_side_l1621_162114


namespace NUMINAMATH_CALUDE_class_gathering_problem_l1621_162141

theorem class_gathering_problem (male_students : ℕ) (female_students : ℕ) :
  female_students = male_students + 6 →
  (female_students : ℚ) / (male_students + female_students) = 2 / 3 →
  male_students + female_students = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_class_gathering_problem_l1621_162141


namespace NUMINAMATH_CALUDE_probability_13_or_more_points_l1621_162193

/-- Represents the face cards in a deck --/
inductive FaceCard
  | A
  | K
  | Q
  | J

/-- Assigns points to a face card --/
def point_value (card : FaceCard) : ℕ :=
  match card with
  | FaceCard.A => 4
  | FaceCard.K => 3
  | FaceCard.Q => 2
  | FaceCard.J => 1

/-- Calculates the total points for a hand of face cards --/
def hand_points (hand : List FaceCard) : ℕ :=
  hand.map point_value |>.sum

/-- Represents all possible 4-card hands of face cards --/
def all_hands : List (List FaceCard) :=
  sorry

/-- Checks if a hand has 13 or more points --/
def has_13_or_more_points (hand : List FaceCard) : Bool :=
  hand_points hand ≥ 13

/-- Counts the number of hands with 13 or more points --/
def count_13_or_more : ℕ :=
  all_hands.filter has_13_or_more_points |>.length

theorem probability_13_or_more_points :
  count_13_or_more / all_hands.length = 197 / 1820 := by
  sorry

end NUMINAMATH_CALUDE_probability_13_or_more_points_l1621_162193


namespace NUMINAMATH_CALUDE_x_range_l1621_162153

theorem x_range (x : ℝ) : (|x + 1| + |x - 1| = 2) ↔ (-1 ≤ x ∧ x ≤ 1) := by sorry

end NUMINAMATH_CALUDE_x_range_l1621_162153


namespace NUMINAMATH_CALUDE_large_box_125_times_small_box_l1621_162127

-- Define the dimensions of the large box
def large_width : ℝ := 30
def large_length : ℝ := 20
def large_height : ℝ := 5

-- Define the dimensions of the small box
def small_width : ℝ := 6
def small_length : ℝ := 4
def small_height : ℝ := 1

-- Define the volume calculation function for a cuboid
def cuboid_volume (width length height : ℝ) : ℝ := width * length * height

-- Theorem statement
theorem large_box_125_times_small_box :
  cuboid_volume large_width large_length large_height =
  125 * cuboid_volume small_width small_length small_height := by
  sorry

end NUMINAMATH_CALUDE_large_box_125_times_small_box_l1621_162127


namespace NUMINAMATH_CALUDE_optimal_strategy_highest_hunter_l1621_162174

/-- Represents a hunter in the treasure division game -/
structure Hunter :=
  (id : Nat)
  (coins : Nat)

/-- Represents the state of the game -/
structure GameState :=
  (n : Nat)  -- Total number of hunters
  (m : Nat)  -- Total number of coins
  (hunters : List Hunter)

/-- Checks if a proposal is accepted by majority vote -/
def isProposalAccepted (state : GameState) (proposal : List Hunter) : Prop :=
  2 * (proposal.filter (fun h => h.coins > 0)).length > state.hunters.length

/-- Generates the optimal proposal for a given hunter -/
def optimalProposal (state : GameState) (hunterId : Nat) : List Hunter :=
  sorry

/-- Theorem: The optimal strategy for the highest-numbered hunter is to propose
    m - (n ÷ 2) coins for themselves and 1 coin each for the even-numbered
    hunters below them, until they secure a majority vote -/
theorem optimal_strategy_highest_hunter (state : GameState) :
  let proposal := optimalProposal state state.n
  isProposalAccepted state proposal ∧
  (proposal.head?.map Hunter.coins).getD 0 = state.m - (state.n / 2) ∧
  (proposal.tail.filter (fun h => h.coins > 0)).all (fun h => h.coins = 1 ∧ h.id % 2 = 0) :=
  sorry


end NUMINAMATH_CALUDE_optimal_strategy_highest_hunter_l1621_162174


namespace NUMINAMATH_CALUDE_artichokey_invested_seven_l1621_162175

/-- Represents the investment and payout of earthworms -/
structure EarthwormInvestment where
  total_earthworms : ℕ
  okeydokey_apples : ℕ
  okeydokey_earthworms : ℕ

/-- Calculates the number of apples Artichokey invested -/
def artichokey_investment (e : EarthwormInvestment) : ℕ :=
  sorry

/-- Theorem stating that Artichokey invested 7 apples -/
theorem artichokey_invested_seven (e : EarthwormInvestment)
  (h1 : e.total_earthworms = 60)
  (h2 : e.okeydokey_apples = 5)
  (h3 : e.okeydokey_earthworms = 25)
  (h4 : e.okeydokey_earthworms * e.total_earthworms = e.okeydokey_apples * (e.total_earthworms + e.okeydokey_earthworms)) :
  artichokey_investment e = 7 :=
sorry

end NUMINAMATH_CALUDE_artichokey_invested_seven_l1621_162175


namespace NUMINAMATH_CALUDE_a_10_equals_512_l1621_162176

/-- The sequence {aₙ} where Sₙ = 2aₙ - 1 for all n ∈ ℕ⁺, and Sₙ is the sum of the first n terms of {aₙ} -/
def sequence_a (n : ℕ+) : ℝ :=
  sorry

/-- The sum of the first n terms of the sequence {aₙ} -/
def S (n : ℕ+) : ℝ :=
  sorry

/-- The main theorem stating that a₁₀ = 512 -/
theorem a_10_equals_512 (h : ∀ n : ℕ+, S n = 2 * sequence_a n - 1) : sequence_a 10 = 512 := by
  sorry

end NUMINAMATH_CALUDE_a_10_equals_512_l1621_162176


namespace NUMINAMATH_CALUDE_min_k_value_l1621_162105

/-- Given a line and a circle in a Cartesian coordinate system,
    prove that the minimum value of k satisfying the conditions is -√3 -/
theorem min_k_value (k : ℝ) : 
  (∃ P : ℝ × ℝ, P.2 = k * (P.1 - 3 * Real.sqrt 3)) →
  (∃ Q : ℝ × ℝ, Q.1^2 + (Q.2 - 1)^2 = 1) →
  (∃ P Q : ℝ × ℝ, P = (3 * Q.1, 3 * Q.2)) →
  -Real.sqrt 3 ≤ k :=
by sorry

end NUMINAMATH_CALUDE_min_k_value_l1621_162105


namespace NUMINAMATH_CALUDE_certain_number_equation_l1621_162155

theorem certain_number_equation (x : ℝ) : 7 * x = 4 * x + 12 + 6 ↔ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_equation_l1621_162155
