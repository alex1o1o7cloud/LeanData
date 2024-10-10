import Mathlib

namespace cheese_weight_l2161_216133

/-- Represents the weight of two pieces of cheese -/
structure CheesePair :=
  (larger : ℕ)
  (smaller : ℕ)

/-- The function that represents taking a bite from the larger piece -/
def take_bite (pair : CheesePair) : CheesePair :=
  ⟨pair.larger - pair.smaller, pair.smaller⟩

/-- The theorem stating the original weight of the cheese -/
theorem cheese_weight (initial : CheesePair) :
  (take_bite (take_bite (take_bite initial))) = ⟨20, 20⟩ →
  initial.larger + initial.smaller = 680 :=
sorry

end cheese_weight_l2161_216133


namespace fraction_multiplication_l2161_216158

theorem fraction_multiplication : (2 : ℚ) / 3 * 5 / 7 * 3 / 4 = 5 / 14 := by
  sorry

end fraction_multiplication_l2161_216158


namespace min_sales_to_break_even_l2161_216117

def old_salary : ℕ := 75000
def new_base_salary : ℕ := 45000
def commission_rate : ℚ := 15 / 100
def sale_amount : ℕ := 750

theorem min_sales_to_break_even :
  let difference := old_salary - new_base_salary
  let commission_per_sale := commission_rate * sale_amount
  let min_sales := (difference : ℚ) / commission_per_sale
  ⌈min_sales⌉ = 267 := by sorry

end min_sales_to_break_even_l2161_216117


namespace sector_area_l2161_216172

/-- The area of a circular sector with central angle 120° and radius 4 is 16π/3 -/
theorem sector_area : 
  let central_angle : ℝ := 120
  let radius : ℝ := 4
  let sector_area : ℝ := (central_angle * π * radius^2) / 360
  sector_area = 16 * π / 3 := by sorry

end sector_area_l2161_216172


namespace triangle_side_angle_relation_l2161_216183

/-- Given a triangle ABC where a, b, c are sides opposite to angles A, B, C respectively,
    if a² = b² + ¼c², then (a cos B) / c = 5/8 -/
theorem triangle_side_angle_relation (a b c : ℝ) (h : a^2 = b^2 + (1/4)*c^2) :
  (a * Real.cos (Real.arccos ((b^2 + c^2 - a^2) / (2*b*c)))) / c = 5/8 := by
  sorry


end triangle_side_angle_relation_l2161_216183


namespace perpendicular_tangent_line_l2161_216177

/-- The equation of a line perpendicular to x + 4y - 4 = 0 and tangent to y = 2x² --/
theorem perpendicular_tangent_line : 
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, a*x + b*y + c = 0) ∧ 
    (∀ x y : ℝ, x + 4*y - 4 = 0 → (a*1 + b*4 = 0)) ∧
    (∃ x₀ : ℝ, a*x₀ + b*(2*x₀^2) + c = 0 ∧ 
              ∀ x : ℝ, a*x + b*(2*x^2) + c ≥ 0) ∧
    (a = 4 ∧ b = -1 ∧ c = -2) := by
  sorry

end perpendicular_tangent_line_l2161_216177


namespace perpendicular_vectors_m_value_l2161_216150

/-- Given two perpendicular vectors a and b in ℝ², prove that m = 2 -/
theorem perpendicular_vectors_m_value (a b : ℝ × ℝ) (h : a.1 * b.1 + a.2 * b.2 = 0) :
  a = (-2, 3) → b.1 = 3 → b.2 = 2 := by
  sorry

end perpendicular_vectors_m_value_l2161_216150


namespace trigonometric_equality_l2161_216112

theorem trigonometric_equality (α : ℝ) :
  1 + Real.sin (3 * (α + π / 2)) * Real.cos (2 * α) +
  2 * Real.sin (3 * α) * Real.cos (3 * π - α) * Real.sin (α - π) =
  2 * (Real.sin (5 * α / 2))^2 := by sorry

end trigonometric_equality_l2161_216112


namespace optimal_price_achieves_target_profit_l2161_216191

/-- Represents the sales data and profit target for a fruit supermarket --/
structure FruitSales where
  cost_price : ℝ
  initial_price : ℝ
  initial_sales : ℝ
  price_reduction : ℝ
  sales_increase : ℝ
  target_profit : ℝ

/-- Calculates the optimal selling price for the fruit --/
def optimal_selling_price (data : FruitSales) : ℝ :=
  data.initial_price - (data.price_reduction * 3)

/-- Theorem stating that the optimal selling price achieves the target profit --/
theorem optimal_price_achieves_target_profit (data : FruitSales) 
  (h1 : data.cost_price = 22)
  (h2 : data.initial_price = 38)
  (h3 : data.initial_sales = 160)
  (h4 : data.price_reduction = 3)
  (h5 : data.sales_increase = 120)
  (h6 : data.target_profit = 3640) :
  let price := optimal_selling_price data
  let sales := data.initial_sales + data.sales_increase
  let profit_per_kg := price - data.cost_price
  profit_per_kg * sales = data.target_profit ∧ 
  price = 29 :=
by sorry

#eval optimal_selling_price { 
  cost_price := 22, 
  initial_price := 38, 
  initial_sales := 160, 
  price_reduction := 3, 
  sales_increase := 120, 
  target_profit := 3640 
}

end optimal_price_achieves_target_profit_l2161_216191


namespace sum_of_fractions_l2161_216114

theorem sum_of_fractions : 
  (1 / (2 * 3 : ℚ)) + (1 / (3 * 4 : ℚ)) + (1 / (4 * 5 : ℚ)) + (1 / (5 * 6 : ℚ)) + (1 / (6 * 7 : ℚ)) = 5 / 14 := by
  sorry

end sum_of_fractions_l2161_216114


namespace existence_of_n_for_k_l2161_216113

/-- f₂(n) is the number of divisors of n which are perfect squares -/
def f₂ (n : ℕ+) : ℕ := sorry

/-- f₃(n) is the number of divisors of n which are perfect cubes -/
def f₃ (n : ℕ+) : ℕ := sorry

/-- For all positive integers k, there exists a positive integer n such that f₂(n)/f₃(n) = k -/
theorem existence_of_n_for_k (k : ℕ+) : ∃ n : ℕ+, (f₂ n : ℚ) / (f₃ n : ℚ) = k := by sorry

end existence_of_n_for_k_l2161_216113


namespace extreme_values_of_f_l2161_216194

-- Define the function f(x) = x^3 - 3x + 1
def f (x : ℝ) := x^3 - 3*x + 1

-- Define the closed interval [-3, 0]
def I : Set ℝ := Set.Icc (-3) 0

-- Theorem statement
theorem extreme_values_of_f :
  ∃ (a b : ℝ), a ∈ I ∧ b ∈ I ∧
  (∀ x ∈ I, f x ≤ f a) ∧
  (∀ x ∈ I, f x ≥ f b) ∧
  f a = 3 ∧ f b = -17 :=
sorry

end extreme_values_of_f_l2161_216194


namespace prob_two_green_apples_l2161_216139

/-- The probability of selecting two green apples from a set of 8 apples,
    where 4 are green, when choosing 2 apples at random. -/
theorem prob_two_green_apples (total : ℕ) (green : ℕ) (choose : ℕ) 
    (h_total : total = 8) 
    (h_green : green = 4) 
    (h_choose : choose = 2) : 
    Nat.choose green choose / Nat.choose total choose = 3 / 14 := by
  sorry

#check prob_two_green_apples

end prob_two_green_apples_l2161_216139


namespace planes_parallel_from_skew_lines_l2161_216122

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the parallelism relation
variable (parallel : Plane → Plane → Prop)
variable (lineParallelToPlane : Line → Plane → Prop)

-- Define skew lines
variable (skew : Line → Line → Prop)

-- Theorem statement
theorem planes_parallel_from_skew_lines 
  (α β : Plane) (l m : Line) :
  skew l m →
  lineParallelToPlane l α →
  lineParallelToPlane l β →
  lineParallelToPlane m α →
  lineParallelToPlane m β →
  parallel α β :=
sorry

end planes_parallel_from_skew_lines_l2161_216122


namespace system_solution_l2161_216149

theorem system_solution (x y : ℝ) :
  (4 * (Real.cos x)^2 - 4 * Real.cos x * (Real.cos (6 * x))^2 + (Real.cos (6 * x))^2 = 0) ∧
  (Real.sin x = Real.cos y) ↔
  (∃ (k n : ℤ),
    ((x = π / 3 + 2 * π * ↑k ∧ (y = π / 6 + 2 * π * ↑n ∨ y = -π / 6 + 2 * π * ↑n)) ∨
     (x = -π / 3 + 2 * π * ↑k ∧ (y = 5 * π / 6 + 2 * π * ↑n ∨ y = -5 * π / 6 + 2 * π * ↑n)))) :=
by sorry

end system_solution_l2161_216149


namespace rectangle_existence_uniqueness_l2161_216128

theorem rectangle_existence_uniqueness 
  (a b : ℝ) 
  (h_ab : 0 < a ∧ a < b) : 
  ∃! (x y : ℝ), 
    x < a ∧ 
    y < b ∧ 
    2 * (x + y) = a + b ∧ 
    x * y = a * b / 4 := by
  sorry

end rectangle_existence_uniqueness_l2161_216128


namespace integer_solution_exists_iff_n_eq_one_l2161_216123

theorem integer_solution_exists_iff_n_eq_one (n : ℕ+) :
  (∃ x : ℤ, x^(n : ℕ) + (2 + x)^(n : ℕ) + (2 - x)^(n : ℕ) = 0) ↔ n = 1 := by
  sorry

end integer_solution_exists_iff_n_eq_one_l2161_216123


namespace trigonometric_expressions_l2161_216124

theorem trigonometric_expressions :
  (2 * Real.sin (30 * π / 180) + 3 * Real.cos (60 * π / 180) - 4 * Real.tan (45 * π / 180) = -3/2) ∧
  (Real.tan (60 * π / 180) - (4 - π)^0 + 2 * Real.cos (30 * π / 180) + (1/4)⁻¹ = 2 * Real.sqrt 3 + 3) :=
by sorry

end trigonometric_expressions_l2161_216124


namespace unique_solution_l2161_216185

theorem unique_solution (a b c : ℝ) 
  (ha : a > 4) (hb : b > 4) (hc : c > 4)
  (heq : (a + 3)^2 / (b + c - 3) + (b + 5)^2 / (c + a - 5) + (c + 7)^2 / (a + b - 7) = 45) :
  a = 12 ∧ b = 10 ∧ c = 8 := by
  sorry

end unique_solution_l2161_216185


namespace ngon_division_formula_l2161_216121

/-- The number of parts into which the diagonals of an n-gon divide it,
    given that no three diagonals intersect at a single point. -/
def ngon_division (n : ℕ) : ℕ :=
  (n * (n - 1) * (n - 2) * (n - 3)) / 24 + (n * (n - 3)) / 2 + 1

/-- Theorem stating that the number of parts into which the diagonals
    of an n-gon divide it, given that no three diagonals intersect at
    a single point, is equal to the formula derived. -/
theorem ngon_division_formula (n : ℕ) (h : n ≥ 3) :
  ngon_division n = (n * (n - 1) * (n - 2) * (n - 3)) / 24 + (n * (n - 3)) / 2 + 1 :=
by sorry

end ngon_division_formula_l2161_216121


namespace paint_needed_l2161_216166

theorem paint_needed (total_needed existing_paint new_paint : ℕ) 
  (h1 : total_needed = 70)
  (h2 : existing_paint = 36)
  (h3 : new_paint = 23) :
  total_needed - (existing_paint + new_paint) = 11 :=
by
  sorry

end paint_needed_l2161_216166


namespace base_b_problem_l2161_216180

/-- Given that 1325 in base b is equal to the square of 35 in base b, prove that b = 10 in base 10 -/
theorem base_b_problem (b : ℕ) : 
  (3 * b + 5)^2 = b^3 + 3 * b^2 + 2 * b + 5 → b = 10 :=
by sorry

end base_b_problem_l2161_216180


namespace books_read_indeterminate_l2161_216165

/-- Represents the 'crazy silly school' series --/
structure CrazySillySchool where
  total_movies : ℕ
  total_books : ℕ
  movies_watched : ℕ
  movies_left : ℕ

/-- Theorem stating that the number of books read cannot be uniquely determined --/
theorem books_read_indeterminate (series : CrazySillySchool)
  (h1 : series.total_movies = 8)
  (h2 : series.total_books = 21)
  (h3 : series.movies_watched = 4)
  (h4 : series.movies_left = 4) :
  ∀ n : ℕ, n ≤ series.total_books → ∃ m : ℕ, m ≠ n ∧ m ≤ series.total_books :=
by sorry

end books_read_indeterminate_l2161_216165


namespace geometry_propositions_l2161_216154

-- Define the type for planes
variable (Plane : Type)

-- Define the type for lines
variable (Line : Type)

-- Define the relation for two planes being distinct
variable (distinct : Plane → Plane → Prop)

-- Define the relation for two lines intersecting
variable (intersect : Line → Line → Prop)

-- Define the relation for a line being within a plane
variable (within : Line → Plane → Prop)

-- Define the relation for two lines being parallel
variable (parallel_lines : Line → Line → Prop)

-- Define the relation for two planes being parallel
variable (parallel_planes : Plane → Plane → Prop)

-- Define the relation for a line being perpendicular to a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the relation for a line being perpendicular to another line
variable (perp_lines : Line → Line → Prop)

-- Define the relation for two planes intersecting along a line
variable (intersect_along : Plane → Plane → Line → Prop)

-- Define the relation for two planes being perpendicular
variable (perp_planes : Plane → Plane → Prop)

-- Define the relation for a line being outside a plane
variable (outside : Line → Plane → Prop)

theorem geometry_propositions 
  (α β : Plane) 
  (h_distinct : distinct α β) :
  (∀ (l1 l2 m1 m2 : Line), 
    intersect l1 l2 ∧ within l1 α ∧ within l2 α ∧ 
    within m1 β ∧ within m2 β ∧ 
    parallel_lines l1 m1 ∧ parallel_lines l2 m2 → 
    parallel_planes α β) ∧ 
  (∃ (l : Line) (m1 m2 : Line), 
    perp_line_plane l α ∧ 
    within m1 α ∧ within m2 α ∧ intersect m1 m2 ∧ 
    perp_lines l m1 ∧ perp_lines l m2 ∧ 
    ¬(∀ (n : Line), within n α ∧ perp_lines l n → perp_line_plane l α)) ∧
  (∃ (l m : Line), 
    intersect_along α β l ∧ within m α ∧ perp_lines m l ∧ ¬perp_planes α β) ∧
  (∀ (l m : Line), 
    outside l α ∧ within m α ∧ parallel_lines l m → 
    ∀ (n : Line), within n α → ¬intersect l n) :=
by sorry

end geometry_propositions_l2161_216154


namespace sum_of_fractions_equals_four_l2161_216109

theorem sum_of_fractions_equals_four (a b c d : ℝ) 
  (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) 
  (h_eq : a / (b + c + d) = b / (a + c + d) ∧ 
          b / (a + c + d) = c / (a + b + d) ∧ 
          c / (a + b + d) = d / (a + b + c)) : 
  (a + b) / (c + d) + (b + c) / (a + d) + 
  (c + d) / (a + b) + (d + a) / (b + c) = 4 := by
  sorry

end sum_of_fractions_equals_four_l2161_216109


namespace multiplicative_inverse_mod_million_l2161_216153

theorem multiplicative_inverse_mod_million : ∃ N : ℕ, 
  (N > 0) ∧ 
  (N < 1000000) ∧ 
  ((123456 * 769230 * N) % 1000000 = 1) ∧ 
  (N = 1053) := by
  sorry

end multiplicative_inverse_mod_million_l2161_216153


namespace min_value_reciprocal_sum_l2161_216192

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  1 / a + 4 / b ≥ 9 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 1 ∧ 1 / a₀ + 4 / b₀ = 9 :=
by sorry

end min_value_reciprocal_sum_l2161_216192


namespace kids_on_soccer_field_l2161_216189

/-- The number of kids initially on the soccer field -/
def initial_kids : ℕ := 14

/-- The number of kids who joined the soccer field -/
def joined_kids : ℕ := 22

/-- The total number of kids on the soccer field after more kids joined -/
def total_kids : ℕ := initial_kids + joined_kids

theorem kids_on_soccer_field : total_kids = 36 := by
  sorry

end kids_on_soccer_field_l2161_216189


namespace mother_daughter_ages_l2161_216135

/-- Given a mother and daughter where:
    1. The mother is 27 years older than her daughter.
    2. A year ago, the mother was twice as old as her daughter.
    Prove that the mother is 55 years old and the daughter is 28 years old. -/
theorem mother_daughter_ages (mother_age daughter_age : ℕ) 
  (h1 : mother_age = daughter_age + 27)
  (h2 : mother_age - 1 = 2 * (daughter_age - 1)) :
  mother_age = 55 ∧ daughter_age = 28 := by
sorry

end mother_daughter_ages_l2161_216135


namespace lamp_cost_theorem_l2161_216187

-- Define the prices of lamps
def price_A : ℝ := sorry
def price_B : ℝ := sorry

-- Define the total number of lamps
def total_lamps : ℕ := 200

-- Define the function for total cost
def total_cost (a : ℕ) : ℝ := sorry

-- Theorem statement
theorem lamp_cost_theorem :
  -- Conditions
  (3 * price_A + 5 * price_B = 50) ∧
  (price_A + 3 * price_B = 26) ∧
  (∀ a : ℕ, total_cost a = price_A * a + price_B * (total_lamps - a)) →
  -- Conclusions
  (price_A = 5 ∧ price_B = 7) ∧
  (∀ a : ℕ, total_cost a = -2 * a + 1400) ∧
  (total_cost 80 = 1240) := by
  sorry


end lamp_cost_theorem_l2161_216187


namespace ratio_simplification_l2161_216186

theorem ratio_simplification (a b c : ℝ) (n m p : ℕ+) 
  (h : a^(n : ℕ) / c^(p : ℕ) = 3 / 7 ∧ b^(m : ℕ) / c^(p : ℕ) = 4 / 7) :
  (a^(n : ℕ) + b^(m : ℕ) + c^(p : ℕ)) / c^(p : ℕ) = 2 := by
  sorry

end ratio_simplification_l2161_216186


namespace odd_coefficients_in_binomial_expansion_l2161_216116

theorem odd_coefficients_in_binomial_expansion :
  let coefficients := List.range 9 |>.map (fun k => Nat.choose 8 k)
  (coefficients.filter (fun c => c % 2 = 1)).length = 2 := by
  sorry

end odd_coefficients_in_binomial_expansion_l2161_216116


namespace spinner_probability_l2161_216102

theorem spinner_probability : ∀ (p_C : ℚ),
  (1 : ℚ) / 5 + (1 : ℚ) / 3 + p_C + p_C + 2 * p_C = 1 →
  p_C = (7 : ℚ) / 60 := by
  sorry

end spinner_probability_l2161_216102


namespace ellipse_intersection_properties_l2161_216156

-- Define the line and ellipse
def line (x y : ℝ) : Prop := y = -x + 1
def ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the intersection points
def intersectionPoints (a b : ℝ) : Prop := ∃ A B : ℝ × ℝ, 
  line A.1 A.2 ∧ line B.1 B.2 ∧ ellipse A.1 A.2 a b ∧ ellipse B.1 B.2 a b

-- Define eccentricity and focal length
def eccentricity (e : ℝ) : Prop := e = Real.sqrt 3 / 3
def focalLength (c : ℝ) : Prop := c = 1

-- Define perpendicularity of OA and OB
def perpendicular (A B : ℝ × ℝ) : Prop := A.1 * B.1 + A.2 * B.2 = 0

-- Main theorem
theorem ellipse_intersection_properties 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : intersectionPoints a b) :
  (∃ A B : ℝ × ℝ, 
    eccentricity ((a^2 - b^2) / a^2) ∧ 
    focalLength ((a^2 - b^2) / 2) → 
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 * Real.sqrt 3 / 5) ∧
  (∃ A B : ℝ × ℝ,
    perpendicular A B → 
    (1/2 : ℝ) ≤ ((a^2 - b^2) / a^2) ∧ ((a^2 - b^2) / a^2) ≤ Real.sqrt 2 / 2 →
    2 * a ≤ Real.sqrt 6) :=
by sorry

end ellipse_intersection_properties_l2161_216156


namespace colonization_combinations_count_l2161_216168

/-- Represents the number of Earth-like planets -/
def earth_like_planets : ℕ := 6

/-- Represents the number of Mars-like planets -/
def mars_like_planets : ℕ := 7

/-- Represents the resource cost to colonize an Earth-like planet -/
def earth_like_cost : ℕ := 2

/-- Represents the resource cost to colonize a Mars-like planet -/
def mars_like_cost : ℕ := 1

/-- Represents the total available resources -/
def total_resources : ℕ := 14

/-- Calculates the number of ways to select planets for colonization -/
def colonization_combinations : ℕ := sorry

/-- Theorem stating that the number of colonization combinations is 336 -/
theorem colonization_combinations_count :
  colonization_combinations = 336 := by sorry

end colonization_combinations_count_l2161_216168


namespace cassidy_grounding_period_l2161_216178

/-- Calculate the total grounding period for Cassidy --/
theorem cassidy_grounding_period :
  let initial_grounding : ℕ := 14
  let below_b_penalty : ℕ := 3
  let main_below_b : ℕ := 4
  let extra_below_b : ℕ := 2
  let a_grades : ℕ := 2
  let main_penalty := (main_below_b * below_b_penalty ^ 2 : ℚ)
  let extra_penalty := (extra_below_b * (below_b_penalty / 2) ^ 2 : ℚ)
  let additional_days := main_penalty + extra_penalty
  let reduced_initial := initial_grounding - a_grades
  let total_days := reduced_initial + additional_days
  ⌈total_days⌉ = 53 := by sorry

end cassidy_grounding_period_l2161_216178


namespace relay_team_permutations_l2161_216104

def team_size : ℕ := 4
def fixed_runner : String := "Lisa"
def fixed_lap : ℕ := 2

theorem relay_team_permutations :
  let remaining_runners := team_size - 1
  let free_laps := team_size - 1
  (remaining_runners.factorial : ℕ) = 6 := by sorry

end relay_team_permutations_l2161_216104


namespace three_heads_with_tail_probability_l2161_216199

/-- A fair coin flip sequence that ends when either three heads in a row or two tails in a row occur -/
inductive CoinFlipSequence
  | Incomplete : List Bool → CoinFlipSequence
  | ThreeHeads : List Bool → CoinFlipSequence
  | TwoTails : List Bool → CoinFlipSequence

/-- The probability of getting three heads in a row with at least one tail before the third head -/
def probability_three_heads_with_tail : ℚ :=
  5 / 64

/-- The main theorem stating that the calculated probability is correct -/
theorem three_heads_with_tail_probability :
  probability_three_heads_with_tail = 5 / 64 := by
  sorry

end three_heads_with_tail_probability_l2161_216199


namespace problem_statement_l2161_216134

theorem problem_statement (x y : ℚ) (hx : x = 2/3) (hy : y = 3/2) : 
  (1/3) * x^8 * y^9 = 1/2 := by
sorry

end problem_statement_l2161_216134


namespace jerome_toy_cars_l2161_216151

theorem jerome_toy_cars (original : ℕ) : original = 25 :=
  let last_month := 5
  let this_month := 2 * last_month
  let total := 40
  have h : original + last_month + this_month = total := by sorry
  sorry

end jerome_toy_cars_l2161_216151


namespace complement_of_M_l2161_216111

-- Define the universal set U as the set of real numbers
def U := Set ℝ

-- Define the set M
def M : Set ℝ := {x | -1 < x ∧ x ≤ 2}

-- State the theorem
theorem complement_of_M (x : ℝ) : 
  x ∈ (Set.univ \ M) ↔ x ≤ -1 ∨ 2 < x := by
  sorry

end complement_of_M_l2161_216111


namespace f_local_min_g_max_local_min_l2161_216101

noncomputable section

open Real

-- Define the function f(x)
def f (x : ℝ) : ℝ := exp (x - 1) - log x

-- Define the function g(x) parameterized by a
def g (a : ℝ) (x : ℝ) : ℝ := f x - a * (x - 1)

-- Theorem for the local minimum of f(x)
theorem f_local_min : ∃ x₀ : ℝ, x₀ > 0 ∧ IsLocalMin f x₀ ∧ f x₀ = 1 := by sorry

-- Theorem for the maximum of the local minimum of g(x)
theorem g_max_local_min : 
  ∃ a₀ : ℝ, ∀ a : ℝ, 
    (∃ x₀ : ℝ, x₀ > 0 ∧ IsLocalMin (g a) x₀) → 
    (∃ x₁ : ℝ, x₁ > 0 ∧ IsLocalMin (g a₀) x₁ ∧ g a₀ x₁ ≥ g a x₀) ∧
    (∃ x₂ : ℝ, x₂ > 0 ∧ IsLocalMin (g a₀) x₂ ∧ g a₀ x₂ = 1) := by sorry

end

end f_local_min_g_max_local_min_l2161_216101


namespace solve_linear_equation_l2161_216162

theorem solve_linear_equation (x : ℝ) : x + 1 = 4 → x = 3 := by
  sorry

end solve_linear_equation_l2161_216162


namespace pumpkin_pies_sold_l2161_216197

/-- Represents the number of pumpkin pies sold -/
def pumpkin_pies : ℕ := sorry

/-- The number of slices in a pumpkin pie -/
def pumpkin_slices : ℕ := 8

/-- The price of a pumpkin pie slice in cents -/
def pumpkin_price : ℕ := 500

/-- The number of slices in a custard pie -/
def custard_slices : ℕ := 6

/-- The price of a custard pie slice in cents -/
def custard_price : ℕ := 600

/-- The number of custard pies sold -/
def custard_pies : ℕ := 5

/-- The total revenue in cents -/
def total_revenue : ℕ := 34000

theorem pumpkin_pies_sold :
  pumpkin_pies * pumpkin_slices * pumpkin_price +
  custard_pies * custard_slices * custard_price = total_revenue →
  pumpkin_pies = 4 := by
  sorry

end pumpkin_pies_sold_l2161_216197


namespace committee_age_difference_l2161_216126

/-- Proves that the age difference between an old and new member in a committee is 40 years,
    given specific conditions about the committee's average age over time. -/
theorem committee_age_difference (n : ℕ) (A : ℝ) (O N : ℝ) : 
  n = 10 → -- The committee has 10 members
  n * A = n * A + n * 4 - (O - N) → -- The total age after 4 years minus the age difference equals the original total age
  O - N = 40 := by
  sorry

end committee_age_difference_l2161_216126


namespace problem_solution_l2161_216152

theorem problem_solution : 
  let x : ℚ := 5
  let intermediate : ℚ := x * 12 / (180 / 3)
  intermediate + 80 = 81 := by
sorry

end problem_solution_l2161_216152


namespace complex_modulus_problem_l2161_216119

theorem complex_modulus_problem (z : ℂ) (h : 3 + z * Complex.I = z - 3 * Complex.I) : 
  Complex.abs z = 3 := by sorry

end complex_modulus_problem_l2161_216119


namespace tank_dimension_l2161_216137

/-- Given a rectangular tank with dimensions 4, x, and 2 feet, 
    if the total cost to cover its surface with insulation at $20 per square foot is $1520, 
    then x = 5 feet. -/
theorem tank_dimension (x : ℝ) : 
  x > 0 →  -- Ensuring positive dimension
  (12 * x + 16) * 20 = 1520 → 
  x = 5 := by
sorry

end tank_dimension_l2161_216137


namespace percentage_study_both_math_and_sociology_l2161_216179

theorem percentage_study_both_math_and_sociology :
  ∀ (S : ℕ) (So Ma Bi MaSo : ℕ),
    S = 200 →
    So = (56 * S) / 100 →
    Ma = (44 * S) / 100 →
    Bi = (40 * S) / 100 →
    Bi - (S - So - Ma + MaSo) ≤ 60 →
    MaSo ≤ Bi - 60 →
    (MaSo * 100) / S = 10 :=
by sorry

end percentage_study_both_math_and_sociology_l2161_216179


namespace quadratic_function_properties_l2161_216182

/-- A quadratic function f(x) = ax^2 + 2bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  h_order : c > b ∧ b > a

/-- The graph of f passes through (1, 0) -/
def passes_through_one_zero (f : QuadraticFunction) : Prop :=
  f.a + 2 * f.b + f.c = 0

/-- The graph of f intersects with y = -a -/
def intersects_neg_a (f : QuadraticFunction) : Prop :=
  ∃ x : ℝ, f.a * x^2 + 2 * f.b * x + f.c = -f.a

/-- The ratio b/a is in [0, 1) -/
def ratio_in_range (f : QuadraticFunction) : Prop :=
  0 ≤ f.b / f.a ∧ f.b / f.a < 1

/-- Line segments AB, BC, CD form an obtuse triangle -/
def forms_obtuse_triangle (A B C D : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let CD := Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)
  AB + CD > BC ∧ AB^2 + CD^2 < BC^2

/-- The ratio b/a is in the specified range -/
def ratio_in_specific_range (f : QuadraticFunction) : Prop :=
  -1 + 4/21 < f.b / f.a ∧ f.b / f.a < -1 + Real.sqrt 15 / 3

theorem quadratic_function_properties (f : QuadraticFunction)
    (h_pass : passes_through_one_zero f)
    (h_intersect : intersects_neg_a f) :
  ratio_in_range f ∧
  (∀ A B C D : ℝ × ℝ, forms_obtuse_triangle A B C D →
    ratio_in_specific_range f) :=
  sorry

end quadratic_function_properties_l2161_216182


namespace complex_modulus_problem_l2161_216164

theorem complex_modulus_problem (z : ℂ) : (1 + Complex.I) * z = 2 * Complex.I → Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_modulus_problem_l2161_216164


namespace inequality_solution_l2161_216161

theorem inequality_solution (a b c d : ℝ) : 
  (∀ x : ℝ, ((x - a) * (x - b) * (x - c)) / (x - d) ≤ 0 ↔ 
    (x < -4 ∨ (1 ≤ x ∧ x ≤ 5) ∨ (24 ≤ x ∧ x ≤ 26))) →
  a < b →
  b < c →
  a + 3*b + 3*c + 4*d = 72 := by
sorry


end inequality_solution_l2161_216161


namespace not_always_true_false_and_implies_true_or_l2161_216175

theorem not_always_true_false_and_implies_true_or : 
  ¬ ∀ (p q : Prop), (¬(p ∧ q)) → (p ∨ q) := by
  sorry

end not_always_true_false_and_implies_true_or_l2161_216175


namespace polynomial_factorization_l2161_216118

theorem polynomial_factorization (a x : ℝ) : 2*a*x^2 - 12*a*x + 18*a = 2*a*(x-3)^2 := by
  sorry

end polynomial_factorization_l2161_216118


namespace pentagon_triangle_angle_sum_l2161_216159

/-- The sum of the interior angle of a regular pentagon and the interior angle of a regular triangle is 168°. -/
theorem pentagon_triangle_angle_sum : 
  let pentagon_angle : ℝ := 180 * (5 - 2) / 5
  let triangle_angle : ℝ := 180 * (3 - 2) / 3
  pentagon_angle + triangle_angle = 168 := by
  sorry

end pentagon_triangle_angle_sum_l2161_216159


namespace unique_valid_arrangement_l2161_216147

/-- Represents the positions in the hexagon --/
inductive Position
| A | B | C | D | E | F

/-- Represents a line in the hexagon --/
structure Line where
  p1 : Position
  p2 : Position
  p3 : Position

/-- The arrangement of digits in the hexagon --/
def Arrangement := Position → Fin 6

/-- The 7 lines in the hexagon --/
def lines : List Line := [
  ⟨Position.A, Position.B, Position.C⟩,
  ⟨Position.A, Position.D, Position.F⟩,
  ⟨Position.A, Position.E, Position.F⟩,
  ⟨Position.B, Position.C, Position.D⟩,
  ⟨Position.B, Position.E, Position.D⟩,
  ⟨Position.C, Position.E, Position.F⟩,
  ⟨Position.D, Position.E, Position.F⟩
]

/-- Check if an arrangement is valid --/
def isValidArrangement (arr : Arrangement) : Prop :=
  (∀ p : Position, arr p ∈ [1, 2, 3, 4, 5, 6]) ∧
  (∀ p q : Position, p ≠ q → arr p ≠ arr q) ∧
  (∀ l : Line, (arr l.p1).val + (arr l.p2).val + (arr l.p3).val = 15)

/-- The unique valid arrangement --/
def uniqueArrangement : Arrangement :=
  fun p => match p with
  | Position.A => 4
  | Position.B => 1
  | Position.C => 2
  | Position.D => 5
  | Position.E => 6
  | Position.F => 3

theorem unique_valid_arrangement :
  isValidArrangement uniqueArrangement ∧
  (∀ arr : Arrangement, isValidArrangement arr → arr = uniqueArrangement) := by
  sorry


end unique_valid_arrangement_l2161_216147


namespace min_workers_for_job_l2161_216167

/-- Represents a construction job with workers -/
structure ConstructionJob where
  totalDays : ℕ
  elapsedDays : ℕ
  initialWorkers : ℕ
  completedPortion : ℚ
  
/-- Calculates the minimum number of workers needed to complete the job on time -/
def minWorkersNeeded (job : ConstructionJob) : ℕ :=
  job.initialWorkers

/-- Theorem stating that for the given job specifications, 
    the minimum number of workers needed is 10 -/
theorem min_workers_for_job :
  let job := ConstructionJob.mk 40 10 10 (1/4)
  minWorkersNeeded job = 10 := by
  sorry

end min_workers_for_job_l2161_216167


namespace non_pine_trees_l2161_216131

theorem non_pine_trees (total : ℕ) (pine_percentage : ℚ) (non_pine : ℕ) : 
  total = 350 → pine_percentage = 70 / 100 → 
  non_pine = total - (pine_percentage * total).floor → non_pine = 105 :=
by sorry

end non_pine_trees_l2161_216131


namespace largest_three_digit_congruence_l2161_216190

theorem largest_three_digit_congruence :
  ∃ (n : ℕ), n = 998 ∧ 
  n < 1000 ∧ 
  n > 99 ∧
  70 * n % 350 = 210 % 350 ∧
  ∀ (m : ℕ), m < 1000 → m > 99 → 70 * m % 350 = 210 % 350 → m ≤ n :=
by sorry

end largest_three_digit_congruence_l2161_216190


namespace min_value_of_function_l2161_216120

theorem min_value_of_function (x : ℝ) (h : x > -1) :
  (x + 5) * (x + 2) / (x + 1) ≥ 9 ∧
  (x + 5) * (x + 2) / (x + 1) = 9 ↔ x = 1 :=
by sorry

end min_value_of_function_l2161_216120


namespace prob_diana_wins_is_half_l2161_216174

/-- Diana's die has 8 sides -/
def diana_sides : ℕ := 8

/-- Apollo's die has 6 sides -/
def apollo_sides : ℕ := 6

/-- The set of possible outcomes for Diana -/
def diana_outcomes : Finset ℕ := Finset.range diana_sides

/-- The set of possible outcomes for Apollo -/
def apollo_outcomes : Finset ℕ := Finset.range apollo_sides

/-- The set of even outcomes for Apollo -/
def apollo_even_outcomes : Finset ℕ := Finset.filter (fun n => n % 2 = 0) apollo_outcomes

/-- The probability that Diana rolls a number larger than Apollo, given that Apollo's number is even -/
def prob_diana_wins_given_apollo_even : ℚ :=
  let total_outcomes := (apollo_even_outcomes.card * diana_outcomes.card : ℚ)
  let favorable_outcomes := (apollo_even_outcomes.sum fun a =>
    (diana_outcomes.filter (fun d => d > a)).card : ℚ)
  favorable_outcomes / total_outcomes

/-- The main theorem: The probability is 1/2 -/
theorem prob_diana_wins_is_half : prob_diana_wins_given_apollo_even = 1/2 := by
  sorry


end prob_diana_wins_is_half_l2161_216174


namespace function_minimum_value_l2161_216198

theorem function_minimum_value (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  ((x^2) / (y - 2) + (y^2) / (x - 2)) ≥ 12 :=
by sorry

end function_minimum_value_l2161_216198


namespace soccer_substitutions_modulo_l2161_216188

def num_players : ℕ := 22
def starting_players : ℕ := 11
def max_substitutions : ℕ := 4

def substitution_ways : ℕ → ℕ
| 0 => 1
| 1 => starting_players * starting_players
| n+1 => substitution_ways n * (starting_players - n) * (starting_players - n)

def total_substitution_ways : ℕ := 
  (List.range (max_substitutions + 1)).map substitution_ways |> List.sum

theorem soccer_substitutions_modulo :
  total_substitution_ways % 1000 = 722 := by sorry

end soccer_substitutions_modulo_l2161_216188


namespace binomial_sum_inequality_l2161_216108

theorem binomial_sum_inequality (x : ℝ) (n : ℕ) (h1 : |x| < 1) (h2 : n ≥ 2) :
  (1 - x)^n + (1 + x)^n < 2^n := by
  sorry

end binomial_sum_inequality_l2161_216108


namespace range_of_ratio_l2161_216125

theorem range_of_ratio (x y : ℝ) (h : x^2 + y^2 - 4*x + 1 = 0) : 
  ∃ (k : ℝ), k = |y / (x + 1)| ∧ k ≤ Real.sqrt 2 / 2 := by
  sorry

end range_of_ratio_l2161_216125


namespace rain_duration_theorem_l2161_216170

def rain_duration_day1 : ℕ := 10

def rain_duration_day2 (d1 : ℕ) : ℕ := d1 + 2

def rain_duration_day3 (d2 : ℕ) : ℕ := 2 * d2

def total_rain_duration (d1 d2 d3 : ℕ) : ℕ := d1 + d2 + d3

theorem rain_duration_theorem :
  total_rain_duration rain_duration_day1
    (rain_duration_day2 rain_duration_day1)
    (rain_duration_day3 (rain_duration_day2 rain_duration_day1)) = 46 := by
  sorry

end rain_duration_theorem_l2161_216170


namespace problem_solution_l2161_216140

theorem problem_solution (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ) 
  (h1 : x₁ + 4*x₂ + 9*x₃ + 16*x₄ + 25*x₅ + 36*x₆ + 49*x₇ = 5)
  (h2 : 4*x₁ + 9*x₂ + 16*x₃ + 25*x₄ + 36*x₅ + 49*x₆ + 64*x₇ = 20)
  (h3 : 9*x₁ + 16*x₂ + 25*x₃ + 36*x₄ + 49*x₅ + 64*x₆ + 81*x₇ = 145) :
  16*x₁ + 25*x₂ + 36*x₃ + 49*x₄ + 64*x₅ + 81*x₆ + 100*x₇ = 380 := by
  sorry

end problem_solution_l2161_216140


namespace junior_score_l2161_216130

theorem junior_score (n : ℝ) (junior_score : ℝ) : 
  n > 0 →
  0.2 * n * junior_score + 0.8 * n * 84 = n * 85 →
  junior_score = 89 := by
sorry

end junior_score_l2161_216130


namespace max_balls_in_cube_l2161_216195

theorem max_balls_in_cube (cube_side : ℝ) (ball_radius : ℝ) : 
  cube_side = 10 → 
  ball_radius = 3 → 
  ⌊(cube_side ^ 3) / ((4 / 3) * Real.pi * ball_radius ^ 3)⌋ = 8 := by
sorry

end max_balls_in_cube_l2161_216195


namespace box_comparison_l2161_216141

-- Define a structure for a box with three dimensions
structure Box where
  x : ℕ
  y : ℕ
  z : ℕ

-- Define the relation "smaller than" for boxes
def smaller (k p : Box) : Prop :=
  (k.x ≤ p.x ∧ k.y ≤ p.y ∧ k.z ≤ p.z) ∨
  (k.x ≤ p.x ∧ k.y ≤ p.z ∧ k.z ≤ p.y) ∨
  (k.x ≤ p.y ∧ k.y ≤ p.x ∧ k.z ≤ p.z) ∨
  (k.x ≤ p.y ∧ k.y ≤ p.z ∧ k.z ≤ p.x) ∨
  (k.x ≤ p.z ∧ k.y ≤ p.x ∧ k.z ≤ p.y) ∨
  (k.x ≤ p.z ∧ k.y ≤ p.y ∧ k.z ≤ p.x)

-- Define boxes A, B, and C
def A : Box := ⟨5, 6, 3⟩
def B : Box := ⟨1, 5, 4⟩
def C : Box := ⟨2, 2, 3⟩

-- Theorem to prove A > B and C < A
theorem box_comparison : smaller B A ∧ smaller C A := by
  sorry


end box_comparison_l2161_216141


namespace pages_to_read_thursday_l2161_216184

def book_pages : ℕ := 158
def monday_pages : ℕ := 23
def tuesday_pages : ℕ := 38
def wednesday_pages : ℕ := 61

theorem pages_to_read_thursday (thursday_pages : ℕ) : 
  thursday_pages = 12 ↔ 
  ∃ (friday_pages : ℕ),
    friday_pages = 2 * thursday_pages ∧
    monday_pages + tuesday_pages + wednesday_pages + thursday_pages + friday_pages = book_pages :=
by sorry

end pages_to_read_thursday_l2161_216184


namespace number_difference_proof_l2161_216181

theorem number_difference_proof (x : ℝ) : x - (3 / 5) * x = 50 → x = 125 := by
  sorry

end number_difference_proof_l2161_216181


namespace candy_bar_profit_l2161_216105

/-- Represents the profit calculation for a candy bar sale --/
theorem candy_bar_profit :
  let total_bars : ℕ := 1200
  let buy_price : ℚ := 5 / 6  -- Price per bar when buying
  let sell_price : ℚ := 2 / 3  -- Price per bar when selling
  let cost : ℚ := total_bars * buy_price
  let revenue : ℚ := total_bars * sell_price
  let profit : ℚ := revenue - cost
  profit = -200
:= by sorry

end candy_bar_profit_l2161_216105


namespace storage_volume_calculation_l2161_216129

/-- Converts cubic yards to cubic feet -/
def cubic_yards_to_cubic_feet (yards : ℝ) : ℝ := 27 * yards

/-- Calculates the total volume in cubic feet -/
def total_volume (initial_yards : ℝ) (additional_feet : ℝ) : ℝ :=
  cubic_yards_to_cubic_feet initial_yards + additional_feet

/-- Theorem: The total volume is 180 cubic feet -/
theorem storage_volume_calculation :
  total_volume 5 45 = 180 := by
  sorry

end storage_volume_calculation_l2161_216129


namespace palindrome_with_five_percentage_l2161_216132

/-- A function that checks if a natural number is a palindrome -/
def isPalindrome (n : ℕ) : Bool :=
  sorry

/-- A function that checks if a natural number contains the digit 5 -/
def containsFive (n : ℕ) : Bool :=
  sorry

/-- The set of palindromes between 100 and 1000 (inclusive) -/
def palindromes : Finset ℕ :=
  sorry

/-- The set of palindromes between 100 and 1000 (inclusive) containing at least one 5 -/
def palindromesWithFive : Finset ℕ :=
  sorry

theorem palindrome_with_five_percentage :
  (palindromesWithFive.card : ℚ) / palindromes.card * 100 = 37 / 180 * 100 :=
sorry

end palindrome_with_five_percentage_l2161_216132


namespace sequence_theorem_l2161_216148

def sequence_property (a : ℕ → ℝ) (S : ℕ → ℝ) (r p : ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → r * (n - p) * S (n + 1) = n^2 * a n + (n^2 - n - 2) * a 1

theorem sequence_theorem (a : ℕ → ℝ) (S : ℕ → ℝ) (r p : ℝ) 
  (h1 : |a 1| ≠ |a 2|)
  (h2 : r ≠ 0)
  (h3 : sequence_property a S r p) :
  (p = 1) ∧ 
  (¬ ∃ k : ℝ, k ≠ 1 ∧ k ≠ -1 ∧ ∀ n : ℕ, n > 0 → a (n + 1) = k * a n) ∧
  (r = 2 → ∃ d : ℝ, ∀ n : ℕ, n > 0 → a (n + 1) = a n + d) :=
by sorry

end sequence_theorem_l2161_216148


namespace triangle_construction_uniqueness_l2161_216115

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by its three vertices -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- The orthocenter of a triangle -/
def orthocenter (t : Triangle) : Point :=
  sorry

/-- The point where the internal angle bisector from A intersects BC -/
def internalBisectorIntersection (t : Triangle) : Point :=
  sorry

/-- The point where the external angle bisector from A intersects BC -/
def externalBisectorIntersection (t : Triangle) : Point :=
  sorry

/-- Predicate to check if a point is within the valid region for M -/
def isValidM (M A' A'' : Point) : Prop :=
  sorry

theorem triangle_construction_uniqueness 
  (M A' A'' : Point) 
  (h_valid : isValidM M A' A'') :
  ∃! t : Triangle, 
    orthocenter t = M ∧ 
    internalBisectorIntersection t = A' ∧ 
    externalBisectorIntersection t = A'' :=
  sorry

end triangle_construction_uniqueness_l2161_216115


namespace complex_number_theorem_l2161_216142

theorem complex_number_theorem (a : ℝ) (z : ℂ) (h1 : z = (a^2 - 1) + (a + 1) * I) 
  (h2 : z.re = 0) : (a + I^2016) / (1 + I) = 1 - I :=
by
  sorry

end complex_number_theorem_l2161_216142


namespace stock_value_change_l2161_216110

/-- Calculates the net percentage change in stock value over three years --/
def netPercentageChange (year1Change year2Change year3Change dividend : ℝ) : ℝ :=
  let value1 := (1 + year1Change) * (1 + dividend)
  let value2 := value1 * (1 + year2Change) * (1 + dividend)
  let value3 := value2 * (1 + year3Change) * (1 + dividend)
  (value3 - 1) * 100

/-- The net percentage change in stock value is approximately 17.52% --/
theorem stock_value_change :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧
  |netPercentageChange (-0.08) 0.10 0.06 0.03 - 17.52| < ε :=
sorry

end stock_value_change_l2161_216110


namespace automobile_finance_credit_l2161_216176

/-- The problem statement as a theorem --/
theorem automobile_finance_credit (total_credit : ℝ) (auto_credit_percentage : ℝ) (finance_company_fraction : ℝ) : 
  total_credit = 416.6666666666667 →
  auto_credit_percentage = 0.36 →
  finance_company_fraction = 0.5 →
  finance_company_fraction * (auto_credit_percentage * total_credit) = 75 := by
sorry

end automobile_finance_credit_l2161_216176


namespace ellipse_sum_l2161_216100

/-- For an ellipse with center (h, k), semi-major axis length a, and semi-minor axis length b,
    prove that h + k + a + b = 4 when the center is (3, -5), a = 4, and b = 2. -/
theorem ellipse_sum (h k a b : ℝ) : 
  h = 3 → k = -5 → a = 4 → b = 2 → h + k + a + b = 4 := by
  sorry

end ellipse_sum_l2161_216100


namespace sum_of_coefficients_l2161_216193

def polynomial (x : ℝ) : ℝ := -2 * (x^7 - x^4 + 3*x^2 - 5) + 4*(x^3 + 2*x) - 3*(x^5 - 4)

theorem sum_of_coefficients : 
  (polynomial 1) = 25 := by sorry

end sum_of_coefficients_l2161_216193


namespace quadratic_polynomial_prime_values_l2161_216171

/-- A quadratic polynomial with integer coefficients -/
def QuadraticPolynomial := ℤ → ℤ

/-- Predicate to check if a number is prime -/
def IsPrime (n : ℤ) : Prop := sorry

/-- Predicate to check if a polynomial takes prime values at three consecutive integer points -/
def TakesPrimeValuesAtThreeConsecutivePoints (f : QuadraticPolynomial) : Prop :=
  ∃ n : ℤ, IsPrime (f (n - 1)) ∧ IsPrime (f n) ∧ IsPrime (f (n + 1))

/-- Predicate to check if a polynomial takes a prime value at least at one more integer point -/
def TakesPrimeValueAtOneMorePoint (f : QuadraticPolynomial) : Prop :=
  ∃ m : ℤ, (∀ n : ℤ, m ≠ n - 1 ∧ m ≠ n ∧ m ≠ n + 1) → IsPrime (f m)

/-- Theorem stating that if a quadratic polynomial with integer coefficients takes prime values
    at three consecutive integer points, then it takes a prime value at least at one more integer point -/
theorem quadratic_polynomial_prime_values (f : QuadraticPolynomial) :
  TakesPrimeValuesAtThreeConsecutivePoints f → TakesPrimeValueAtOneMorePoint f :=
sorry

end quadratic_polynomial_prime_values_l2161_216171


namespace mothers_age_l2161_216196

theorem mothers_age (certain_age : ℕ) (mothers_age : ℕ) : 
  mothers_age = 3 * certain_age → 
  certain_age + mothers_age = 40 → 
  mothers_age = 30 := by
sorry

end mothers_age_l2161_216196


namespace units_digit_of_m_squared_plus_3_to_m_l2161_216163

def m : ℕ := 2011^2 + 3^2011

theorem units_digit_of_m_squared_plus_3_to_m (m : ℕ := 2011^2 + 3^2011) : 
  (m^2 + 3^m) % 10 = 5 := by
  sorry

end units_digit_of_m_squared_plus_3_to_m_l2161_216163


namespace child_patients_per_hour_l2161_216127

/-- Represents the number of adult patients seen per hour -/
def adults_per_hour : ℕ := 4

/-- Represents the cost of an adult office visit in dollars -/
def adult_visit_cost : ℕ := 50

/-- Represents the cost of a child office visit in dollars -/
def child_visit_cost : ℕ := 25

/-- Represents the total revenue for a typical 8-hour day in dollars -/
def total_daily_revenue : ℕ := 2200

/-- Represents the number of hours in a typical workday -/
def hours_per_day : ℕ := 8

/-- 
Proves that the number of child patients seen per hour is 3, 
given the conditions specified in the problem.
-/
theorem child_patients_per_hour : 
  ∃ (c : ℕ), 
    hours_per_day * (adults_per_hour * adult_visit_cost + c * child_visit_cost) = total_daily_revenue ∧
    c = 3 := by
  sorry

end child_patients_per_hour_l2161_216127


namespace sandys_marks_per_correct_sum_l2161_216106

/-- Given Sandy's quiz results, calculate the marks for each correct sum -/
theorem sandys_marks_per_correct_sum 
  (total_sums : ℕ) 
  (correct_sums : ℕ) 
  (total_marks : ℤ) 
  (penalty_per_incorrect : ℕ) 
  (h1 : total_sums = 30) 
  (h2 : correct_sums = 23) 
  (h3 : total_marks = 55) 
  (h4 : penalty_per_incorrect = 2) :
  ∃ (marks_per_correct : ℕ), 
    marks_per_correct * correct_sums - 
    penalty_per_incorrect * (total_sums - correct_sums) = total_marks ∧ 
    marks_per_correct = 3 := by
  sorry

end sandys_marks_per_correct_sum_l2161_216106


namespace tan_sum_from_sin_cos_sum_l2161_216107

theorem tan_sum_from_sin_cos_sum (x y : ℝ) 
  (h1 : Real.sin x + Real.sin y = 1/2)
  (h2 : Real.cos x + Real.cos y = Real.sqrt 3 / 2) :
  Real.tan x + Real.tan y = - Real.sqrt 3 := by
  sorry

end tan_sum_from_sin_cos_sum_l2161_216107


namespace product_of_sums_equals_power_specific_product_equals_power_l2161_216145

theorem product_of_sums_equals_power (a b : ℕ) :
  (a + b) * (a^2 + b^2) * (a^4 + b^4) * (a^8 + b^8) * 
  (a^16 + b^16) * (a^32 + b^32) * (a^64 + b^64) = (a + b)^127 :=
by
  sorry

theorem specific_product_equals_power :
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * 
  (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) = 9^127 :=
by
  sorry

end product_of_sums_equals_power_specific_product_equals_power_l2161_216145


namespace circle_center_on_line_l2161_216173

theorem circle_center_on_line (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 2*a*x + 4*y - 6 = 0 → 
    ∃ h k : ℝ, (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 2*a*h + 4*k - 6) ∧ 
    h + 2*k + 1 = 0) →
  a = 3 := by
sorry

end circle_center_on_line_l2161_216173


namespace smallest_n_for_sock_arrangement_l2161_216146

theorem smallest_n_for_sock_arrangement : 
  (∃ n : ℕ, n > 0 ∧ (n + 1) * (n + 2) / 2 > 1000000 ∧ 
   ∀ m : ℕ, m > 0 → (m + 1) * (m + 2) / 2 > 1000000 → m ≥ n) →
  (∃ n : ℕ, n > 0 ∧ (n + 1) * (n + 2) / 2 > 1000000 ∧ 
   ∀ m : ℕ, m > 0 → (m + 1) * (m + 2) / 2 > 1000000 → m ≥ n ∧ n = 1413) :=
by sorry

end smallest_n_for_sock_arrangement_l2161_216146


namespace two_over_a_necessary_not_sufficient_l2161_216157

theorem two_over_a_necessary_not_sufficient (a : ℝ) (h : a ≠ 0) :
  (∀ a, a ^ 2 > 4 → 2 / a < 1) ∧
  (∃ a, 2 / a < 1 ∧ a ^ 2 ≤ 4) :=
by sorry

end two_over_a_necessary_not_sufficient_l2161_216157


namespace arithmetic_sequence_sum_l2161_216103

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 3 + a 4 + a 5 + a 6 + a 7 = 450 →
  a 2 + a 8 = 180 := by
  sorry

end arithmetic_sequence_sum_l2161_216103


namespace robert_reading_theorem_l2161_216144

/-- Calculates the maximum number of complete books that can be read given the reading speed, book length, and available time. -/
def max_complete_books_read (reading_speed : ℕ) (book_length : ℕ) (available_time : ℕ) : ℕ :=
  (available_time * reading_speed) / book_length

/-- Theorem: Given Robert's reading speed of 120 pages per hour, the maximum number of complete 360-page books he can read in 8 hours is 2. -/
theorem robert_reading_theorem : 
  max_complete_books_read 120 360 8 = 2 := by
  sorry

end robert_reading_theorem_l2161_216144


namespace initial_liquid_a_amount_l2161_216155

/-- Given a mixture of liquids A and B with an initial ratio and a replacement process,
    calculate the initial amount of liquid A. -/
theorem initial_liquid_a_amount
  (initial_ratio_a : ℚ)
  (initial_ratio_b : ℚ)
  (replacement_amount : ℚ)
  (final_ratio_a : ℚ)
  (final_ratio_b : ℚ)
  (h_initial_ratio : initial_ratio_a / initial_ratio_b = 4 / 1)
  (h_replacement : replacement_amount = 20)
  (h_final_ratio : final_ratio_a / final_ratio_b = 2 / 3)
  : initial_ratio_a * (initial_ratio_a + initial_ratio_b) / (initial_ratio_a + initial_ratio_b) = 16 := by
  sorry


end initial_liquid_a_amount_l2161_216155


namespace weight_of_new_person_l2161_216160

theorem weight_of_new_person (n : ℕ) (avg_increase : ℝ) (replaced_weight : ℝ) :
  n = 5 →
  avg_increase = 1.5 →
  replaced_weight = 65 →
  ∃ (new_weight : ℝ), new_weight = 72.5 ∧
    n * avg_increase = new_weight - replaced_weight :=
by sorry

end weight_of_new_person_l2161_216160


namespace gcf_of_75_and_100_l2161_216143

theorem gcf_of_75_and_100 : Nat.gcd 75 100 = 25 := by
  sorry

end gcf_of_75_and_100_l2161_216143


namespace only_event3_mutually_exclusive_l2161_216138

-- Define the set of numbers
def numbers : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the concept of odd and even numbers
def isOdd (n : Nat) : Prop := n % 2 = 1
def isEven (n : Nat) : Prop := n % 2 = 0

-- Define the events
def event1 (a b : Nat) : Prop := (isOdd a ∧ isEven b) ∨ (isEven a ∧ isOdd b)
def event2 (a b : Nat) : Prop := isOdd a ∨ isOdd b
def event3 (a b : Nat) : Prop := (isOdd a ∨ isOdd b) ∧ (isEven a ∧ isEven b)
def event4 (a b : Nat) : Prop := (isOdd a ∨ isOdd b) ∧ (isEven a ∨ isEven b)

-- Define mutual exclusivity
def mutuallyExclusive (e1 e2 : (Nat → Nat → Prop)) : Prop :=
  ∀ a b, a ∈ numbers → b ∈ numbers → ¬(e1 a b ∧ e2 a b)

-- Theorem statement
theorem only_event3_mutually_exclusive :
  (mutuallyExclusive event1 event3) ∧
  (¬mutuallyExclusive event1 event1) ∧
  (¬mutuallyExclusive event2 event4) ∧
  (¬mutuallyExclusive event4 event4) :=
sorry


end only_event3_mutually_exclusive_l2161_216138


namespace triangle_side_length_l2161_216136

theorem triangle_side_length (a c : ℝ) (B : ℝ) (h1 : a = 2) (h2 : c = 2 * Real.sqrt 3) (h3 : B = π / 6) :
  ∃ b : ℝ, b ^ 2 = a ^ 2 + c ^ 2 - 2 * a * c * Real.cos B ∧ b = 2 := by
  sorry

end triangle_side_length_l2161_216136


namespace set_intersection_equality_l2161_216169

def M : Set ℝ := {x | (2 - x) / (x + 1) ≥ 0}
def N : Set ℝ := {x | ∃ y, y = Real.log x}

theorem set_intersection_equality : M ∩ N = Set.Ioo 0 2 := by
  sorry

end set_intersection_equality_l2161_216169
