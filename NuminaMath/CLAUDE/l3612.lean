import Mathlib

namespace gcd_of_powers_of_two_l3612_361240

theorem gcd_of_powers_of_two : Nat.gcd (2^1010 - 1) (2^1000 - 1) = 2^10 - 1 := by
  sorry

end gcd_of_powers_of_two_l3612_361240


namespace pairwise_product_inequality_l3612_361281

theorem pairwise_product_inequality (a b c : ℕ+) : 
  (a * b : ℕ) + (b * c : ℕ) + (a * c : ℕ) ≤ 3 * (a * b * c : ℕ) := by
  sorry

end pairwise_product_inequality_l3612_361281


namespace unique_solution_l3612_361229

theorem unique_solution (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (eq1 : x*y + y*z + z*x = 12)
  (eq2 : x*y*z = 2 + x + y + z) :
  x = 2 ∧ y = 2 ∧ z = 2 := by
sorry

end unique_solution_l3612_361229


namespace pool_capacity_theorem_l3612_361202

/-- Represents the dimensions of a pool -/
structure PoolDimensions where
  width : ℝ
  length : ℝ
  depth : ℝ

/-- Calculates the volume of a pool given its dimensions -/
def poolVolume (d : PoolDimensions) : ℝ := d.width * d.length * d.depth

/-- Represents the draining parameters of a pool -/
structure DrainParameters where
  rate : ℝ
  time : ℝ

/-- Calculates the amount of water drained given drain parameters -/
def waterDrained (p : DrainParameters) : ℝ := p.rate * p.time

/-- Theorem: The initial capacity of the pool was 80% of its total volume -/
theorem pool_capacity_theorem (d : PoolDimensions) (p : DrainParameters) :
  d.width = 60 ∧ d.length = 150 ∧ d.depth = 10 ∧
  p.rate = 60 ∧ p.time = 1200 →
  waterDrained p / poolVolume d = 0.8 := by
  sorry

#eval (80 : ℚ) / 100  -- Expected output: 4/5

end pool_capacity_theorem_l3612_361202


namespace minesweeper_sum_invariant_l3612_361262

/-- Represents a cell in the Minesweeper grid -/
inductive Cell
| Mine : Cell
| Number (n : ℕ) : Cell

/-- A 10x10 Minesweeper grid -/
def MinesweeperGrid := Fin 10 → Fin 10 → Cell

/-- Calculates the sum of all numbers in a Minesweeper grid -/
def gridSum (grid : MinesweeperGrid) : ℕ := sorry

/-- Flips the state of all cells in a Minesweeper grid -/
def flipGrid (grid : MinesweeperGrid) : MinesweeperGrid := sorry

/-- Theorem stating that the sum of numbers remains constant after flipping the grid -/
theorem minesweeper_sum_invariant (grid : MinesweeperGrid) : 
  gridSum grid = gridSum (flipGrid grid) := by sorry

end minesweeper_sum_invariant_l3612_361262


namespace set_relationship_l3612_361268

theorem set_relationship (A B C : Set α) (hAnonempty : A.Nonempty) (hBnonempty : B.Nonempty) (hCnonempty : C.Nonempty)
  (hUnion : A ∪ B = C) (hNotSubset : ¬(B ⊆ A)) :
  (∀ x, x ∈ A → x ∈ C) ∧ ¬(∀ x, x ∈ C → x ∈ A) := by
sorry

end set_relationship_l3612_361268


namespace equation_solution_l3612_361277

theorem equation_solution (x : ℝ) (h : x ≥ 0) :
  (2021 * x = 2022 * (x^(2021/2022)) - 1) ↔ (x = 1) :=
sorry

end equation_solution_l3612_361277


namespace temperature_drop_l3612_361287

theorem temperature_drop (initial_temp final_temp drop : ℤ) :
  initial_temp = -6 ∧ drop = 5 → final_temp = initial_temp - drop → final_temp = -11 := by
  sorry

end temperature_drop_l3612_361287


namespace line_perp_parallel_planes_l3612_361216

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (parallel : Plane → Plane → Prop)

-- Theorem statement
theorem line_perp_parallel_planes 
  (a : Line) (α β : Plane) 
  (h1 : perpendicular a α) 
  (h2 : parallel α β) : 
  perpendicular a β :=
sorry

end line_perp_parallel_planes_l3612_361216


namespace max_slope_product_l3612_361250

theorem max_slope_product (m₁ m₂ : ℝ) : 
  m₁ ≠ 0 → m₂ ≠ 0 →                           -- non-horizontal, non-vertical lines
  |((m₂ - m₁) / (1 + m₁ * m₂))| = 1 →         -- 45° angle intersection
  m₂ = 6 * m₁ →                               -- one slope is 6 times the other
  ∃ (p : ℝ), p = m₁ * m₂ ∧ p ≤ (3/2 : ℝ) ∧ 
  ∀ (q : ℝ), (∃ (n₁ n₂ : ℝ), n₁ ≠ 0 ∧ n₂ ≠ 0 ∧ 
              |((n₂ - n₁) / (1 + n₁ * n₂))| = 1 ∧ 
              n₂ = 6 * n₁ ∧ q = n₁ * n₂) → q ≤ p :=
by sorry

end max_slope_product_l3612_361250


namespace numbers_satisfying_conditions_l3612_361256

def satisfies_conditions (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 1000 ∧
  ∃ k : ℕ, n = 7 * k ∧
  ((∃ m : ℕ, n = 4 * m + 3) ∨ (∃ m : ℕ, n = 9 * m + 3))

theorem numbers_satisfying_conditions :
  {n : ℕ | satisfies_conditions n} = {147, 399, 651, 903} := by
  sorry

end numbers_satisfying_conditions_l3612_361256


namespace tank_capacity_l3612_361280

theorem tank_capacity (initial_fraction : ℚ) (added_gallons : ℚ) (final_fraction : ℚ) :
  initial_fraction = 3/4 →
  added_gallons = 9 →
  final_fraction = 7/8 →
  initial_fraction * C + added_gallons = final_fraction * C →
  C = 72 :=
by
  sorry

#check tank_capacity

end tank_capacity_l3612_361280


namespace library_visitors_l3612_361214

/-- Proves that the average number of visitors on Sundays is 510 given the conditions -/
theorem library_visitors (total_days : Nat) (sunday_count : Nat) (avg_visitors : Nat) (non_sunday_visitors : Nat) :
  total_days = 30 ∧ 
  sunday_count = 5 ∧ 
  avg_visitors = 285 ∧ 
  non_sunday_visitors = 240 →
  (sunday_count * (total_days * avg_visitors - (total_days - sunday_count) * non_sunday_visitors)) / 
  (sunday_count * total_days) = 510 := by
  sorry


end library_visitors_l3612_361214


namespace coefficient_of_fifth_power_l3612_361218

theorem coefficient_of_fifth_power (x : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (2*x - 1)^5 * (x + 2) = a₀ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + a₅*(x-1)^5 + a₆*(x-1)^6 →
  a₅ = 176 := by
sorry

end coefficient_of_fifth_power_l3612_361218


namespace circle_area_decrease_l3612_361209

theorem circle_area_decrease (r : ℝ) (hr : r > 0) :
  let original_area := π * r^2
  let new_radius := r / 2
  let new_area := π * new_radius^2
  (original_area - new_area) / original_area = 3/4 := by
sorry

end circle_area_decrease_l3612_361209


namespace translation_property_l3612_361299

-- Define a translation as a function from ℂ to ℂ
def Translation := ℂ → ℂ

-- Define the property of a translation taking one point to another
def TranslatesTo (T : Translation) (z w : ℂ) : Prop := T z = w

theorem translation_property (T : Translation) :
  TranslatesTo T (1 - 2*I) (4 + 3*I) →
  TranslatesTo T (2 + 4*I) (5 + 9*I) := by
  sorry

end translation_property_l3612_361299


namespace line_passes_through_fixed_point_l3612_361294

/-- A line y = mx + (2m+1) always passes through the point (-2, 1) for any real m -/
theorem line_passes_through_fixed_point (m : ℝ) : 
  let f : ℝ → ℝ := fun x => m * x + (2 * m + 1)
  f (-2) = 1 := by sorry

end line_passes_through_fixed_point_l3612_361294


namespace squared_roots_polynomial_l3612_361273

theorem squared_roots_polynomial (x : ℝ) : 
  let f (x : ℝ) := x^3 + x^2 - 2*x - 1
  let g (x : ℝ) := x^3 - 5*x^2 + 6*x - 1
  ∀ r : ℝ, f r = 0 → ∃ s : ℝ, g s = 0 ∧ s = r^2 :=
by sorry

end squared_roots_polynomial_l3612_361273


namespace squares_after_seven_dwarfs_l3612_361292

/-- Represents the process of a dwarf cutting a square --/
def dwarf_cut (n : ℕ) : ℕ := n + 3

/-- Calculates the number of squares after n dwarfs have performed their cuts --/
def squares_after_cuts (n : ℕ) : ℕ := 
  Nat.iterate dwarf_cut n 1

/-- The theorem stating that after 7 dwarfs, there are 22 squares --/
theorem squares_after_seven_dwarfs : 
  squares_after_cuts 7 = 22 := by sorry

end squares_after_seven_dwarfs_l3612_361292


namespace factors_of_23232_l3612_361267

theorem factors_of_23232 : Nat.card (Nat.divisors 23232) = 42 := by
  sorry

end factors_of_23232_l3612_361267


namespace initial_amount_of_A_l3612_361252

/-- Represents the money exchange problem with three people --/
structure MoneyExchange where
  a : ℕ  -- Initial amount of A
  b : ℕ  -- Initial amount of B
  c : ℕ  -- Initial amount of C

/-- Predicate that checks if the money exchange satisfies the problem conditions --/
def satisfies_conditions (m : MoneyExchange) : Prop :=
  -- After all exchanges, everyone has 16 dollars
  4 * (m.a - m.b - m.c) = 16 ∧
  6 * m.b - 2 * m.a - 2 * m.c = 16 ∧
  7 * m.c - m.a - m.b = 16

/-- Theorem stating that if the conditions are satisfied, A's initial amount was 29 --/
theorem initial_amount_of_A (m : MoneyExchange) :
  satisfies_conditions m → m.a = 29 := by
  sorry


end initial_amount_of_A_l3612_361252


namespace consecutive_even_integers_sum_l3612_361286

theorem consecutive_even_integers_sum (n : ℤ) : 
  (n % 2 = 0) ∧ (n * (n + 2) * (n + 4) = 480) → n + (n + 2) + (n + 4) = 24 := by
  sorry

end consecutive_even_integers_sum_l3612_361286


namespace f_properties_l3612_361261

noncomputable def f (x : ℝ) : ℝ := x * Real.log x + x^2 - 3*x - x / Real.exp x

theorem f_properties (h : ∀ x, x > 0 → f x = x * Real.log x + x^2 - 3*x - x / Real.exp x) :
  (∃ x₀ > 0, ∀ x > 0, f x ≥ f x₀ ∧ f x₀ = -2 - 1 / Real.exp 1) ∧
  (∀ x, Real.exp x ≥ x + 1) ∧
  (∀ x y, 0 < x ∧ x < y → (deriv f x) < (deriv f y)) :=
sorry

end f_properties_l3612_361261


namespace negation_of_positive_square_is_false_l3612_361208

theorem negation_of_positive_square_is_false :
  ¬(∀ x : ℝ, x > 0 → x^2 > 0) = False :=
by sorry

end negation_of_positive_square_is_false_l3612_361208


namespace pharmacy_tubs_in_storage_l3612_361206

theorem pharmacy_tubs_in_storage (total_needed : ℕ) (bought_usual : ℕ) : ℕ :=
  let tubs_in_storage := total_needed - (bought_usual + bought_usual / 3)
  by
    sorry

#check pharmacy_tubs_in_storage 100 60

end pharmacy_tubs_in_storage_l3612_361206


namespace washington_goat_count_l3612_361225

/-- The number of goats Washington has -/
def washington_goats : ℕ := 140

/-- The number of goats Paddington has -/
def paddington_goats : ℕ := washington_goats + 40

/-- The total number of goats -/
def total_goats : ℕ := 320

theorem washington_goat_count : washington_goats = 140 :=
  by sorry

end washington_goat_count_l3612_361225


namespace polynomial_division_degree_l3612_361296

theorem polynomial_division_degree (f d q r : Polynomial ℝ) : 
  Polynomial.degree f = 15 →
  Polynomial.degree q = 9 →
  r = 5 * X^4 + 6 * X^3 - 2 * X + 7 →
  f = d * q + r →
  Polynomial.degree d = 6 := by
sorry

end polynomial_division_degree_l3612_361296


namespace fraction_relation_l3612_361213

theorem fraction_relation (x y z w : ℝ) 
  (h1 : x / y = 5)
  (h2 : y / z = 1 / 4)
  (h3 : z / w = 7) :
  w / x = 4 / 35 := by
sorry

end fraction_relation_l3612_361213


namespace added_classes_l3612_361282

theorem added_classes (initial_classes : ℕ) (students_per_class : ℕ) (new_total_students : ℕ)
  (h1 : initial_classes = 15)
  (h2 : students_per_class = 20)
  (h3 : new_total_students = 400) :
  (new_total_students - initial_classes * students_per_class) / students_per_class = 5 := by
sorry

end added_classes_l3612_361282


namespace absolute_value_equation_solution_difference_l3612_361212

theorem absolute_value_equation_solution_difference : ∃ (x₁ x₂ : ℝ), 
  (|x₁ + 3| = 15 ∧ |x₂ + 3| = 15) ∧ |x₁ - x₂| = 30 := by
  sorry

end absolute_value_equation_solution_difference_l3612_361212


namespace min_shots_theorem_l3612_361285

/-- The hit rate for each shot -/
def hit_rate : ℝ := 0.25

/-- The desired probability of hitting the target at least once -/
def desired_probability : ℝ := 0.75

/-- The probability of hitting the target at least once given n shots -/
def prob_hit_at_least_once (n : ℕ) : ℝ := 1 - (1 - hit_rate) ^ n

/-- The minimum number of shots required to achieve the desired probability -/
def min_shots : ℕ := 5

theorem min_shots_theorem :
  (∀ k < min_shots, prob_hit_at_least_once k < desired_probability) ∧
  prob_hit_at_least_once min_shots ≥ desired_probability :=
by sorry

end min_shots_theorem_l3612_361285


namespace centroid_distance_sum_l3612_361232

/-- Given a triangle DEF with centroid G, prove that if the sum of squared distances
    from G to the vertices is 90, then the sum of squared side lengths is 270. -/
theorem centroid_distance_sum (D E F G : ℝ × ℝ) : 
  (G = ((D.1 + E.1 + F.1) / 3, (D.2 + E.2 + F.2) / 3)) →  -- G is the centroid
  ((G.1 - D.1)^2 + (G.2 - D.2)^2 + 
   (G.1 - E.1)^2 + (G.2 - E.2)^2 + 
   (G.1 - F.1)^2 + (G.2 - F.2)^2 = 90) →  -- Sum of squared distances from G to vertices is 90
  ((D.1 - E.1)^2 + (D.2 - E.2)^2 + 
   (D.1 - F.1)^2 + (D.2 - F.2)^2 + 
   (E.1 - F.1)^2 + (E.2 - F.2)^2 = 270)  -- Sum of squared side lengths is 270
:= by sorry

end centroid_distance_sum_l3612_361232


namespace x_value_l3612_361295

theorem x_value (x y : ℚ) (eq1 : 3 * x - 2 * y = 7) (eq2 : x + 3 * y = 8) : x = 37 / 11 := by
  sorry

end x_value_l3612_361295


namespace inequality_proof_l3612_361241

theorem inequality_proof (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) 
  (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1) 
  (hd : 0 ≤ d ∧ d ≤ 1) : 
  a * b * (a - b) + b * c * (b - c) + c * d * (c - d) + d * a * (d - a) ≤ 8 / 27 := by
sorry

end inequality_proof_l3612_361241


namespace hyperbola_foci_l3612_361203

def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 12 = 1

def is_focus (x y : ℝ) : Prop :=
  (x = 4 ∧ y = 0) ∨ (x = -4 ∧ y = 0)

theorem hyperbola_foci :
  ∀ x y : ℝ, hyperbola_equation x y → is_focus x y :=
by sorry

end hyperbola_foci_l3612_361203


namespace four_x_plus_g_is_odd_l3612_361249

theorem four_x_plus_g_is_odd (x g : ℤ) (h : 2 * x - g = 11) : 
  ∃ k : ℤ, 4 * x + g = 2 * k + 1 := by
sorry

end four_x_plus_g_is_odd_l3612_361249


namespace parabola_symmetry_condition_l3612_361210

theorem parabola_symmetry_condition (a : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    y₁ = a * x₁^2 - 1 ∧ 
    y₂ = a * x₂^2 - 1 ∧ 
    x₁ + y₁ = -(x₂ + y₂) ∧ 
    x₁ ≠ x₂) → 
  a > 3/4 :=
by sorry

end parabola_symmetry_condition_l3612_361210


namespace total_cost_is_124_80_l3612_361215

-- Define the number of lessons and prices for each studio
def total_lessons : Nat := 10
def price_A : ℚ := 15
def price_B : ℚ := 12
def price_C : ℚ := 18

-- Define the number of lessons Tom takes at each studio
def lessons_A : Nat := 4
def lessons_B : Nat := 3
def lessons_C : Nat := 3

-- Define the discount percentage for Studio B
def discount_B : ℚ := 20 / 100

-- Define the number of free lessons at Studio C
def free_lessons_C : Nat := 1

-- Theorem to prove
theorem total_cost_is_124_80 : 
  (lessons_A * price_A) + 
  (lessons_B * price_B * (1 - discount_B)) + 
  ((lessons_C - free_lessons_C) * price_C) = 124.80 := by
  sorry

end total_cost_is_124_80_l3612_361215


namespace acute_slope_implies_a_is_one_l3612_361228

/-- The curve C is defined by y = x³ - 2ax² + 2ax -/
def C (a : ℝ) (x : ℝ) : ℝ := x^3 - 2*a*x^2 + 2*a*x

/-- The derivative of C with respect to x -/
def C_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*a*x + 2*a

/-- The slope is acute if it's greater than 0 -/
def is_slope_acute (slope : ℝ) : Prop := slope > 0

theorem acute_slope_implies_a_is_one :
  ∀ a : ℤ, (∀ x : ℝ, is_slope_acute (C_derivative a x)) → a = 1 := by
  sorry

end acute_slope_implies_a_is_one_l3612_361228


namespace root_in_interval_l3612_361266

theorem root_in_interval :
  ∃ x : ℝ, x ∈ Set.Icc 2 3 ∧ Real.exp x + Real.log x = 0 := by sorry

end root_in_interval_l3612_361266


namespace line_equation_l3612_361221

-- Define the lines
def line1 (x y : ℝ) : Prop := 2*x + y - 8 = 0
def line2 (x y : ℝ) : Prop := x - 2*y + 1 = 0
def line3 (x y : ℝ) : Prop := 4*x - 3*y - 7 = 0

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define parallelism
def parallel (a b c d e f : ℝ) : Prop := a*e = b*d

-- Define the line l
def line_l (x y : ℝ) : Prop := 4*x - 3*y - 6 = 0

-- Theorem statement
theorem line_equation : 
  ∃ (x₀ y₀ : ℝ), intersection_point x₀ y₀ ∧ 
  parallel 4 (-3) (-6) 4 (-3) (-7) ∧
  line_l x₀ y₀ := by sorry

end line_equation_l3612_361221


namespace arithmetic_calculations_l3612_361211

theorem arithmetic_calculations :
  (4.6 - (1.75 + 2.08) = 0.77) ∧
  (9.5 + 4.85 - 6.36 = 7.99) ∧
  (5.6 + 2.7 + 4.4 = 12.7) ∧
  (13 - 4.85 - 3.15 = 5) := by
  sorry

end arithmetic_calculations_l3612_361211


namespace flowchart_output_l3612_361278

def iterate_add_two (x : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n + 1 => iterate_add_two (x + 2) n

theorem flowchart_output : iterate_add_two 10 3 = 16 := by
  sorry

end flowchart_output_l3612_361278


namespace equation_solution_l3612_361233

theorem equation_solution : ∃ x : ℚ, x - (x + 2) / 2 = (2 * x - 1) / 3 - 1 ∧ x = 2 := by
  sorry

end equation_solution_l3612_361233


namespace infinitely_many_rational_solutions_l3612_361255

theorem infinitely_many_rational_solutions :
  ∃ f : ℕ → ℚ × ℚ,
    (∀ n : ℕ, (f n).1^3 + (f n).2^3 = 9) ∧
    (∀ m n : ℕ, m ≠ n → f m ≠ f n) :=
sorry

end infinitely_many_rational_solutions_l3612_361255


namespace bucket_capacity_proof_l3612_361224

/-- The capacity of a bucket in the first scenario, in litres. -/
def first_bucket_capacity : ℝ := 13.5

/-- The number of buckets required to fill the tank in the first scenario. -/
def first_scenario_buckets : ℕ := 28

/-- The number of buckets required to fill the tank in the second scenario. -/
def second_scenario_buckets : ℕ := 42

/-- The capacity of a bucket in the second scenario, in litres. -/
def second_bucket_capacity : ℝ := 9

theorem bucket_capacity_proof :
  first_bucket_capacity * first_scenario_buckets =
  second_bucket_capacity * second_scenario_buckets := by
sorry

end bucket_capacity_proof_l3612_361224


namespace rectangle_area_reduction_l3612_361276

/-- Given a rectangle with dimensions 5 and 7 inches, if reducing one side by 2 inches
    results in an area of 21 square inches, then reducing the other side by 2 inches
    will result in an area of 25 square inches. -/
theorem rectangle_area_reduction (w h : ℝ) : 
  w = 5 ∧ h = 7 ∧ (w - 2) * h = 21 → w * (h - 2) = 25 := by
  sorry

end rectangle_area_reduction_l3612_361276


namespace ratio_chain_l3612_361243

theorem ratio_chain (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hbc : b / c = 2 / 3)
  (hcd : c / d = 3 / 5) :
  a / d = 1 / 2 := by
sorry

end ratio_chain_l3612_361243


namespace rectangle_segment_ratio_l3612_361275

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a rectangle ABCD -/
structure Rectangle :=
  (A B C D : Point)
  (AB_length : ℝ)
  (BC_length : ℝ)

/-- Represents the ratio of segments -/
structure Ratio :=
  (r s t u : ℕ)

def is_on_segment (P Q R : Point) : Prop := sorry

def intersect (P Q R S : Point) : Point := sorry

def parallel (P Q R S : Point) : Prop := sorry

theorem rectangle_segment_ratio 
  (ABCD : Rectangle)
  (E F G : Point)
  (P Q R : Point)
  (h1 : ABCD.AB_length = 8)
  (h2 : ABCD.BC_length = 4)
  (h3 : is_on_segment ABCD.B E ABCD.C)
  (h4 : is_on_segment ABCD.B F ABCD.C)
  (h5 : is_on_segment ABCD.C G ABCD.D)
  (h6 : (ABCD.B.x - E.x) / (E.x - ABCD.C.x) = 1 / 2)
  (h7 : (ABCD.B.x - F.x) / (F.x - ABCD.C.x) = 2 / 1)
  (h8 : P = intersect ABCD.A E ABCD.B ABCD.D)
  (h9 : Q = intersect ABCD.A F ABCD.B ABCD.D)
  (h10 : R = intersect ABCD.A G ABCD.B ABCD.D)
  (h11 : parallel ABCD.A G ABCD.B ABCD.C) :
  ∃ (ratio : Ratio), 
    ratio.r = 3 ∧ 
    ratio.s = 2 ∧ 
    ratio.t = 6 ∧ 
    ratio.u = 6 ∧
    ratio.r + ratio.s + ratio.t + ratio.u = 17 := by sorry

end rectangle_segment_ratio_l3612_361275


namespace sector_area_l3612_361239

theorem sector_area (arc_length : ℝ) (central_angle : ℝ) (h1 : arc_length = 6) (h2 : central_angle = 2) :
  let radius := arc_length / central_angle
  (1/2) * radius^2 * central_angle = 9 := by
sorry

end sector_area_l3612_361239


namespace integral_equals_four_l3612_361246

theorem integral_equals_four : ∫ x in (1:ℝ)..2, (3*x^2 - 2*x) = 4 := by
  sorry

end integral_equals_four_l3612_361246


namespace interest_calculation_years_l3612_361271

/-- Proves that the number of years for which the interest is calculated on the first part is 8 --/
theorem interest_calculation_years (total_sum : ℚ) (second_part : ℚ) 
  (interest_rate_first : ℚ) (interest_rate_second : ℚ) (time_second : ℚ) :
  total_sum = 2795 →
  second_part = 1720 →
  interest_rate_first = 3 / 100 →
  interest_rate_second = 5 / 100 →
  time_second = 3 →
  let first_part := total_sum - second_part
  let interest_second := second_part * interest_rate_second * time_second
  ∃ (time_first : ℚ), first_part * interest_rate_first * time_first = interest_second ∧ time_first = 8 :=
by sorry

end interest_calculation_years_l3612_361271


namespace sum_mod_nine_l3612_361293

theorem sum_mod_nine :
  (2 + 333 + 5555 + 77777 + 999999 + 2222222 + 44444444 + 666666666) % 9 = 4 := by
  sorry

end sum_mod_nine_l3612_361293


namespace variance_of_transformed_binomial_l3612_361247

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- The variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

/-- The transformation of the random variable -/
def eta (X : BinomialRV) : ℝ → ℝ := fun x ↦ -2 * x + 1

theorem variance_of_transformed_binomial (X : BinomialRV) 
  (h_n : X.n = 6) (h_p : X.p = 0.4) : 
  variance X * 4 = 5.76 := by
  sorry

end variance_of_transformed_binomial_l3612_361247


namespace sqrt_100_equals_10_l3612_361260

theorem sqrt_100_equals_10 : Real.sqrt 100 = 10 := by
  sorry

end sqrt_100_equals_10_l3612_361260


namespace congruence_problem_l3612_361223

theorem congruence_problem (x y n : ℤ) : 
  x ≡ 45 [ZMOD 60] →
  y ≡ 98 [ZMOD 60] →
  n ∈ Finset.Icc 150 210 →
  (x - y ≡ n [ZMOD 60]) ↔ n = 187 := by
sorry

end congruence_problem_l3612_361223


namespace subtraction_problem_l3612_361234

theorem subtraction_problem (v : Nat) : v < 10 → 400 + 10 * v + 7 - 189 = 268 → v = 5 := by
  sorry

end subtraction_problem_l3612_361234


namespace shaded_fraction_is_one_eighth_l3612_361226

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

theorem shaded_fraction_is_one_eighth (r : Rectangle) 
  (h1 : r.width = 15)
  (h2 : r.height = 20)
  (h3 : ∃ (shaded_area : ℝ), shaded_area = (1/4) * ((1/2) * r.area)) :
  ∃ (shaded_area : ℝ), shaded_area / r.area = 1/8 := by
  sorry

end shaded_fraction_is_one_eighth_l3612_361226


namespace surface_area_reduction_approx_l3612_361230

/-- The number of faces in a single cube -/
def cube_faces : ℕ := 6

/-- The number of faces lost when splicing two cubes into a cuboid -/
def faces_lost : ℕ := 2

/-- The percentage reduction in surface area when splicing two cubes into a cuboid -/
def surface_area_reduction : ℚ :=
  (faces_lost : ℚ) / (2 * cube_faces : ℚ) * 100

theorem surface_area_reduction_approx :
  ∃ ε > 0, abs (surface_area_reduction - 167/10) < ε ∧ ε < 1/10 := by
  sorry

end surface_area_reduction_approx_l3612_361230


namespace remainder_three_to_forty_plus_five_mod_five_l3612_361217

theorem remainder_three_to_forty_plus_five_mod_five :
  (3^40 + 5) % 5 = 1 := by
  sorry

end remainder_three_to_forty_plus_five_mod_five_l3612_361217


namespace percentage_of_red_non_honda_cars_l3612_361270

theorem percentage_of_red_non_honda_cars
  (total_cars : ℕ)
  (honda_cars : ℕ)
  (honda_red_ratio : ℚ)
  (total_red_ratio : ℚ)
  (h1 : total_cars = 9000)
  (h2 : honda_cars = 5000)
  (h3 : honda_red_ratio = 90 / 100)
  (h4 : total_red_ratio = 60 / 100)
  : (((total_red_ratio * total_cars) - (honda_red_ratio * honda_cars)) /
     (total_cars - honda_cars)) = 225 / 1000 := by
  sorry

end percentage_of_red_non_honda_cars_l3612_361270


namespace quadratic_root_existence_l3612_361279

theorem quadratic_root_existence (a b c x₁ x₂ : ℝ) 
  (ha : a ≠ 0)
  (hx₁ : a * x₁^2 + b * x₁ + c = 0)
  (hx₂ : -a * x₂^2 + b * x₂ + c = 0) :
  ∃ x₃, (x₃ ∈ Set.Icc x₁ x₂ ∨ x₃ ∈ Set.Icc x₂ x₁) ∧ 
        (a / 2) * x₃^2 + b * x₃ + c = 0 :=
by sorry

end quadratic_root_existence_l3612_361279


namespace equal_perimeters_if_base_equals_height_base_equals_height_if_equal_perimeters_l3612_361289

/-- Represents a triangle ABC with inscribed rectangles --/
structure TriangleWithRectangles where
  -- The length of the base AB
  base : ℝ
  -- The height of the triangle corresponding to base AB
  height : ℝ
  -- A function that given a real number between 0 and 1, returns the dimensions of the inscribed rectangle
  rectangleDimensions : ℝ → (ℝ × ℝ)

/-- The perimeter of a rectangle given its dimensions --/
def rectanglePerimeter (dimensions : ℝ × ℝ) : ℝ :=
  2 * (dimensions.1 + dimensions.2)

/-- Theorem: If base equals height, then all inscribed rectangles have equal perimeters --/
theorem equal_perimeters_if_base_equals_height (triangle : TriangleWithRectangles) :
  triangle.base = triangle.height →
  ∀ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 →
  rectanglePerimeter (triangle.rectangleDimensions x) = rectanglePerimeter (triangle.rectangleDimensions y) :=
sorry

/-- Theorem: If all inscribed rectangles have equal perimeters, then base equals height --/
theorem base_equals_height_if_equal_perimeters (triangle : TriangleWithRectangles) :
  (∀ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 →
   rectanglePerimeter (triangle.rectangleDimensions x) = rectanglePerimeter (triangle.rectangleDimensions y)) →
  triangle.base = triangle.height :=
sorry

end equal_perimeters_if_base_equals_height_base_equals_height_if_equal_perimeters_l3612_361289


namespace complex_equation_solution_l3612_361200

variable (z : ℂ)

theorem complex_equation_solution :
  (1 - Complex.I) * z = 2 * Complex.I →
  z = -1 + Complex.I ∧ Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_equation_solution_l3612_361200


namespace mrs_hilt_nickels_l3612_361244

/-- Represents the number of coins of each type a person has -/
structure CoinCount where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ

/-- Calculates the total value in cents given a CoinCount -/
def totalValue (coins : CoinCount) : ℕ :=
  coins.pennies + 5 * coins.nickels + 10 * coins.dimes

/-- Mrs. Hilt's coin count, with unknown number of nickels -/
def mrsHilt (n : ℕ) : CoinCount :=
  { pennies := 2, nickels := n, dimes := 2 }

/-- Jacob's coin count -/
def jacob : CoinCount :=
  { pennies := 4, nickels := 1, dimes := 1 }

/-- Theorem stating that Mrs. Hilt must have 2 nickels -/
theorem mrs_hilt_nickels :
  ∃ n : ℕ, totalValue (mrsHilt n) - totalValue jacob = 13 ∧ n = 2 := by
  sorry

end mrs_hilt_nickels_l3612_361244


namespace sam_and_billy_total_money_l3612_361219

/-- Given that Sam has $75 and Billy has $25 less than twice Sam's money, 
    prove that their total money is $200. -/
theorem sam_and_billy_total_money :
  ∀ (sam_money billy_money : ℕ),
    sam_money = 75 →
    billy_money = 2 * sam_money - 25 →
    sam_money + billy_money = 200 := by
sorry

end sam_and_billy_total_money_l3612_361219


namespace pyramid_arrangements_10_l3612_361222

/-- The number of distinguishable ways to form a pyramid with n distinct pool balls -/
def pyramid_arrangements (n : ℕ) : ℕ :=
  n.factorial / 9

/-- The theorem stating that the number of distinguishable ways to form a pyramid
    with 10 distinct pool balls is 403,200 -/
theorem pyramid_arrangements_10 :
  pyramid_arrangements 10 = 403200 := by
  sorry

end pyramid_arrangements_10_l3612_361222


namespace smallest_distance_to_target_l3612_361248

def jump_distance_1 : ℕ := 364
def jump_distance_2 : ℕ := 715
def target_point : ℕ := 2010

theorem smallest_distance_to_target : 
  ∃ (x y : ℤ), 
    (∀ (a b : ℤ), |target_point - (jump_distance_1 * a + jump_distance_2 * b)| ≥ 
                   |target_point - (jump_distance_1 * x + jump_distance_2 * y)|) ∧
    |target_point - (jump_distance_1 * x + jump_distance_2 * y)| = 5 :=
by sorry

end smallest_distance_to_target_l3612_361248


namespace TU_length_l3612_361242

-- Define the points
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry
def R : ℝ × ℝ := sorry
def S : ℝ × ℝ := (16, 0)
def T : ℝ × ℝ := (16, 9.6)
def U : ℝ × ℝ := sorry

-- Define the triangles
def triangle_PQR : Set (ℝ × ℝ) := {P, Q, R}
def triangle_STU : Set (ℝ × ℝ) := {S, T, U}

-- Define the similarity of triangles
def similar_triangles (t1 t2 : Set (ℝ × ℝ)) : Prop := sorry

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem TU_length :
  similar_triangles triangle_PQR triangle_STU →
  distance Q R = 24 →
  distance P Q = 16 →
  distance P R = 19.2 →
  distance T U = 12.8 := by sorry

end TU_length_l3612_361242


namespace smallest_winning_number_l3612_361264

theorem smallest_winning_number :
  ∃ N : ℕ,
    N ≤ 499 ∧
    (∀ m : ℕ, m < N →
      (3 * m < 500 ∧
       3 * m + 25 < 500 ∧
       3 * (3 * m + 25) < 500 ∧
       3 * (3 * m + 25) + 25 ≥ 500 →
       False)) ∧
    3 * N < 500 ∧
    3 * N + 25 < 500 ∧
    3 * (3 * N + 25) < 500 ∧
    3 * (3 * N + 25) + 25 ≥ 500 ∧
    N = 45 :=
by sorry

#eval (45 / 10 + 45 % 10) -- Sum of digits of 45

end smallest_winning_number_l3612_361264


namespace oliver_learning_time_l3612_361259

/-- The number of vowels in the English alphabet -/
def num_vowels : ℕ := 5

/-- The total number of days Oliver needs to learn all vowels -/
def total_days : ℕ := 25

/-- The number of days Oliver needs to learn one vowel -/
def days_per_vowel : ℕ := total_days / num_vowels

theorem oliver_learning_time : days_per_vowel = 5 := by
  sorry

end oliver_learning_time_l3612_361259


namespace polynomial_remainder_l3612_361284

/-- Given a cubic polynomial g(x) = cx³ + 5x² + dx + 7, prove that if g(2) = 19 and g(-1) = -9, 
    then c = -25/3 and d = 88/3 -/
theorem polynomial_remainder (c d : ℚ) : 
  let g : ℚ → ℚ := λ x => c * x^3 + 5 * x^2 + d * x + 7
  (g 2 = 19 ∧ g (-1) = -9) → c = -25/3 ∧ d = 88/3 := by
sorry

end polynomial_remainder_l3612_361284


namespace joydens_number_difference_l3612_361245

theorem joydens_number_difference (m j c : ℕ) : 
  m = j + 20 →
  j < c →
  c = 80 →
  m + j + c = 180 →
  c - j = 40 :=
by
  sorry

end joydens_number_difference_l3612_361245


namespace product_of_three_numbers_l3612_361204

theorem product_of_three_numbers (a b c m : ℚ) : 
  a + b + c = 240 ∧ 
  6 * a = m ∧ 
  b - 12 = m ∧ 
  c + 12 = m ∧ 
  a ≤ c ∧ 
  c ≤ b → 
  a * b * c = 490108320 / 2197 := by
  sorry

end product_of_three_numbers_l3612_361204


namespace unique_assignment_l3612_361238

/-- Represents the digits assigned to letters --/
structure Assignment where
  A : Fin 5
  M : Fin 5
  E : Fin 5
  H : Fin 5
  Z : Fin 5
  N : Fin 5

/-- Checks if all assigned digits are different --/
def Assignment.allDifferent (a : Assignment) : Prop :=
  a.A ≠ a.M ∧ a.A ≠ a.E ∧ a.A ≠ a.H ∧ a.A ≠ a.Z ∧ a.A ≠ a.N ∧
  a.M ≠ a.E ∧ a.M ≠ a.H ∧ a.M ≠ a.Z ∧ a.M ≠ a.N ∧
  a.E ≠ a.H ∧ a.E ≠ a.Z ∧ a.E ≠ a.N ∧
  a.H ≠ a.Z ∧ a.H ≠ a.N ∧
  a.Z ≠ a.N

/-- Checks if the assignment satisfies the given inequalities --/
def Assignment.satisfiesInequalities (a : Assignment) : Prop :=
  3 > a.A.val + 1 ∧ 
  a.A.val + 1 > a.M.val + 1 ∧ 
  a.M.val + 1 < a.E.val + 1 ∧ 
  a.E.val + 1 < a.H.val + 1 ∧ 
  a.H.val + 1 < a.A.val + 1

/-- Checks if the assignment results in the correct ZAMENA number --/
def Assignment.correctZAMENA (a : Assignment) : Prop :=
  a.Z.val + 1 = 5 ∧
  a.A.val + 1 = 4 ∧
  a.M.val + 1 = 1 ∧
  a.E.val + 1 = 2 ∧
  a.N.val + 1 = 4 ∧
  a.H.val + 1 = 3

theorem unique_assignment :
  ∀ a : Assignment,
    a.allDifferent ∧ a.satisfiesInequalities → a.correctZAMENA :=
by sorry

end unique_assignment_l3612_361238


namespace conference_games_count_l3612_361257

/-- The number of teams in Division A -/
def teams_a : Nat := 7

/-- The number of teams in Division B -/
def teams_b : Nat := 5

/-- The number of games each team plays against others in its division -/
def intra_division_games : Nat := 2

/-- The number of games each team plays against teams in the other division (excluding rivalry game) -/
def inter_division_games : Nat := 1

/-- The number of special pre-season rivalry games per team -/
def rivalry_games : Nat := 1

/-- The total number of conference games scheduled -/
def total_games : Nat := 
  -- Games within Division A
  (teams_a * (teams_a - 1) / 2) * intra_division_games +
  -- Games within Division B
  (teams_b * (teams_b - 1) / 2) * intra_division_games +
  -- Regular inter-division games
  teams_a * teams_b * inter_division_games +
  -- Special pre-season rivalry games
  teams_a * rivalry_games

theorem conference_games_count : total_games = 104 := by
  sorry

end conference_games_count_l3612_361257


namespace intersection_implies_sum_zero_l3612_361258

theorem intersection_implies_sum_zero (α β : ℝ) :
  (∃ x₀ : ℝ, x₀ / (Real.sin α + Real.sin β) + (-x₀) / (Real.sin α + Real.cos β) = 1 ∧
              x₀ / (Real.cos α + Real.sin β) + (-x₀) / (Real.cos α + Real.cos β) = 1) →
  Real.sin α + Real.cos α + Real.sin β + Real.cos β = 0 := by
sorry

end intersection_implies_sum_zero_l3612_361258


namespace abc_product_l3612_361227

theorem abc_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a * b = 45 * Real.rpow 3 (1/3))
  (hac : a * c = 75 * Real.rpow 3 (1/3))
  (hbc : b * c = 30 * Real.rpow 3 (1/3)) :
  a * b * c = 75 * Real.sqrt 2 := by
  sorry

end abc_product_l3612_361227


namespace determinant_of_special_matrix_l3612_361265

open Matrix Real

theorem determinant_of_special_matrix (α β : ℝ) :
  let M : Matrix (Fin 3) (Fin 3) ℝ := !![0, cos α, sin α;
                                       sin α, 0, cos β;
                                       -cos α, -sin β, 0]
  det M = cos (β - 2*α) := by
sorry

end determinant_of_special_matrix_l3612_361265


namespace unique_polynomial_coefficients_l3612_361269

theorem unique_polynomial_coefficients :
  ∃! (a b c : ℕ+),
  let x : ℝ := Real.sqrt ((Real.sqrt 105) / 2 + 7 / 2)
  x^100 = 3*x^98 + 15*x^96 + 12*x^94 - x^50 + (a:ℝ)*x^46 + (b:ℝ)*x^44 + (c:ℝ)*x^40 ∧
  a + b + c = 5824 := by
sorry

end unique_polynomial_coefficients_l3612_361269


namespace negation_of_universal_statement_l3612_361290

theorem negation_of_universal_statement :
  (¬ ∀ x ∈ Set.Icc 0 2, x^2 - 2*x ≤ 0) ↔ (∃ x ∈ Set.Icc 0 2, x^2 - 2*x > 0) := by
  sorry

end negation_of_universal_statement_l3612_361290


namespace route_upper_bound_l3612_361231

/-- Represents the number of possible routes in a grid city -/
def f (m n : ℕ) : ℕ := sorry

/-- Theorem: The number of possible routes in a grid city is at most 2^(m*n) -/
theorem route_upper_bound (m n : ℕ) : f m n ≤ 2^(m*n) := by sorry

end route_upper_bound_l3612_361231


namespace rectangular_to_polar_conversion_l3612_361237

theorem rectangular_to_polar_conversion :
  ∀ (x y : ℝ), x = 6 ∧ y = 2 * Real.sqrt 3 →
  ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧
  r = 4 * Real.sqrt 3 ∧ θ = Real.pi / 6 ∧
  x = r * Real.cos θ ∧ y = r * Real.sin θ := by
  sorry

end rectangular_to_polar_conversion_l3612_361237


namespace unique_equal_sum_existence_l3612_361254

/-- The sum of the first n terms of an arithmetic sequence with first term a and common difference d -/
def arithmetic_sum (a d n : ℤ) : ℤ := n * (2 * a + (n - 1) * d) / 2

/-- The statement that there exists exactly one positive integer n such that
    the sum of the first n terms of the arithmetic sequence (8, 12, ...)
    equals the sum of the first n terms of the arithmetic sequence (17, 19, ...) -/
theorem unique_equal_sum_existence : ∃! (n : ℕ), n > 0 ∧ 
  arithmetic_sum 8 4 n = arithmetic_sum 17 2 n := by sorry

end unique_equal_sum_existence_l3612_361254


namespace arithmetic_sequence_pattern_l3612_361235

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_pattern (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 1 - 2 * a 2 + a 3 = 0) →
  (a 1 - 3 * a 2 + 3 * a 3 - a 4 = 0) →
  (a 1 - 4 * a 2 + 6 * a 3 - 4 * a 4 + a 5 = 0) →
  (a 1 - 5 * a 2 + 10 * a 3 - 10 * a 4 + 5 * a 5 - a 6 = 0) :=
by sorry

end arithmetic_sequence_pattern_l3612_361235


namespace triangle_angle_A_is_30_degrees_l3612_361263

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)
  (a b c : Real)
  (is_triangle : A + B + C = Real.pi)

-- State the theorem
theorem triangle_angle_A_is_30_degrees
  (abc : Triangle)
  (h1 : abc.a^2 - abc.b^2 = Real.sqrt 3 * abc.b * abc.c)
  (h2 : Real.sin abc.C = 2 * Real.sqrt 3 * Real.sin abc.B) :
  abc.A = Real.pi / 6 := by
sorry

end triangle_angle_A_is_30_degrees_l3612_361263


namespace rectangular_envelope_foldable_l3612_361274

-- Define a rectangular envelope
structure RectangularEnvelope where
  length : ℝ
  width : ℝ
  length_positive : length > 0
  width_positive : width > 0

-- Define a tetrahedron
structure Tetrahedron where
  surface_area : ℝ
  surface_area_positive : surface_area > 0

-- Define the property of being able to fold into two congruent tetrahedrons
def can_fold_into_congruent_tetrahedrons (env : RectangularEnvelope) : Prop :=
  ∃ (t : Tetrahedron), 
    t.surface_area = (env.length * env.width) / 2 ∧ 
    env.length ≠ env.width

-- State the theorem
theorem rectangular_envelope_foldable (env : RectangularEnvelope) :
  can_fold_into_congruent_tetrahedrons env :=
sorry

end rectangular_envelope_foldable_l3612_361274


namespace buses_passed_count_l3612_361205

/-- Represents the frequency of Dallas to Houston buses in minutes -/
def dallas_to_houston_frequency : ℕ := 40

/-- Represents the frequency of Houston to Dallas buses in minutes -/
def houston_to_dallas_frequency : ℕ := 60

/-- Represents the trip duration in hours -/
def trip_duration : ℕ := 6

/-- Represents the minute offset for Houston to Dallas buses -/
def houston_to_dallas_offset : ℕ := 30

/-- Calculates the number of Dallas-bound buses a Houston-bound bus passes on the highway -/
def buses_passed : ℕ := 
  sorry

theorem buses_passed_count : buses_passed = 10 := by
  sorry

end buses_passed_count_l3612_361205


namespace polynomial_d_value_l3612_361298

/-- Represents a polynomial of degree 4 -/
structure Polynomial4 (α : Type) [Field α] where
  a : α
  b : α
  c : α
  d : α

/-- Calculates the sum of coefficients for a polynomial of degree 4 -/
def sumCoefficients {α : Type} [Field α] (p : Polynomial4 α) : α :=
  1 + p.a + p.b + p.c + p.d

/-- Calculates the mean of zeros for a polynomial of degree 4 -/
def meanZeros {α : Type} [Field α] (p : Polynomial4 α) : α :=
  -p.a / 4

/-- The main theorem -/
theorem polynomial_d_value
  {α : Type} [Field α]
  (p : Polynomial4 α)
  (h1 : meanZeros p = p.d)
  (h2 : p.d = sumCoefficients p)
  (h3 : p.d = 3) :
  p.d = 3 := by sorry

end polynomial_d_value_l3612_361298


namespace selection_methods_eq_twelve_l3612_361297

/-- Represents the number of teachers available for selection -/
def total_teachers : ℕ := 4

/-- Represents the number of teachers to be selected -/
def selected_teachers : ℕ := 3

/-- Represents the number of phases in the training -/
def training_phases : ℕ := 3

/-- Represents the number of teachers who cannot participate in the first phase -/
def restricted_teachers : ℕ := 2

/-- Calculates the number of different selection methods -/
def selection_methods : ℕ := sorry

/-- Theorem stating that the number of selection methods is 12 -/
theorem selection_methods_eq_twelve : selection_methods = 12 := by sorry

end selection_methods_eq_twelve_l3612_361297


namespace expression_value_l3612_361253

theorem expression_value (a : ℝ) (h : a^2 + 2*a + 2 - Real.sqrt 3 = 0) :
  1 / (a + 1) - (a + 3) / (a^2 - 1) * (a^2 - 2*a + 1) / (a^2 + 4*a + 3) = Real.sqrt 3 + 1 := by
  sorry

end expression_value_l3612_361253


namespace slope_product_theorem_l3612_361207

theorem slope_product_theorem (m n : ℝ) : 
  m ≠ 0 → n ≠ 0 →  -- non-horizontal lines
  (∃ θ₁ θ₂ : ℝ, θ₁ = 3 * θ₂ ∧ m = Real.tan θ₁ ∧ n = Real.tan θ₂) →  -- angle relationship
  m = 6 * n →  -- slope relationship
  m * n = 9 / 17 := by
sorry

end slope_product_theorem_l3612_361207


namespace digit_multiplication_puzzle_l3612_361201

theorem digit_multiplication_puzzle :
  ∃! (A B C D E F : ℕ),
    A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 ∧ D ≤ 9 ∧ E ≤ 9 ∧ F ≤ 9 ∧
    A * (10 * B + A) = 10 * C + D ∧
    F * (10 * B + E) = 10 * D + C ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
    C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
    D ≠ E ∧ D ≠ F ∧
    E ≠ F :=
by sorry

end digit_multiplication_puzzle_l3612_361201


namespace profit_increase_l3612_361283

theorem profit_increase (cost_price selling_price : ℝ) (a : ℝ) 
  (h1 : cost_price > 0)
  (h2 : selling_price > cost_price)
  (h3 : (selling_price - cost_price) / cost_price = a / 100)
  (h4 : (selling_price - cost_price * 0.95) / (cost_price * 0.95) = (a + 15) / 100) :
  a = 185 := by
sorry

end profit_increase_l3612_361283


namespace rectangle_width_l3612_361220

/-- A rectangle with area 50 square meters and perimeter 30 meters has a width of 5 meters. -/
theorem rectangle_width (length width : ℝ) 
  (area_eq : length * width = 50)
  (perimeter_eq : 2 * length + 2 * width = 30) :
  width = 5 := by
  sorry

end rectangle_width_l3612_361220


namespace repeating_decimal_fraction_sum_numerator_denominator_l3612_361236

def repeating_decimal : ℚ := 345 / 999

theorem repeating_decimal_fraction :
  repeating_decimal = 115 / 111 :=
sorry

theorem sum_numerator_denominator :
  115 + 111 = 226 :=
sorry

end repeating_decimal_fraction_sum_numerator_denominator_l3612_361236


namespace mushroom_collection_l3612_361251

theorem mushroom_collection 
  (N I A V : ℝ) 
  (h_non_negative : 0 ≤ N ∧ 0 ≤ I ∧ 0 ≤ A ∧ 0 ≤ V)
  (h_natasha_most : N > I ∧ N > A ∧ N > V)
  (h_ira_least : I ≤ N ∧ I ≤ A ∧ I ≤ V)
  (h_alyosha_more : A > V) : 
  N + I > A + V := by
  sorry

end mushroom_collection_l3612_361251


namespace power_tower_comparison_l3612_361272

theorem power_tower_comparison : 3^(3^(3^3)) > 2^(2^(2^(2^2))) := by
  sorry

end power_tower_comparison_l3612_361272


namespace gcd_5800_14025_l3612_361291

theorem gcd_5800_14025 : Nat.gcd 5800 14025 = 25 := by
  sorry

end gcd_5800_14025_l3612_361291


namespace quadratic_inequality_1_l3612_361288

theorem quadratic_inequality_1 (x : ℝ) :
  x^2 - 7*x + 12 > 0 ↔ x < 3 ∨ x > 4 := by sorry

end quadratic_inequality_1_l3612_361288
