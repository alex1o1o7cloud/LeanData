import Mathlib

namespace complex_expression_equals_five_l3860_386006

theorem complex_expression_equals_five :
  (3 * Real.sqrt 12 - 2 * Real.sqrt (1/3) + Real.sqrt 48) / (2 * Real.sqrt 3) + (Real.sqrt (1/3))^2 = 5 := by
  sorry

end complex_expression_equals_five_l3860_386006


namespace sin_kpi_minus_x_is_odd_cos_squared_when_tan_pi_minus_x_is_two_cos_2x_plus_pi_third_symmetry_l3860_386050

-- Statement 1
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem sin_kpi_minus_x_is_odd (k : ℤ) :
  is_odd_function (λ x => Real.sin (k * Real.pi - x)) :=
sorry

-- Statement 2
theorem cos_squared_when_tan_pi_minus_x_is_two :
  ∀ x, Real.tan (Real.pi - x) = 2 → Real.cos x ^ 2 = 1/5 :=
sorry

-- Statement 3
def is_line_of_symmetry (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

theorem cos_2x_plus_pi_third_symmetry :
  is_line_of_symmetry (λ x => Real.cos (2*x + Real.pi/3)) (-2*Real.pi/3) :=
sorry

end sin_kpi_minus_x_is_odd_cos_squared_when_tan_pi_minus_x_is_two_cos_2x_plus_pi_third_symmetry_l3860_386050


namespace increasing_f_range_l3860_386014

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2 - a) * x + 1 else a^x

theorem increasing_f_range (a : ℝ) 
  (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : ∀ x y : ℝ, x < y → f a x < f a y) : 
  a ∈ Set.Icc (3/2) 2 ∧ a < 2 :=
sorry

end increasing_f_range_l3860_386014


namespace saree_price_calculation_l3860_386048

theorem saree_price_calculation (final_price : ℝ) 
  (h1 : final_price = 227.70) 
  (first_discount : ℝ) (second_discount : ℝ)
  (h2 : first_discount = 0.12)
  (h3 : second_discount = 0.25) : ∃ P : ℝ, 
  P * (1 - first_discount) * (1 - second_discount) = final_price ∧ P = 345 := by
  sorry

end saree_price_calculation_l3860_386048


namespace jack_son_birth_time_l3860_386084

def jack_lifetime : ℝ := 84

theorem jack_son_birth_time (adolescence : ℝ) (facial_hair : ℝ) (marriage : ℝ) (son_lifetime : ℝ) :
  adolescence = jack_lifetime / 6 →
  facial_hair = jack_lifetime / 6 + jack_lifetime / 12 →
  marriage = jack_lifetime / 6 + jack_lifetime / 12 + jack_lifetime / 7 →
  son_lifetime = jack_lifetime / 2 →
  jack_lifetime - (marriage + (jack_lifetime - son_lifetime - 4)) = 5 := by
sorry

end jack_son_birth_time_l3860_386084


namespace initial_student_count_l3860_386044

/-- Given the initial average weight and the new average weight after admitting a new student,
    prove that the initial number of students is 29. -/
theorem initial_student_count
  (initial_avg : ℝ)
  (new_avg : ℝ)
  (new_student_weight : ℝ)
  (h1 : initial_avg = 28)
  (h2 : new_avg = 27.1)
  (h3 : new_student_weight = 1)
  : ∃ n : ℕ, n = 29 ∧ 
    n * initial_avg + new_student_weight = (n + 1) * new_avg :=
by
  sorry


end initial_student_count_l3860_386044


namespace quadratic_roots_sum_l3860_386055

theorem quadratic_roots_sum (m n : ℝ) : 
  m^2 + 2*m - 5 = 0 → n^2 + 2*n - 5 = 0 → m^2 + m*n + 3*m + n = -2 := by
  sorry

end quadratic_roots_sum_l3860_386055


namespace vector_magnitude_proof_l3860_386012

open Real

theorem vector_magnitude_proof (a b : ℝ × ℝ) :
  let angle := 60 * π / 180
  norm a = 2 ∧ norm b = 5 ∧ 
  a.1 * b.1 + a.2 * b.2 = norm a * norm b * cos angle →
  norm (2 • a - b) = sqrt 21 := by
  sorry

end vector_magnitude_proof_l3860_386012


namespace david_fewer_crunches_l3860_386030

/-- Represents the number of exercises done by a person -/
structure ExerciseCount where
  pushups : ℕ
  crunches : ℕ

/-- Given the exercise counts for David and Zachary, proves that David did 17 fewer crunches than Zachary -/
theorem david_fewer_crunches (david zachary : ExerciseCount) 
  (h1 : david.pushups = zachary.pushups + 40)
  (h2 : david.crunches < zachary.crunches)
  (h3 : zachary.pushups = 34)
  (h4 : zachary.crunches = 62)
  (h5 : david.crunches = 45) :
  zachary.crunches - david.crunches = 17 := by
  sorry


end david_fewer_crunches_l3860_386030


namespace largest_lower_bound_area_l3860_386068

/-- A point in the Euclidean plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The convex hull of a set of points -/
def convex_hull (points : Set Point) : Set Point := sorry

/-- The area of a set of points -/
def area (s : Set Point) : ℝ := sorry

/-- A convex set of points -/
def is_convex (s : Set Point) : Prop := sorry

theorem largest_lower_bound_area (points : Set Point) :
  ∀ (s : Set Point), (is_convex s ∧ points ⊆ s) →
    area (convex_hull points) ≤ area s :=
by sorry

end largest_lower_bound_area_l3860_386068


namespace tangent_line_to_circle_l3860_386052

theorem tangent_line_to_circle (r : ℝ) (hr : r > 0) : 
  (∀ x y : ℝ, 2*x + y = r → x^2 + y^2 = 2*r → 
   ∃ x₀ y₀ : ℝ, 2*x₀ + y₀ = r ∧ x₀^2 + y₀^2 = 2*r ∧ 
   ∀ x₁ y₁ : ℝ, 2*x₁ + y₁ = r → x₁^2 + y₁^2 ≥ 2*r) →
  r = 10 :=
by sorry

end tangent_line_to_circle_l3860_386052


namespace additional_cats_needed_prove_additional_cats_l3860_386019

theorem additional_cats_needed (total_mice : ℕ) (initial_cats : ℕ) (initial_days : ℕ) (total_days : ℕ) : ℕ :=
  let initial_work := total_mice / 2
  let remaining_work := total_mice - initial_work
  let initial_rate := initial_work / (initial_cats * initial_days)
  let remaining_days := total_days - initial_days
  let additional_cats := (remaining_work / (initial_rate * remaining_days)) - initial_cats
  additional_cats

theorem prove_additional_cats :
  additional_cats_needed 100 2 5 7 = 3 := by
  sorry

end additional_cats_needed_prove_additional_cats_l3860_386019


namespace right_triangle_arctans_l3860_386091

theorem right_triangle_arctans (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (right_angle : a^2 = b^2 + c^2) : 
  Real.arctan (b / (a + c)) + Real.arctan (c / (a + b)) = π / 4 := by
  sorry

end right_triangle_arctans_l3860_386091


namespace complex_number_in_first_quadrant_l3860_386098

theorem complex_number_in_first_quadrant (z : ℂ) (h : z * (4 + I) = 3 + I) : 
  0 < z.re ∧ 0 < z.im := by
  sorry

end complex_number_in_first_quadrant_l3860_386098


namespace combined_data_mode_l3860_386026

/-- Given two sets of data with specified averages, proves that the mode of the combined set is 8 -/
theorem combined_data_mode (x y : ℝ) : 
  (3 + x + 2*y + 5) / 4 = 6 →
  (x + 6 + y) / 3 = 6 →
  let combined_set := [3, x, 2*y, 5, x, 6, y]
  ∃ (mode : ℝ), mode = 8 ∧ 
    (∀ z ∈ combined_set, (combined_set.filter (λ t => t = z)).length ≤ 
                         (combined_set.filter (λ t => t = mode)).length) :=
by sorry

end combined_data_mode_l3860_386026


namespace hyperbola_asymptotes_l3860_386041

/-- Given a hyperbola with equation x²/9 - y² = 1, its asymptotes are y = x/3 and y = -x/3 -/
theorem hyperbola_asymptotes :
  let hyperbola := fun (x y : ℝ) => x^2 / 9 - y^2 = 1
  let asymptote1 := fun (x y : ℝ) => y = x / 3
  let asymptote2 := fun (x y : ℝ) => y = -x / 3
  (∀ x y, hyperbola x y → (asymptote1 x y ∨ asymptote2 x y)) :=
by
  sorry

end hyperbola_asymptotes_l3860_386041


namespace billiard_ball_trajectory_l3860_386058

theorem billiard_ball_trajectory :
  ∀ (x y : ℚ),
    (x ≥ 0 ∧ y ≥ 0) →  -- Restricting to first quadrant
    (y = x / Real.sqrt 2) →  -- Line equation
    (¬ ∃ (m n : ℤ), (x = ↑m ∧ y = ↑n)) :=  -- No integer coordinate intersection
by sorry

end billiard_ball_trajectory_l3860_386058


namespace island_population_even_l3860_386005

/-- Represents the type of inhabitants on the island -/
inductive Inhabitant
| Knight
| Liar

/-- Represents a claim about the number of inhabitants -/
inductive Claim
| EvenKnights
| OddLiars

/-- Function that determines if a given inhabitant tells the truth about a claim -/
def tellsTruth (i : Inhabitant) (c : Claim) : Prop :=
  match i, c with
  | Inhabitant.Knight, _ => true
  | Inhabitant.Liar, _ => false

/-- The island population -/
structure Island where
  inhabitants : List Inhabitant
  claims : List (Inhabitant × Claim)
  all_claimed : ∀ i ∈ inhabitants, ∃ c, (i, c) ∈ claims

theorem island_population_even (isle : Island) : Even (List.length isle.inhabitants) := by
  sorry


end island_population_even_l3860_386005


namespace distance_ratio_cars_l3860_386072

/-- Represents a car with its speed and travel time -/
structure Car where
  speed : ℝ
  time : ℝ

/-- Calculates the distance traveled by a car -/
def distance (car : Car) : ℝ := car.speed * car.time

/-- Theorem: The ratio of distances covered by Car A and Car B is 2:1 -/
theorem distance_ratio_cars (carA carB : Car)
  (hA_speed : carA.speed = 80)
  (hA_time : carA.time = 5)
  (hB_speed : carB.speed = 100)
  (hB_time : carB.time = 2) :
  distance carA / distance carB = 2 := by
  sorry

#check distance_ratio_cars

end distance_ratio_cars_l3860_386072


namespace square_difference_1989_l3860_386043

theorem square_difference_1989 :
  {(a, b) : ℕ × ℕ | a > b ∧ a ^ 2 - b ^ 2 = 1989} =
  {(995, 994), (333, 330), (115, 106), (83, 70), (67, 50), (45, 6)} :=
by sorry

end square_difference_1989_l3860_386043


namespace unanswered_questions_count_l3860_386076

/-- Represents the scoring system for a math test. -/
structure ScoringSystem where
  correct : Int
  wrong : Int
  unanswered : Int
  initial : Int

/-- Represents the result of a math test. -/
structure TestResult where
  correct : Nat
  wrong : Nat
  unanswered : Nat
  total_score : Int

/-- Calculates the score based on a given scoring system and test result. -/
def calculate_score (system : ScoringSystem) (result : TestResult) : Int :=
  system.initial +
  system.correct * result.correct +
  system.wrong * result.wrong +
  system.unanswered * result.unanswered

theorem unanswered_questions_count
  (new_system : ScoringSystem)
  (old_system : ScoringSystem)
  (result : TestResult)
  (h1 : new_system = { correct := 6, wrong := 0, unanswered := 3, initial := 0 })
  (h2 : old_system = { correct := 4, wrong := -1, unanswered := 0, initial := 40 })
  (h3 : result.correct + result.wrong + result.unanswered = 35)
  (h4 : calculate_score new_system result = 120)
  (h5 : calculate_score old_system result = 100) :
  result.unanswered = 5 := by
  sorry

end unanswered_questions_count_l3860_386076


namespace min_value_theorem_l3860_386032

theorem min_value_theorem (a b c d : ℝ) 
  (h : |b - Real.log a / a| + |c - d + 2| = 0) : 
  ∃ (min_val : ℝ), min_val = 9/2 ∧ 
    ∀ (x y : ℝ), (x - y)^2 + (Real.log x / x - (y + 2))^2 ≥ min_val :=
sorry

end min_value_theorem_l3860_386032


namespace triangle_inequality_l3860_386018

theorem triangle_inequality (a b c A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  3 * a^2 + 3 * b^2 = c^2 + 4 * a * b →
  Real.tan (Real.sin A) ≤ Real.tan (Real.cos B) :=
sorry

end triangle_inequality_l3860_386018


namespace cubic_equation_solutions_l3860_386051

theorem cubic_equation_solutions :
  let x₁ : ℂ := 4
  let x₂ : ℂ := -2 + 2 * Complex.I * Real.sqrt 3
  let x₃ : ℂ := -2 - 2 * Complex.I * Real.sqrt 3
  (∀ x : ℂ, 2 * x^3 = 128 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) :=
by sorry

end cubic_equation_solutions_l3860_386051


namespace parabola_intersection_l3860_386093

theorem parabola_intersection :
  let f (x : ℝ) := 4 * x^2 + 3 * x - 4
  let g (x : ℝ) := 2 * x^2 + 15
  ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧
    f x₁ = g x₁ ∧ f x₂ = g x₂ ∧
    x₁ = -19/2 ∧ x₂ = 5/2 ∧
    f x₁ = 195.5 ∧ f x₂ = 27.5 ∧
    ∀ (x : ℝ), f x = g x → x = x₁ ∨ x = x₂ :=
by sorry

end parabola_intersection_l3860_386093


namespace num_routes_eq_factorial_power_l3860_386066

/-- Represents the number of southern cities -/
def num_southern_cities : ℕ := 4

/-- Represents the number of northern cities -/
def num_northern_cities : ℕ := 5

/-- Represents the number of transfers between southern cities -/
def num_transfers : ℕ := num_southern_cities

/-- Calculates the number of different routes for the traveler -/
def num_routes : ℕ := (Nat.factorial (num_southern_cities - 1)) * (num_northern_cities ^ num_transfers)

/-- Theorem stating that the number of routes is equal to 3! × 5^4 -/
theorem num_routes_eq_factorial_power : num_routes = 3750 := by sorry

end num_routes_eq_factorial_power_l3860_386066


namespace min_value_expression_l3860_386080

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (x + 1/y) * (x + 1/y - 1024) + (y + 1/x) * (y + 1/x - 1024) ≥ -524288 ∧
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
    (a + 1/b) * (a + 1/b - 1024) + (b + 1/a) * (b + 1/a - 1024) = -524288 :=
by sorry

end min_value_expression_l3860_386080


namespace problem_solution_l3860_386013

theorem problem_solution :
  ∀ (a b : ℝ),
  let A := 2 * a^2 + 3 * a * b - 2 * a - 1
  let B := -a^2 + a * b + a + 3
  (a = -1 ∧ b = 10 → 4 * A - (3 * A - 2 * B) = -45) ∧
  (a * b = 1 → 4 * A - (3 * A - 2 * B) = 10) :=
by sorry

end problem_solution_l3860_386013


namespace inequality_proof_l3860_386090

theorem inequality_proof (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_ineq : x + y + z ≥ x*y + y*z + z*x) : 
  x/(y*z) + y/(z*x) + z/(x*y) ≥ Real.sqrt 3 :=
sorry

end inequality_proof_l3860_386090


namespace min_wings_theorem_l3860_386038

/-- Represents the number and cost of birds John bought -/
structure BirdPurchase where
  parrots : Nat
  pigeons : Nat
  canaries : Nat
  total_cost : Nat

/-- Calculates the total number of wings for a given bird purchase -/
def total_wings (purchase : BirdPurchase) : Nat :=
  2 * (purchase.parrots + purchase.pigeons + purchase.canaries)

/-- Checks if the purchase satisfies all conditions -/
def is_valid_purchase (purchase : BirdPurchase) : Prop :=
  purchase.parrots ≥ 1 ∧
  purchase.pigeons ≥ 1 ∧
  purchase.canaries ≥ 1 ∧
  purchase.total_cost = 200 ∧
  purchase.total_cost = 30 * purchase.parrots + 20 * purchase.pigeons + 15 * purchase.canaries

theorem min_wings_theorem :
  ∃ (purchase : BirdPurchase),
    is_valid_purchase purchase ∧
    (∀ (other : BirdPurchase), is_valid_purchase other → total_wings purchase ≤ total_wings other) ∧
    total_wings purchase = 24 := by
  sorry

end min_wings_theorem_l3860_386038


namespace staircase_dissection_l3860_386081

/-- Represents an n-staircase polyomino -/
structure Staircase (n : ℕ+) where
  cells : Fin (n * (n + 1) / 2) → Bool

/-- Predicate to check if a staircase is valid -/
def is_valid_staircase (n : ℕ+) (s : Staircase n) : Prop :=
  ∀ i : Fin n, ∃ j : Fin (i + 1), s.cells ⟨i * (i + 1) / 2 + j, sorry⟩ = true

/-- Predicate to check if a staircase can be dissected into smaller staircases -/
def can_be_dissected (n : ℕ+) (s : Staircase n) : Prop :=
  ∃ (m : ℕ) (smaller_staircases : Fin m → Staircase n),
    (∀ i : Fin m, is_valid_staircase n (smaller_staircases i)) ∧
    (∀ i : Fin m, Staircase.cells (smaller_staircases i) ≠ s.cells) ∧
    (∀ cell : Fin (n * (n + 1) / 2), 
      s.cells cell = true ↔ ∃ i : Fin m, (smaller_staircases i).cells cell = true)

/-- Theorem stating that any n-staircase can be dissected into strictly smaller n-staircases -/
theorem staircase_dissection (n : ℕ+) (s : Staircase n) 
  (h : is_valid_staircase n s) : can_be_dissected n s :=
sorry

end staircase_dissection_l3860_386081


namespace complex_equation_solution_l3860_386009

theorem complex_equation_solution (z : ℂ) : (z + 1) * Complex.I = 1 - Complex.I → z = -2 - Complex.I := by
  sorry

end complex_equation_solution_l3860_386009


namespace mango_ratio_proof_l3860_386016

/-- Proves that the ratio of mangoes sold at the market to total mangoes harvested is 1:2 -/
theorem mango_ratio_proof (total_mangoes : ℕ) (num_neighbors : ℕ) (mangoes_per_neighbor : ℕ)
  (h1 : total_mangoes = 560)
  (h2 : num_neighbors = 8)
  (h3 : mangoes_per_neighbor = 35) :
  (total_mangoes - num_neighbors * mangoes_per_neighbor) / total_mangoes = 1 / 2 := by
  sorry

end mango_ratio_proof_l3860_386016


namespace mary_pies_count_l3860_386011

theorem mary_pies_count (apples_per_pie : ℕ) (harvested_apples : ℕ) (apples_to_buy : ℕ) :
  apples_per_pie = 8 →
  harvested_apples = 50 →
  apples_to_buy = 30 →
  (harvested_apples + apples_to_buy) / apples_per_pie = 10 :=
by
  sorry

end mary_pies_count_l3860_386011


namespace monotonic_iff_m_geq_one_third_l3860_386046

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + x^2 + m*x + 1

-- State the theorem
theorem monotonic_iff_m_geq_one_third :
  ∀ m : ℝ, (∀ x : ℝ, Monotone (f m)) ↔ m ≥ 1/3 := by sorry

end monotonic_iff_m_geq_one_third_l3860_386046


namespace math_question_probability_l3860_386040

/-- The probability of drawing a math question in a quiz -/
theorem math_question_probability :
  let chinese_questions : ℕ := 2
  let math_questions : ℕ := 3
  let comprehensive_questions : ℕ := 4
  let total_questions : ℕ := chinese_questions + math_questions + comprehensive_questions
  (math_questions : ℚ) / (total_questions : ℚ) = 1 / 3 := by
  sorry

end math_question_probability_l3860_386040


namespace odd_divisors_of_180_l3860_386053

/-- The number of positive divisors of 180 that are not divisible by 2 -/
def count_odd_divisors (n : ℕ) : ℕ :=
  (Finset.filter (fun d => d ∣ n ∧ ¬ 2 ∣ d) (Finset.range (n + 1))).card

/-- Theorem stating that the number of positive divisors of 180 not divisible by 2 is 6 -/
theorem odd_divisors_of_180 : count_odd_divisors 180 = 6 := by
  sorry

end odd_divisors_of_180_l3860_386053


namespace gary_initial_amount_l3860_386096

/-- Gary's initial amount of money -/
def initial_amount : ℕ := sorry

/-- Amount Gary spent on the snake -/
def spent_amount : ℕ := 55

/-- Amount Gary has left -/
def remaining_amount : ℕ := 18

/-- Theorem: Gary's initial amount equals the sum of spent and remaining amounts -/
theorem gary_initial_amount : initial_amount = spent_amount + remaining_amount := by sorry

end gary_initial_amount_l3860_386096


namespace sin_70_in_terms_of_sin_10_l3860_386008

theorem sin_70_in_terms_of_sin_10 (k : ℝ) (h : Real.sin (10 * π / 180) = k) :
  Real.sin (70 * π / 180) = 1 - 2 * k^2 := by
  sorry

end sin_70_in_terms_of_sin_10_l3860_386008


namespace symmetric_line_passes_through_fixed_point_l3860_386047

/-- A line in 2D space represented by its slope and a point it passes through -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The point of symmetry -/
def symmetryPoint : ℝ × ℝ := (2, 1)

/-- Line l₁ defined as y = k(x-4) -/
def l₁ (k : ℝ) : Line :=
  { slope := k, point := (4, 0) }

/-- Line l₂ symmetric to l₁ with respect to the symmetry point -/
def l₂ (k : ℝ) : Line :=
  sorry -- definition omitted as it's not directly given in the problem

theorem symmetric_line_passes_through_fixed_point (k : ℝ) :
  (0, 2) ∈ {p : ℝ × ℝ | p.2 = (l₂ k).slope * (p.1 - (l₂ k).point.1) + (l₂ k).point.2} :=
sorry

end symmetric_line_passes_through_fixed_point_l3860_386047


namespace latin_square_symmetric_diagonal_l3860_386002

/-- A Latin square of order 7 -/
def LatinSquare7 (A : Fin 7 → Fin 7 → Fin 7) : Prop :=
  ∀ i j : Fin 7, ∀ k : Fin 7, (∃! x : Fin 7, A i x = k) ∧ (∃! y : Fin 7, A y j = k)

/-- Symmetry with respect to the main diagonal -/
def SymmetricMatrix (A : Fin 7 → Fin 7 → Fin 7) : Prop :=
  ∀ i j : Fin 7, A i j = A j i

/-- All numbers from 1 to 7 appear on the main diagonal -/
def AllNumbersOnDiagonal (A : Fin 7 → Fin 7 → Fin 7) : Prop :=
  ∀ k : Fin 7, ∃ i : Fin 7, A i i = k

theorem latin_square_symmetric_diagonal 
  (A : Fin 7 → Fin 7 → Fin 7) 
  (h1 : LatinSquare7 A) 
  (h2 : SymmetricMatrix A) : 
  AllNumbersOnDiagonal A :=
sorry

end latin_square_symmetric_diagonal_l3860_386002


namespace fourth_rectangle_area_l3860_386067

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a partition of a rectangle into four smaller rectangles -/
structure RectanglePartition where
  total : Rectangle
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ

/-- The theorem statement -/
theorem fourth_rectangle_area 
  (partition : RectanglePartition)
  (h1 : partition.total.length = 20)
  (h2 : partition.total.width = 12)
  (h3 : partition.area1 = 24)
  (h4 : partition.area2 = 48)
  (h5 : partition.area3 = 36) :
  partition.total.length * partition.total.width - (partition.area1 + partition.area2 + partition.area3) = 112 :=
by sorry

end fourth_rectangle_area_l3860_386067


namespace real_part_of_z_l3860_386088

theorem real_part_of_z (z : ℂ) (h : Complex.I * (z + 1) = -3 + 2 * Complex.I) : 
  z.re = 1 := by
  sorry

end real_part_of_z_l3860_386088


namespace percentage_less_than_l3860_386045

theorem percentage_less_than (x y z : ℝ) :
  x = 1.2 * y →
  x = 0.84 * z →
  y = 0.7 * z :=
by
  sorry

end percentage_less_than_l3860_386045


namespace wood_sawing_time_l3860_386069

/-- Time to saw wood into segments -/
def saw_time (segments : ℕ) (time : ℕ) : Prop :=
  segments > 1 ∧ time = (segments - 1) * (12 / 3)

theorem wood_sawing_time :
  saw_time 4 12 →
  saw_time 8 28 ∧ ¬saw_time 8 24 := by
sorry

end wood_sawing_time_l3860_386069


namespace larry_channels_l3860_386035

/-- Calculates the final number of channels for Larry given the initial count and subsequent changes. -/
def final_channels (initial : ℕ) (removed : ℕ) (replaced : ℕ) (reduced : ℕ) (sports : ℕ) (supreme : ℕ) : ℕ :=
  initial - removed + replaced - reduced + sports + supreme

/-- Theorem stating that Larry's final channel count is 147 given the specific changes. -/
theorem larry_channels : 
  final_channels 150 20 12 10 8 7 = 147 := by
  sorry

end larry_channels_l3860_386035


namespace hayley_initial_meatballs_hayley_initial_meatballs_proof_l3860_386060

theorem hayley_initial_meatballs : ℕ → ℕ → ℕ → Prop :=
  fun initial_meatballs stolen_meatballs remaining_meatballs =>
    (stolen_meatballs = 14) →
    (remaining_meatballs = 11) →
    (initial_meatballs = stolen_meatballs + remaining_meatballs) →
    (initial_meatballs = 25)

-- Proof
theorem hayley_initial_meatballs_proof :
  hayley_initial_meatballs 25 14 11 := by
  sorry

end hayley_initial_meatballs_hayley_initial_meatballs_proof_l3860_386060


namespace max_value_P_l3860_386061

theorem max_value_P (a b x₁ x₂ x₃ : ℝ) 
  (h1 : a = x₁ + x₂ + x₃)
  (h2 : a = x₁ * x₂ * x₃)
  (h3 : a * b = x₁ * x₂ + x₂ * x₃ + x₃ * x₁)
  (h4 : x₁ > 0)
  (h5 : x₂ > 0)
  (h6 : x₃ > 0) :
  let P := (a^2 + 6*b + 1) / (a^2 + a)
  ∃ (max_P : ℝ), ∀ (P_val : ℝ), P ≤ P_val → max_P ≥ P_val ∧ max_P = (9 + Real.sqrt 3) / 9 := by
  sorry


end max_value_P_l3860_386061


namespace function_solution_l3860_386036

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → f x + 2 * f (1 / x) = 3 * x

/-- The theorem stating that the function satisfying the equation has a specific form -/
theorem function_solution (f : ℝ → ℝ) (h : FunctionalEquation f) :
    ∀ x : ℝ, x ≠ 0 → f x = -x + 2 / x := by
  sorry

end function_solution_l3860_386036


namespace renovation_cost_calculation_l3860_386042

def renovation_cost (hourly_rates : List ℝ) (hours_per_day : ℝ) (days : ℕ) 
  (meal_cost : ℝ) (material_cost : ℝ) (unexpected_costs : List ℝ) : ℝ :=
  let daily_labor_cost := hourly_rates.sum * hours_per_day
  let total_labor_cost := daily_labor_cost * days
  let total_meal_cost := meal_cost * hourly_rates.length * days
  let total_unexpected_cost := unexpected_costs.sum
  total_labor_cost + total_meal_cost + material_cost + total_unexpected_cost

theorem renovation_cost_calculation : 
  renovation_cost [15, 20, 18, 22] 8 10 10 2500 [750, 500, 400] = 10550 := by
  sorry

end renovation_cost_calculation_l3860_386042


namespace inequality_proof_l3860_386010

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 * b^2 + b^2 * c^2 + a^2 * c^2 ≥ a * b * c * (a + b + c) := by
  sorry

end inequality_proof_l3860_386010


namespace number_added_l3860_386063

theorem number_added (x y : ℝ) (h1 : x = 55) (h2 : (x / 5) + y = 21) : y = 10 := by
  sorry

end number_added_l3860_386063


namespace dance_step_time_ratio_l3860_386025

/-- Proves that the ratio of time spent on the third dance step to the combined time
    spent on the first and second steps is 1:1, given the specified conditions. -/
theorem dance_step_time_ratio :
  ∀ (time_step1 time_step2 time_step3 total_time : ℕ),
  time_step1 = 30 →
  time_step2 = time_step1 / 2 →
  total_time = 90 →
  total_time = time_step1 + time_step2 + time_step3 →
  time_step3 = time_step1 + time_step2 :=
by
  sorry

#check dance_step_time_ratio

end dance_step_time_ratio_l3860_386025


namespace line_vector_to_slope_intercept_l3860_386078

/-- Given a line in vector form, prove it's equivalent to a specific slope-intercept form -/
theorem line_vector_to_slope_intercept :
  ∀ (x y : ℝ), (-2 : ℝ) * (x - 5) + 4 * (y + 6) = 0 ↔ y = (1/2 : ℝ) * x - (17/2 : ℝ) :=
by sorry

end line_vector_to_slope_intercept_l3860_386078


namespace lamp_post_break_height_l3860_386062

/-- Given a 6-meter tall lamp post that breaks and hits the ground 2 meters away from its base,
    the breaking point is √10 meters above the ground. -/
theorem lamp_post_break_height :
  ∀ (x : ℝ),
  x > 0 →
  x < 6 →
  x * x + 2 * 2 = (6 - x) * (6 - x) →
  x = Real.sqrt 10 := by
sorry

end lamp_post_break_height_l3860_386062


namespace point_in_fourth_quadrant_l3860_386028

theorem point_in_fourth_quadrant (α : Real) (h : -π/2 < α ∧ α < 0) :
  let P : ℝ × ℝ := (Real.tan α, Real.cos α)
  P.1 < 0 ∧ P.2 > 0 := by
  sorry

end point_in_fourth_quadrant_l3860_386028


namespace quadratic_equation_solution_l3860_386065

theorem quadratic_equation_solution (x : ℝ) : 
  x^2 - 6*x + 8 = 0 ∧ x ≠ 0 → x = 2 ∨ x = 4 := by
  sorry

end quadratic_equation_solution_l3860_386065


namespace mrs_hilt_pie_arrangement_l3860_386094

/-- Given the number of pecan pies, apple pies, and rows, 
    calculate the number of pies in each row -/
def piesPerRow (pecanPies applePies rows : ℕ) : ℕ :=
  (pecanPies + applePies) / rows

/-- Theorem: Given 16 pecan pies, 14 apple pies, and 30 rows,
    the number of pies in each row is 1 -/
theorem mrs_hilt_pie_arrangement :
  piesPerRow 16 14 30 = 1 := by
  sorry

end mrs_hilt_pie_arrangement_l3860_386094


namespace quadratic_root_proof_l3860_386007

theorem quadratic_root_proof :
  let x : ℝ := (-5 + Real.sqrt (5^2 + 4*3*1)) / (2*3)
  3 * x^2 + 5 * x - 1 = 0 ∨
  let x : ℝ := (-5 - Real.sqrt (5^2 + 4*3*1)) / (2*3)
  3 * x^2 + 5 * x - 1 = 0 :=
by sorry

end quadratic_root_proof_l3860_386007


namespace min_photos_theorem_l3860_386027

/-- Represents a photo of two children -/
structure Photo where
  child1 : Nat
  child2 : Nat
  deriving Repr

/-- The set of all possible photos -/
def AllPhotos : Set Photo := sorry

/-- The set of photos with two boys -/
def BoyBoyPhotos : Set Photo := sorry

/-- The set of photos with two girls -/
def GirlGirlPhotos : Set Photo := sorry

/-- Predicate to check if two photos are the same -/
def SamePhoto (p1 p2 : Photo) : Prop := sorry

theorem min_photos_theorem (n : Nat) (photos : Fin n → Photo) :
  (∀ i : Fin n, photos i ∈ AllPhotos) →
  (n ≥ 33) →
  (∃ i : Fin n, photos i ∈ BoyBoyPhotos) ∨
  (∃ i : Fin n, photos i ∈ GirlGirlPhotos) ∨
  (∃ i j : Fin n, i ≠ j ∧ SamePhoto (photos i) (photos j)) := by
  sorry

#check min_photos_theorem

end min_photos_theorem_l3860_386027


namespace cos_alpha_plus_pi_fourth_l3860_386070

theorem cos_alpha_plus_pi_fourth (α : Real) :
  (∃ (x y : Real), x = 4 ∧ y = -3 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.cos (α + π/4) = 7 * Real.sqrt 2 / 10 := by
sorry

end cos_alpha_plus_pi_fourth_l3860_386070


namespace range_of_a_for_zero_in_interval_l3860_386086

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * abs x - 3 * a - 1

-- State the theorem
theorem range_of_a_for_zero_in_interval :
  ∀ a : ℝ, (∃ x₀ : ℝ, x₀ ∈ [-1, 1] ∧ f a x₀ = 0) → a ∈ [-1/2, -1/3] :=
by sorry

end range_of_a_for_zero_in_interval_l3860_386086


namespace function_properties_l3860_386033

noncomputable def f (x φ : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem function_properties (φ : ℝ) 
  (h1 : 0 < φ) (h2 : φ < Real.pi / 2) 
  (h3 : f (Real.pi / 12) φ = f (Real.pi / 4) φ) :
  (φ = Real.pi / 6) ∧ 
  (∀ x, f x φ = f (-x - Real.pi / 6) φ) ∧
  (∀ x ∈ Set.Ioo (-Real.pi / 12) (Real.pi / 6), 
    ∀ y ∈ Set.Ioo (-Real.pi / 12) (Real.pi / 6), 
    x < y → f x φ < f y φ) := by
  sorry

end function_properties_l3860_386033


namespace swimming_scenario_l3860_386074

/-- The number of weeks in the swimming scenario -/
def weeks : ℕ := 4

/-- Camden's total number of swims -/
def camden_total : ℕ := 16

/-- Susannah's total number of swims -/
def susannah_total : ℕ := 24

/-- Camden's swims per week -/
def camden_per_week : ℚ := camden_total / weeks

/-- Susannah's swims per week -/
def susannah_per_week : ℚ := susannah_total / weeks

theorem swimming_scenario :
  (susannah_per_week = camden_per_week + 2) ∧
  (camden_per_week * weeks = camden_total) ∧
  (susannah_per_week * weeks = susannah_total) :=
by sorry

end swimming_scenario_l3860_386074


namespace max_ab_linear_function_l3860_386001

/-- Given a linear function f(x) = ax + b where a and b are real numbers,
    if |f(x)| ≤ 1 for all x in [0, 1], then the maximum value of ab is 1/4. -/
theorem max_ab_linear_function (a b : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → |a * x + b| ≤ 1) →
  ab ≤ (1 : ℝ) / 4 ∧ ∃ a' b' : ℝ, (∀ x : ℝ, x ∈ Set.Icc 0 1 → |a' * x + b'| ≤ 1) ∧ a' * b' = (1 : ℝ) / 4 := by
  sorry

end max_ab_linear_function_l3860_386001


namespace cylinder_height_relation_l3860_386087

theorem cylinder_height_relation (r₁ h₁ r₂ h₂ : ℝ) :
  r₁ > 0 ∧ h₁ > 0 ∧ r₂ > 0 ∧ h₂ > 0 →
  r₂ = 1.2 * r₁ →
  π * r₁^2 * h₁ = π * r₂^2 * h₂ →
  h₁ = 1.44 * h₂ :=
by sorry

end cylinder_height_relation_l3860_386087


namespace sqrt_equation_solution_l3860_386015

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (3 + Real.sqrt x) = 4 → x = 169 := by
  sorry

end sqrt_equation_solution_l3860_386015


namespace count_numbers_with_property_l3860_386079

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

def satisfies_property (n : ℕ) : Prop :=
  is_two_digit n ∧ (n + reverse_digits n) % 13 = 0

theorem count_numbers_with_property :
  ∃ (S : Finset ℕ), (∀ n ∈ S, satisfies_property n) ∧ S.card = 6 :=
sorry

end count_numbers_with_property_l3860_386079


namespace range_of_a_l3860_386089

-- Define the sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x | x - a < 0}

-- State the theorem
theorem range_of_a (h : A ⊆ B a) : a ∈ Set.Ici 4 := by
  sorry

end range_of_a_l3860_386089


namespace height_difference_l3860_386085

theorem height_difference (height_B : ℝ) (height_A : ℝ) 
  (h : height_A = height_B * 1.25) : 
  (height_B - height_A) / height_A * 100 = -20 := by
  sorry

end height_difference_l3860_386085


namespace prob_draw_queen_l3860_386056

/-- Represents a standard deck of playing cards -/
structure Deck :=
  (cards : Nat)
  (ranks : Nat)
  (suits : Nat)

/-- Represents the number of a specific card in the deck -/
def cardsOfType (d : Deck) : Nat := d.suits

/-- A standard deck of cards -/
def standardDeck : Deck :=
  { cards := 52
  , ranks := 13
  , suits := 4 }

/-- The probability of drawing a specific card from the deck -/
def probDraw (d : Deck) : ℚ := (cardsOfType d : ℚ) / (d.cards : ℚ)

theorem prob_draw_queen (d : Deck := standardDeck) :
  probDraw d = 1 / 13 := by
  sorry

#eval probDraw standardDeck

end prob_draw_queen_l3860_386056


namespace f_sum_symmetric_l3860_386023

/-- A function f(x) = ax^4 + bx^2 + 5 where a and b are real constants -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^2 + 5

/-- Theorem: If f(20) = 3, then f(20) + f(-20) = 6 -/
theorem f_sum_symmetric (a b : ℝ) (h : f a b 20 = 3) : f a b 20 + f a b (-20) = 6 := by
  sorry

end f_sum_symmetric_l3860_386023


namespace compound_interest_problem_l3860_386059

theorem compound_interest_problem :
  ∃ (P r : ℝ), P > 0 ∧ r > 0 ∧ 
  P * (1 + r)^2 = 8840 ∧
  P * (1 + r)^3 = 9261 := by
sorry

end compound_interest_problem_l3860_386059


namespace abc_inequalities_l3860_386022

theorem abc_inequalities (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_prod : (a + 1) * (b + 1) * (c + 1) = 8) :
  a + b + c ≥ 3 ∧ a * b * c ≤ 1 ∧
  (a + b + c = 3 ∧ a * b * c = 1 ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end abc_inequalities_l3860_386022


namespace thomas_weight_vest_cost_l3860_386021

/-- Calculates the total cost for Thomas to increase his weight vest weight --/
def calculate_total_cost (initial_weight : ℕ) (increase_percentage : ℕ) (ingot_weight : ℕ) (ingot_cost : ℕ) : ℚ :=
  let additional_weight := initial_weight * increase_percentage / 100
  let num_ingots := (additional_weight + ingot_weight - 1) / ingot_weight
  let base_cost := num_ingots * ingot_cost
  let discounted_cost := 
    if num_ingots ≤ 10 then base_cost
    else if num_ingots ≤ 20 then base_cost * 80 / 100
    else if num_ingots ≤ 30 then base_cost * 75 / 100
    else base_cost * 70 / 100
  let taxed_cost :=
    if num_ingots ≤ 20 then discounted_cost * 105 / 100
    else if num_ingots ≤ 30 then discounted_cost * 103 / 100
    else discounted_cost * 101 / 100
  let shipping_fee :=
    if num_ingots * ingot_weight ≤ 20 then 10
    else if num_ingots * ingot_weight ≤ 40 then 15
    else 20
  taxed_cost + shipping_fee

/-- Theorem stating that the total cost for Thomas is $90.60 --/
theorem thomas_weight_vest_cost :
  calculate_total_cost 60 60 2 5 = 9060 / 100 :=
sorry

end thomas_weight_vest_cost_l3860_386021


namespace kenny_contribution_percentage_l3860_386054

/-- Represents the contributions and total cost for house painting -/
structure PaintingContributions where
  total_cost : ℕ
  judson_contribution : ℕ
  kenny_contribution : ℕ
  camilo_contribution : ℕ

/-- Defines the conditions for the house painting contributions -/
def valid_contributions (c : PaintingContributions) : Prop :=
  c.total_cost = 1900 ∧
  c.judson_contribution = 500 ∧
  c.kenny_contribution > c.judson_contribution ∧
  c.camilo_contribution = c.kenny_contribution + 200 ∧
  c.judson_contribution + c.kenny_contribution + c.camilo_contribution = c.total_cost

/-- Calculates the percentage difference between Kenny's and Judson's contributions -/
def percentage_difference (c : PaintingContributions) : ℚ :=
  (c.kenny_contribution - c.judson_contribution : ℚ) / c.judson_contribution * 100

/-- Theorem stating that Kenny contributed 20% more than Judson -/
theorem kenny_contribution_percentage (c : PaintingContributions)
  (h : valid_contributions c) : percentage_difference c = 20 := by
  sorry


end kenny_contribution_percentage_l3860_386054


namespace square_addition_l3860_386039

theorem square_addition (b : ℝ) : b^2 + b^2 = 2 * b^2 := by
  sorry

end square_addition_l3860_386039


namespace park_benches_l3860_386073

theorem park_benches (bench_capacity : ℕ) (people_sitting : ℕ) (spaces_available : ℕ) : 
  bench_capacity = 4 →
  people_sitting = 80 →
  spaces_available = 120 →
  (people_sitting + spaces_available) / bench_capacity = 50 := by
  sorry

end park_benches_l3860_386073


namespace circle_properties_l3860_386097

def circle_equation (x y : ℝ) : Prop :=
  x^2 - 4*y - 36 = -y^2 + 12*x + 16

def is_center (a b : ℝ) : Prop :=
  ∀ x y : ℝ, circle_equation x y ↔ (x - a)^2 + (y - b)^2 = (2 * Real.sqrt 23)^2

theorem circle_properties :
  ∃ a b : ℝ,
    is_center a b ∧
    a = 6 ∧
    b = 2 ∧
    a + b + 2 * Real.sqrt 23 = 8 + 2 * Real.sqrt 23 :=
by sorry

end circle_properties_l3860_386097


namespace base5_44_to_decimal_l3860_386077

/-- Converts a base-5 number to its decimal equivalent -/
def base5ToDecimal (d₁ d₀ : ℕ) : ℕ := d₁ * 5^1 + d₀ * 5^0

/-- The base-5 number 44₅ -/
def base5_44 : ℕ × ℕ := (4, 4)

theorem base5_44_to_decimal :
  base5ToDecimal base5_44.1 base5_44.2 = 24 := by sorry

end base5_44_to_decimal_l3860_386077


namespace chicken_duck_difference_l3860_386057

theorem chicken_duck_difference (total_birds ducks : ℕ) 
  (h1 : total_birds = 95) 
  (h2 : ducks = 32) : 
  total_birds - ducks - ducks = 31 := by
  sorry

end chicken_duck_difference_l3860_386057


namespace positive_2x2_square_exists_l3860_386082

/-- Represents a 50 by 50 grid of integers -/
def Grid := Fin 50 → Fin 50 → ℤ

/-- Represents a configuration G of 8 squares obtained by taking a 3 by 3 grid and removing the central square -/
def G (grid : Grid) (i j : Fin 48) : ℤ :=
  (grid i j) + (grid i (j+1)) + (grid i (j+2)) +
  (grid (i+1) j) + (grid (i+1) (j+2)) +
  (grid (i+2) j) + (grid (i+2) (j+1)) + (grid (i+2) (j+2))

/-- Represents a 2 by 2 square in the grid -/
def Square2x2 (grid : Grid) (i j : Fin 49) : ℤ :=
  (grid i j) + (grid i (j+1)) + (grid (i+1) j) + (grid (i+1) (j+1))

/-- The main theorem -/
theorem positive_2x2_square_exists (grid : Grid) 
  (h : ∀ i j : Fin 48, G grid i j > 0) :
  ∃ i j : Fin 49, Square2x2 grid i j > 0 := by
  sorry

end positive_2x2_square_exists_l3860_386082


namespace stratified_sampling_athletes_l3860_386003

theorem stratified_sampling_athletes (total_male : ℕ) (total_female : ℕ) 
  (selected_male : ℕ) (selected_female : ℕ) 
  (h1 : total_male = 56) (h2 : total_female = 42) (h3 : selected_male = 8) :
  (selected_male : ℚ) / total_male = selected_female / total_female → selected_female = 6 := by
  sorry

end stratified_sampling_athletes_l3860_386003


namespace gcd_digit_sum_theorem_l3860_386000

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem gcd_digit_sum_theorem : 
  let a := 4665 - 1305
  let b := 6905 - 4665
  let c := 6905 - 1305
  let gcd_result := Nat.gcd (Nat.gcd a b) c
  sum_of_digits gcd_result = 4 := by sorry

end gcd_digit_sum_theorem_l3860_386000


namespace cubic_equation_one_real_solution_l3860_386037

theorem cubic_equation_one_real_solution :
  ∀ (a : ℝ), ∃! (x : ℝ), x^3 - a*x^2 - 3*a*x + a^2 - 1 = 0 := by
  sorry

end cubic_equation_one_real_solution_l3860_386037


namespace xiao_ming_envelopes_l3860_386034

def red_envelopes : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def total_sum : ℕ := red_envelopes.sum

def each_person_sum : ℕ := total_sum / 3

def father_envelopes : List ℕ := [1, 3]
def mother_envelopes : List ℕ := [8, 9]

theorem xiao_ming_envelopes :
  ∀ (xm : List ℕ),
    xm.length = 4 →
    father_envelopes.length = 4 →
    mother_envelopes.length = 4 →
    xm.sum = each_person_sum →
    father_envelopes.sum = each_person_sum →
    mother_envelopes.sum = each_person_sum →
    (∀ x ∈ xm, x ∈ red_envelopes) →
    (∀ x ∈ father_envelopes, x ∈ red_envelopes) →
    (∀ x ∈ mother_envelopes, x ∈ red_envelopes) →
    (∀ x ∈ red_envelopes, x ∈ xm ∨ x ∈ father_envelopes ∨ x ∈ mother_envelopes) →
    6 ∈ xm ∧ 11 ∈ xm :=
by
  sorry

#check xiao_ming_envelopes

end xiao_ming_envelopes_l3860_386034


namespace plane_existence_and_uniqueness_l3860_386095

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The first bisector plane -/
def firstBisectorPlane : Plane3D := sorry

/-- Check if a point lies in a plane -/
def pointInPlane (p : Point3D) (plane : Plane3D) : Prop := sorry

/-- First angle of projection of a plane -/
def firstProjectionAngle (plane : Plane3D) : ℝ := sorry

/-- Angle between first and second trace lines of a plane -/
def traceLinesAngle (plane : Plane3D) : ℝ := sorry

/-- Theorem: Existence and uniqueness of a plane with given properties -/
theorem plane_existence_and_uniqueness 
  (P : Point3D) 
  (α β : ℝ) 
  (h_P : pointInPlane P firstBisectorPlane) :
  ∃! s : Plane3D, 
    pointInPlane P s ∧ 
    firstProjectionAngle s = α ∧ 
    traceLinesAngle s = β := by
  sorry

end plane_existence_and_uniqueness_l3860_386095


namespace sum_of_digits_of_2012_power_l3860_386075

def A : ℕ := 2012^2012

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def B : ℕ := sum_of_digits A
def C : ℕ := sum_of_digits B
def D : ℕ := sum_of_digits C

theorem sum_of_digits_of_2012_power : D = 7 := by
  sorry

end sum_of_digits_of_2012_power_l3860_386075


namespace no_natural_solution_l3860_386099

theorem no_natural_solution :
  ¬∃ (n m : ℕ), (n + 1) * (2 * n + 1) = 18 * m^2 := by
sorry

end no_natural_solution_l3860_386099


namespace pencil_pen_cost_l3860_386083

/-- Given the cost of pencils and pens, prove the cost of one pencil and two pens -/
theorem pencil_pen_cost (p q : ℝ) 
  (h1 : 3 * p + 4 * q = 3.20)
  (h2 : 2 * p + 3 * q = 2.50) :
  p + 2 * q = 1.80 := by
  sorry

end pencil_pen_cost_l3860_386083


namespace product_expansion_sum_l3860_386071

theorem product_expansion_sum (a b c d : ℝ) : 
  (∀ x, (4 * x^2 - 6 * x + 3) * (8 - 3 * x) = a * x^3 + b * x^2 + c * x + d) →
  8 * a + 4 * b + 2 * c + d = 14 := by
sorry

end product_expansion_sum_l3860_386071


namespace square_area_of_adjacent_corners_l3860_386064

-- Define points A and B
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (5, 6)

-- Define the square area function
def squareArea (p1 p2 : ℝ × ℝ) : ℝ :=
  let dx := p2.1 - p1.1
  let dy := p2.2 - p1.2
  (dx * dx + dy * dy)

-- Theorem statement
theorem square_area_of_adjacent_corners :
  squareArea A B = 32 := by
  sorry

end square_area_of_adjacent_corners_l3860_386064


namespace michelle_final_crayons_l3860_386020

/-- 
Given:
- Michelle initially has x crayons
- Janet initially has y crayons
- Both Michelle and Janet receive z more crayons each
- Janet gives all of her crayons to Michelle

Prove that Michelle will have x + y + 2z crayons in total.
-/
theorem michelle_final_crayons (x y z : ℕ) : x + z + (y + z) = x + y + 2*z :=
by sorry

end michelle_final_crayons_l3860_386020


namespace pole_length_theorem_l3860_386029

/-- The length of a pole after two cuts -/
def pole_length_after_cuts (initial_length : ℝ) (first_cut_percentage : ℝ) (second_cut_percentage : ℝ) : ℝ :=
  initial_length * (1 - first_cut_percentage) * (1 - second_cut_percentage)

/-- Theorem stating that a 20-meter pole, after cuts of 30% and 25%, will be 10.5 meters long -/
theorem pole_length_theorem :
  pole_length_after_cuts 20 0.3 0.25 = 10.5 := by
  sorry

#eval pole_length_after_cuts 20 0.3 0.25

end pole_length_theorem_l3860_386029


namespace daniel_earnings_l3860_386024

-- Define the delivery schedule and prices
def monday_fabric : ℕ := 20
def monday_yarn : ℕ := 15
def tuesday_fabric : ℕ := 2 * monday_fabric
def tuesday_yarn : ℕ := monday_yarn + 10
def wednesday_fabric : ℕ := tuesday_fabric / 4
def wednesday_yarn : ℕ := tuesday_yarn / 2 + 1  -- Rounded up

def fabric_price : ℕ := 2
def yarn_price : ℕ := 3

-- Calculate total yards of fabric and yarn
def total_fabric : ℕ := monday_fabric + tuesday_fabric + wednesday_fabric
def total_yarn : ℕ := monday_yarn + tuesday_yarn + wednesday_yarn

-- Calculate total earnings
def total_earnings : ℕ := fabric_price * total_fabric + yarn_price * total_yarn

-- Theorem to prove
theorem daniel_earnings : total_earnings = 299 := by
  sorry

end daniel_earnings_l3860_386024


namespace specific_rhombus_area_l3860_386017

/-- Represents a rhombus with given properties -/
structure Rhombus where
  side_length : ℝ
  diagonal_difference : ℝ
  diagonals_perpendicular : Bool

/-- Calculates the area of a rhombus given its properties -/
def rhombus_area (r : Rhombus) : ℝ :=
  sorry

/-- Theorem stating the area of a specific rhombus -/
theorem specific_rhombus_area :
  let r : Rhombus := { 
    side_length := Real.sqrt 145,
    diagonal_difference := 8,
    diagonals_perpendicular := true
  }
  rhombus_area r = (Real.sqrt 274 * (Real.sqrt 274 - 4)) / 4 := by
  sorry

end specific_rhombus_area_l3860_386017


namespace equal_roots_quadratic_l3860_386031

/-- For a quadratic equation kx^2 + 2x - 1 = 0 to have two equal real roots, k must equal -1 -/
theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, k * x^2 + 2 * x - 1 = 0 ∧ 
   ∀ y : ℝ, k * y^2 + 2 * y - 1 = 0 → y = x) → 
  k = -1 :=
sorry

end equal_roots_quadratic_l3860_386031


namespace irrational_approximation_l3860_386004

theorem irrational_approximation 
  (r₁ r₂ : ℝ) 
  (h_irrational : Irrational (r₁ / r₂)) :
  ∀ (x p : ℝ), p > 0 → ∃ (k₁ k₂ : ℤ), |x - (↑k₁ * r₁ + ↑k₂ * r₂)| < p := by
  sorry

end irrational_approximation_l3860_386004


namespace consecutive_four_product_ending_l3860_386092

theorem consecutive_four_product_ending (n : ℕ) :
  ∃ (k : ℕ), (n * (n + 1) * (n + 2) * (n + 3) % 1000 = 24 ∧ k = n * (n + 1) * (n + 2) * (n + 3) / 1000) ∨
              (n * (n + 1) * (n + 2) * (n + 3) % 10 = 0 ∧ (k = n * (n + 1) * (n + 2) * (n + 3) / 10) ∧ k % 4 = 0) :=
by sorry

end consecutive_four_product_ending_l3860_386092


namespace jean_grandchildren_l3860_386049

/-- The number of cards Jean buys for each grandchild per year -/
def cards_per_grandchild : ℕ := 2

/-- The amount of money Jean puts in each card -/
def money_per_card : ℕ := 80

/-- The total amount of money Jean gives away to her grandchildren per year -/
def total_money_given : ℕ := 480

/-- The number of grandchildren Jean has -/
def num_grandchildren : ℕ := total_money_given / (cards_per_grandchild * money_per_card)

theorem jean_grandchildren :
  num_grandchildren = 3 :=
sorry

end jean_grandchildren_l3860_386049
