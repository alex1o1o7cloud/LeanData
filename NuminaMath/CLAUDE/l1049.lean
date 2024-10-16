import Mathlib

namespace NUMINAMATH_CALUDE_arrange_13_blue_5_red_l1049_104962

/-- The number of ways to arrange blue and red balls with constraints -/
def arrange_balls (blue_balls red_balls : ℕ) : ℕ :=
  Nat.choose (blue_balls - red_balls + 1 + red_balls) (red_balls + 1)

/-- Theorem: Arranging 13 blue balls and 5 red balls with constraints yields 2002 ways -/
theorem arrange_13_blue_5_red :
  arrange_balls 13 5 = 2002 := by
  sorry

#eval arrange_balls 13 5

end NUMINAMATH_CALUDE_arrange_13_blue_5_red_l1049_104962


namespace NUMINAMATH_CALUDE_remaining_quarters_l1049_104912

-- Define the initial amount, total spent, and value of a quarter
def initial_amount : ℚ := 40
def total_spent : ℚ := 32.25
def quarter_value : ℚ := 0.25

-- Theorem to prove
theorem remaining_quarters : 
  (initial_amount - total_spent) / quarter_value = 31 := by
  sorry

end NUMINAMATH_CALUDE_remaining_quarters_l1049_104912


namespace NUMINAMATH_CALUDE_fraction_equality_l1049_104961

theorem fraction_equality (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) 
  (h3 : (4*x - y) / (3*x + 2*y) = 3) : 
  (3*x - 2*y) / (4*x + y) = 31/23 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l1049_104961


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1049_104989

open Set

def M : Set ℝ := {x | -1 < x ∧ x < 3}
def N : Set ℝ := {x | x < 1}

theorem intersection_of_M_and_N : M ∩ N = Ioo (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1049_104989


namespace NUMINAMATH_CALUDE_discount_profit_theorem_l1049_104969

theorem discount_profit_theorem (cost : ℝ) (h_cost_pos : cost > 0) : 
  let discount_rate : ℝ := 0.1
  let profit_rate_with_discount : ℝ := 0.2
  let selling_price_with_discount : ℝ := (1 - discount_rate) * ((1 + profit_rate_with_discount) * cost)
  let selling_price_without_discount : ℝ := selling_price_with_discount / (1 - discount_rate)
  let profit_without_discount : ℝ := selling_price_without_discount - cost
  let profit_rate_without_discount : ℝ := profit_without_discount / cost
  profit_rate_without_discount = 1/3 := by sorry

end NUMINAMATH_CALUDE_discount_profit_theorem_l1049_104969


namespace NUMINAMATH_CALUDE_katie_candy_count_l1049_104977

theorem katie_candy_count (sister_candy : ℕ) (eaten_candy : ℕ) (remaining_candy : ℕ) 
  (h1 : sister_candy = 23)
  (h2 : eaten_candy = 8)
  (h3 : remaining_candy = 23) :
  ∃ (katie_candy : ℕ), katie_candy = 8 ∧ katie_candy + sister_candy - eaten_candy = remaining_candy :=
by sorry

end NUMINAMATH_CALUDE_katie_candy_count_l1049_104977


namespace NUMINAMATH_CALUDE_triangle_count_l1049_104965

theorem triangle_count : ∃ (n : ℕ), n = 36 ∧ 
  n = (Finset.filter (fun p : ℕ × ℕ => 
    p.1 ≤ p.2 ∧ p.1 + p.2 > 11) 
    (Finset.product (Finset.range 12) (Finset.range 12))).card :=
by sorry

end NUMINAMATH_CALUDE_triangle_count_l1049_104965


namespace NUMINAMATH_CALUDE_line_contains_point_l1049_104979

/-- Given a line represented by the equation -3/4 - 3kx = 7y that contains the point (1/3, -8),
    prove that k = 55.25 -/
theorem line_contains_point (k : ℝ) : 
  (-3/4 : ℝ) - 3 * k * (1/3 : ℝ) = 7 * (-8 : ℝ) → k = 55.25 := by
  sorry

end NUMINAMATH_CALUDE_line_contains_point_l1049_104979


namespace NUMINAMATH_CALUDE_min_value_of_f_l1049_104978

/-- The function to be minimized -/
def f (x y : ℝ) : ℝ := x^2 + 4*x*y + 5*y^2 - 8*x + 6*y + 2

/-- Theorem stating that the minimum value of f is -7 -/
theorem min_value_of_f :
  ∃ (x y : ℝ), ∀ (a b : ℝ), f x y ≤ f a b ∧ f x y = -7 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1049_104978


namespace NUMINAMATH_CALUDE_fan_shooting_theorem_l1049_104934

/-- Represents a fan with four blades rotating at a given speed -/
structure Fan :=
  (revolution_speed : ℝ)
  (num_blades : ℕ)

/-- Represents a bullet trajectory -/
structure BulletTrajectory :=
  (angle : ℝ)
  (speed : ℝ)

/-- Checks if a bullet trajectory intersects all blades of a fan -/
def intersects_all_blades (f : Fan) (bt : BulletTrajectory) : Prop :=
  sorry

/-- The main theorem stating that there exists a bullet trajectory that intersects all blades -/
theorem fan_shooting_theorem (f : Fan) 
  (h1 : f.revolution_speed = 50)
  (h2 : f.num_blades = 4) : 
  ∃ (bt : BulletTrajectory), intersects_all_blades f bt :=
sorry

end NUMINAMATH_CALUDE_fan_shooting_theorem_l1049_104934


namespace NUMINAMATH_CALUDE_complex_number_purely_imaginary_l1049_104984

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_number_purely_imaginary (m : ℝ) :
  let z : ℂ := Complex.mk (m^2 - m) m
  is_purely_imaginary z → m = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_number_purely_imaginary_l1049_104984


namespace NUMINAMATH_CALUDE_smaller_circle_circumference_l1049_104926

/-- Given a square and two circles with specific relationships, 
    prove the circumference of the smaller circle -/
theorem smaller_circle_circumference 
  (square_area : ℝ) 
  (larger_radius smaller_radius : ℝ) 
  (h1 : square_area = 784)
  (h2 : square_area = (2 * larger_radius)^2)
  (h3 : larger_radius = (7/3) * smaller_radius) : 
  2 * Real.pi * smaller_radius = 12 * Real.pi := by
sorry

end NUMINAMATH_CALUDE_smaller_circle_circumference_l1049_104926


namespace NUMINAMATH_CALUDE_painting_price_increase_l1049_104953

theorem painting_price_increase (x : ℝ) : 
  (1 + x / 100) * (1 - 0.25) = 0.9 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_painting_price_increase_l1049_104953


namespace NUMINAMATH_CALUDE_solve_potato_problem_l1049_104968

def potatoesProblem (initialPotatoes : ℕ) (ginaAmount : ℕ) : Prop :=
  let tomAmount := 2 * ginaAmount
  let anneAmount := tomAmount / 3
  let remainingPotatoes := initialPotatoes - (ginaAmount + tomAmount + anneAmount)
  remainingPotatoes = 47

theorem solve_potato_problem :
  potatoesProblem 300 69 := by
  sorry

end NUMINAMATH_CALUDE_solve_potato_problem_l1049_104968


namespace NUMINAMATH_CALUDE_triangle_right_angled_from_arithmetic_progression_l1049_104970

/-- Given a triangle with side lengths a, b, c, and incircle diameter 2r
    forming an arithmetic progression, prove that the triangle is right-angled. -/
theorem triangle_right_angled_from_arithmetic_progression 
  (a b c r : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_arithmetic : ∃ (d : ℝ), b = a + d ∧ c = b + d ∧ 2*r = c + d) :
  ∃ (A B C : ℝ), A + B + C = π ∧ max A B = π/2 ∧ max B C = π/2 ∧ max C A = π/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_right_angled_from_arithmetic_progression_l1049_104970


namespace NUMINAMATH_CALUDE_gdp_growth_problem_l1049_104996

/-- Calculates the GDP after compound growth -/
def gdp_growth (initial_gdp : ℝ) (growth_rate : ℝ) (years : ℕ) : ℝ :=
  initial_gdp * (1 + growth_rate) ^ years

/-- The GDP growth problem -/
theorem gdp_growth_problem :
  let initial_gdp : ℝ := 9593.3
  let growth_rate : ℝ := 0.073
  let years : ℕ := 4
  let final_gdp : ℝ := gdp_growth initial_gdp growth_rate years
  ∃ ε > 0, |final_gdp - 127254| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_gdp_growth_problem_l1049_104996


namespace NUMINAMATH_CALUDE_doughnut_cost_calculation_l1049_104918

/-- Calculates the total cost of doughnuts for a class -/
theorem doughnut_cost_calculation (total_students : ℕ) 
  (chocolate_lovers : ℕ) (glazed_lovers : ℕ) 
  (chocolate_cost : ℕ) (glazed_cost : ℕ) : 
  total_students = 25 →
  chocolate_lovers = 10 →
  glazed_lovers = 15 →
  chocolate_cost = 2 →
  glazed_cost = 1 →
  chocolate_lovers * chocolate_cost + glazed_lovers * glazed_cost = 35 :=
by
  sorry

#check doughnut_cost_calculation

end NUMINAMATH_CALUDE_doughnut_cost_calculation_l1049_104918


namespace NUMINAMATH_CALUDE_grandpa_mingming_age_ratio_l1049_104935

/-- Given Grandpa's and Mingming's current ages, prove that next year Grandpa's age will be 11 times Mingming's age -/
theorem grandpa_mingming_age_ratio (grandpa_age mingming_age : ℕ) 
  (h1 : grandpa_age = 65) (h2 : mingming_age = 5) : 
  (grandpa_age + 1) = 11 * (mingming_age + 1) := by
  sorry

end NUMINAMATH_CALUDE_grandpa_mingming_age_ratio_l1049_104935


namespace NUMINAMATH_CALUDE_parallel_vectors_dot_product_l1049_104976

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- The dot product of two 2D vectors -/
def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

theorem parallel_vectors_dot_product :
  ∀ x : ℝ, 
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, -4)
  parallel a b → dot_product a b = -10 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_dot_product_l1049_104976


namespace NUMINAMATH_CALUDE_adult_ticket_price_l1049_104975

theorem adult_ticket_price
  (total_tickets : ℕ)
  (senior_price : ℚ)
  (total_receipts : ℚ)
  (senior_tickets : ℕ)
  (h1 : total_tickets = 529)
  (h2 : senior_price = 15)
  (h3 : total_receipts = 9745)
  (h4 : senior_tickets = 348) :
  (total_receipts - senior_price * senior_tickets) / (total_tickets - senior_tickets) = 25 := by
  sorry

end NUMINAMATH_CALUDE_adult_ticket_price_l1049_104975


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l1049_104900

theorem negation_of_existential_proposition :
  (¬∃ x₀ : ℝ, x₀ ∈ Set.Icc (-3) 3 ∧ x₀^2 + 2*x₀ + 1 ≤ 0) ↔
  (∀ x₀ : ℝ, x₀ ∈ Set.Icc (-3) 3 → x₀^2 + 2*x₀ + 1 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l1049_104900


namespace NUMINAMATH_CALUDE_norm_took_110_photos_l1049_104994

/-- The number of photos taken by each photographer --/
structure PhotoCount where
  lisa : ℕ
  mike : ℕ
  norm : ℕ

/-- The conditions of the problem --/
def satisfies_conditions (p : PhotoCount) : Prop :=
  p.lisa + p.mike = p.mike + p.norm - 60 ∧
  p.norm = 2 * p.lisa + 10

/-- The theorem stating that Norm took 110 photos --/
theorem norm_took_110_photos (p : PhotoCount) 
  (h : satisfies_conditions p) : p.norm = 110 := by
  sorry

end NUMINAMATH_CALUDE_norm_took_110_photos_l1049_104994


namespace NUMINAMATH_CALUDE_students_with_one_problem_l1049_104946

/-- Represents the number of problems created by students from each course -/
def ProblemsCourses : Type := Fin 5 → ℕ

/-- Represents the number of students in each course -/
def StudentsCourses : Type := Fin 5 → ℕ

/-- The total number of students -/
def TotalStudents : ℕ := 30

/-- The total number of problems created -/
def TotalProblems : ℕ := 40

/-- The condition that students from different courses created different numbers of problems -/
def DifferentProblems (p : ProblemsCourses) : Prop :=
  ∀ i j, i ≠ j → p i ≠ p j

/-- The condition that the total number of problems created matches the given total -/
def MatchesTotalProblems (p : ProblemsCourses) (s : StudentsCourses) : Prop :=
  (Finset.sum Finset.univ (λ i => p i * s i)) = TotalProblems

/-- The condition that the total number of students matches the given total -/
def MatchesTotalStudents (s : StudentsCourses) : Prop :=
  (Finset.sum Finset.univ s) = TotalStudents

theorem students_with_one_problem
  (p : ProblemsCourses)
  (s : StudentsCourses)
  (h1 : DifferentProblems p)
  (h2 : MatchesTotalProblems p s)
  (h3 : MatchesTotalStudents s) :
  (Finset.filter (λ i => p i = 1) Finset.univ).card = 26 := by
  sorry

end NUMINAMATH_CALUDE_students_with_one_problem_l1049_104946


namespace NUMINAMATH_CALUDE_inequality_counterexample_l1049_104915

theorem inequality_counterexample :
  ∃ (a b c d : ℝ), a > b ∧ c > d ∧ a + d ≤ b + c := by
  sorry

end NUMINAMATH_CALUDE_inequality_counterexample_l1049_104915


namespace NUMINAMATH_CALUDE_joan_remaining_oranges_l1049_104959

theorem joan_remaining_oranges (joan_initial : ℕ) (sara_sold : ℕ) (joan_remaining : ℕ) : 
  joan_initial = 37 → sara_sold = 10 → joan_remaining = joan_initial - sara_sold → joan_remaining = 27 := by
  sorry

end NUMINAMATH_CALUDE_joan_remaining_oranges_l1049_104959


namespace NUMINAMATH_CALUDE_jessicas_carrots_l1049_104950

theorem jessicas_carrots (joan_carrots : ℕ) (total_carrots : ℕ) 
  (h1 : joan_carrots = 29) 
  (h2 : total_carrots = 40) : 
  total_carrots - joan_carrots = 11 := by
  sorry

end NUMINAMATH_CALUDE_jessicas_carrots_l1049_104950


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_negative_twenty_two_thirds_l1049_104931

theorem greatest_integer_less_than_negative_twenty_two_thirds :
  Int.floor (-22 / 3) = -8 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_negative_twenty_two_thirds_l1049_104931


namespace NUMINAMATH_CALUDE_marie_erasers_l1049_104922

/-- Given Marie's eraser situation, prove that she ends up with 755 erasers. -/
theorem marie_erasers (initial : ℕ) (lost : ℕ) (packs_bought : ℕ) (erasers_per_pack : ℕ) 
  (h1 : initial = 950)
  (h2 : lost = 420)
  (h3 : packs_bought = 3)
  (h4 : erasers_per_pack = 75) : 
  initial - lost + packs_bought * erasers_per_pack = 755 := by
  sorry

#check marie_erasers

end NUMINAMATH_CALUDE_marie_erasers_l1049_104922


namespace NUMINAMATH_CALUDE_rectangle_area_l1049_104941

/-- Given a rectangle where the length is four times the width and the perimeter is 200 cm,
    prove that its area is 1600 square centimeters. -/
theorem rectangle_area (w : ℝ) (h1 : w > 0) : 
  let l := 4 * w
  2 * l + 2 * w = 200 →
  l * w = 1600 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l1049_104941


namespace NUMINAMATH_CALUDE_functional_equation_implies_ge_l1049_104942

/-- A function f: ℝ⁺ → ℝ⁺ satisfying f(f(x)) + x = f(2x) for all x > 0 -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x > 0, f x > 0 ∧ f (f x) + x = f (2 * x)

/-- Theorem: If f satisfies the functional equation, then f(x) ≥ x for all x > 0 -/
theorem functional_equation_implies_ge (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∀ x > 0, f x ≥ x := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_implies_ge_l1049_104942


namespace NUMINAMATH_CALUDE_problem_solution_l1049_104951

def f (x : ℝ) : ℝ := |x - 1| + |x + 1| - 1

theorem problem_solution :
  (∀ x : ℝ, f x ≤ x + 1 ↔ x ∈ Set.Icc 0 2) ∧
  ((∀ a : ℝ, a ≠ 0 → ∀ x : ℝ, f x ≥ (|a + 1| - |2*a - 1|) / |a|) →
   ∀ x : ℝ, f x ≥ 3 ↔ x ∈ Set.Iic (-2) ∪ Set.Ici 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1049_104951


namespace NUMINAMATH_CALUDE_nested_square_root_value_l1049_104980

theorem nested_square_root_value :
  ∀ y : ℝ, y = Real.sqrt (3 + y) → y = (1 + Real.sqrt 13) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_square_root_value_l1049_104980


namespace NUMINAMATH_CALUDE_problem_types_not_mutually_exclusive_l1049_104938

/-- Represents a mathematical problem type -/
inductive ProblemType
  | Proof
  | Computation
  | Construction

/-- Represents a mathematical problem -/
structure Problem where
  type : ProblemType
  hasProofElement : Bool
  hasComputationElement : Bool
  hasConstructionElement : Bool

/-- Theorem stating that problem types are not mutually exclusive -/
theorem problem_types_not_mutually_exclusive :
  ∃ (p : Problem), (p.type = ProblemType.Proof ∨ p.type = ProblemType.Computation ∨ p.type = ProblemType.Construction) ∧
    p.hasProofElement ∧ p.hasComputationElement ∧ p.hasConstructionElement :=
sorry

end NUMINAMATH_CALUDE_problem_types_not_mutually_exclusive_l1049_104938


namespace NUMINAMATH_CALUDE_skateboard_padding_cost_increase_l1049_104902

/-- Calculates the percent increase in the combined cost of a skateboard and padding set. -/
theorem skateboard_padding_cost_increase 
  (skateboard_cost : ℝ) 
  (padding_cost : ℝ) 
  (skateboard_increase : ℝ) 
  (padding_increase : ℝ) : 
  skateboard_cost = 120 →
  padding_cost = 30 →
  skateboard_increase = 0.08 →
  padding_increase = 0.15 →
  let new_skateboard_cost := skateboard_cost * (1 + skateboard_increase)
  let new_padding_cost := padding_cost * (1 + padding_increase)
  let original_total := skateboard_cost + padding_cost
  let new_total := new_skateboard_cost + new_padding_cost
  (new_total - original_total) / original_total = 0.094 := by
  sorry

end NUMINAMATH_CALUDE_skateboard_padding_cost_increase_l1049_104902


namespace NUMINAMATH_CALUDE_b_alone_time_l1049_104945

-- Define the work rates
def work_rate_b : ℚ := 1
def work_rate_a : ℚ := 2 * work_rate_b
def work_rate_c : ℚ := 3 * work_rate_a

-- Define the total work (completed job)
def total_work : ℚ := 1

-- Define the time taken by all three together
def total_time : ℚ := 9

-- Theorem to prove
theorem b_alone_time (h1 : work_rate_a = 2 * work_rate_b)
                     (h2 : work_rate_c = 3 * work_rate_a)
                     (h3 : (work_rate_a + work_rate_b + work_rate_c) * total_time = total_work) :
  total_work / work_rate_b = 81 := by
  sorry


end NUMINAMATH_CALUDE_b_alone_time_l1049_104945


namespace NUMINAMATH_CALUDE_min_visible_pairs_155_birds_l1049_104990

/-- The number of birds on the circle -/
def num_birds : ℕ := 155

/-- The visibility threshold in degrees -/
def visibility_threshold : ℝ := 10

/-- A function that calculates the minimum number of mutually visible bird pairs -/
def min_visible_pairs (n : ℕ) (threshold : ℝ) : ℕ :=
  sorry

/-- Theorem stating the minimum number of mutually visible bird pairs -/
theorem min_visible_pairs_155_birds :
  min_visible_pairs num_birds visibility_threshold = 270 :=
sorry

end NUMINAMATH_CALUDE_min_visible_pairs_155_birds_l1049_104990


namespace NUMINAMATH_CALUDE_alice_nike_sales_alice_nike_sales_proof_l1049_104909

/-- Proves that Alice sold 8 Nike shoes given the problem conditions -/
theorem alice_nike_sales : Int → Prop :=
  fun x =>
    let quota : Int := 1000
    let adidas_price : Int := 45
    let nike_price : Int := 60
    let reebok_price : Int := 35
    let adidas_sold : Int := 6
    let reebok_sold : Int := 9
    let over_goal : Int := 65
    (adidas_price * adidas_sold + nike_price * x + reebok_price * reebok_sold = quota + over_goal) →
    x = 8

/-- Proof of the theorem -/
theorem alice_nike_sales_proof : ∃ x, alice_nike_sales x :=
  sorry

end NUMINAMATH_CALUDE_alice_nike_sales_alice_nike_sales_proof_l1049_104909


namespace NUMINAMATH_CALUDE_abc_value_l1049_104993

theorem abc_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (eq1 : a + 1/b = 5)
  (eq2 : b + 1/c = 2)
  (eq3 : c + 1/a = 3) :
  a * b * c = 1 := by
sorry

end NUMINAMATH_CALUDE_abc_value_l1049_104993


namespace NUMINAMATH_CALUDE_aaron_guitar_loan_l1049_104901

/-- Calculates the total amount owed for a loan with monthly payments and interest. -/
def totalAmountOwed (monthlyPayment : ℝ) (numberOfMonths : ℕ) (interestRate : ℝ) : ℝ :=
  let totalWithoutInterest := monthlyPayment * numberOfMonths
  let interestAmount := totalWithoutInterest * interestRate
  totalWithoutInterest + interestAmount

/-- Theorem stating that given the specific conditions of Aaron's guitar purchase,
    the total amount owed is $1320. -/
theorem aaron_guitar_loan :
  totalAmountOwed 100 12 0.1 = 1320 := by
  sorry

end NUMINAMATH_CALUDE_aaron_guitar_loan_l1049_104901


namespace NUMINAMATH_CALUDE_function_property_l1049_104964

def is_valid_function (f : ℕ+ → ℝ) : Prop :=
  ∀ k : ℕ+, f k ≥ k^2 → f (k + 1) ≥ (k + 1)^2

theorem function_property (f : ℕ+ → ℝ) (h : is_valid_function f) (h4 : f 4 ≥ 25) :
  ∀ k : ℕ+, k ≥ 4 → f k ≥ k^2 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l1049_104964


namespace NUMINAMATH_CALUDE_balloon_difference_is_two_l1049_104956

/-- The number of balloons Allan brought to the park -/
def allan_balloons : ℕ := 5

/-- The number of balloons Jake brought to the park -/
def jake_balloons : ℕ := 3

/-- The difference in the number of balloons between Allan and Jake -/
def balloon_difference : ℕ := allan_balloons - jake_balloons

theorem balloon_difference_is_two : balloon_difference = 2 := by sorry

end NUMINAMATH_CALUDE_balloon_difference_is_two_l1049_104956


namespace NUMINAMATH_CALUDE_cubic_polynomial_roots_l1049_104999

theorem cubic_polynomial_roots (x₁ x₂ x₃ s t u : ℝ) 
  (h₁ : x₁ + x₂ + x₃ = s) 
  (h₂ : x₁ * x₂ + x₂ * x₃ + x₃ * x₁ = t) 
  (h₃ : x₁ * x₂ * x₃ = u) : 
  (X : ℝ) → (X - x₁) * (X - x₂) * (X - x₃) = X^3 - s*X^2 + t*X - u := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_roots_l1049_104999


namespace NUMINAMATH_CALUDE_max_students_on_field_trip_l1049_104958

def budget : ℕ := 350
def bus_rental : ℕ := 100
def admission_cost : ℕ := 10

theorem max_students_on_field_trip : 
  (budget - bus_rental) / admission_cost = 25 := by
  sorry

end NUMINAMATH_CALUDE_max_students_on_field_trip_l1049_104958


namespace NUMINAMATH_CALUDE_work_earnings_equation_l1049_104910

theorem work_earnings_equation (t : ℚ) : 
  (t + 2) * (4 * t - 4) = (4 * t - 2) * (t + 3) + 3 → t = -14/9 := by
  sorry

end NUMINAMATH_CALUDE_work_earnings_equation_l1049_104910


namespace NUMINAMATH_CALUDE_train_length_calculation_l1049_104923

/-- The length of two trains given their speeds and overtaking time -/
theorem train_length_calculation (faster_speed slower_speed : ℝ) (overtake_time : ℝ) : 
  faster_speed = 46 →
  slower_speed = 36 →
  overtake_time = 45 →
  ∃ (train_length : ℝ), train_length = 62.5 := by
  sorry


end NUMINAMATH_CALUDE_train_length_calculation_l1049_104923


namespace NUMINAMATH_CALUDE_tetrahedron_sum_l1049_104924

-- Define the tetrahedron with four positive integers on its faces
def Tetrahedron (a b c d : ℕ+) : Prop :=
  -- The sum of the products of each combination of three numbers is 770
  a.val * b.val * c.val + a.val * b.val * d.val + a.val * c.val * d.val + b.val * c.val * d.val = 770

-- Theorem statement
theorem tetrahedron_sum (a b c d : ℕ+) (h : Tetrahedron a b c d) :
  a.val + b.val + c.val + d.val = 57 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_sum_l1049_104924


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_slopes_l1049_104917

/-- Given a line y = mx - 3 intersecting the ellipse 4x^2 + 25y^2 = 100,
    prove that the possible slopes m satisfy m^2 ≥ 1/5 -/
theorem line_ellipse_intersection_slopes 
  (m : ℝ) -- slope of the line
  (h : ∃ x y : ℝ, y = m * x - 3 ∧ 4 * x^2 + 25 * y^2 = 100) -- intersection condition
  : m^2 ≥ 1/5 := by
  sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_slopes_l1049_104917


namespace NUMINAMATH_CALUDE_power_three_nineteen_mod_ten_l1049_104947

theorem power_three_nineteen_mod_ten : 3^19 % 10 = 7 := by sorry

end NUMINAMATH_CALUDE_power_three_nineteen_mod_ten_l1049_104947


namespace NUMINAMATH_CALUDE_f_neg_l1049_104973

-- Define an odd function f on the real numbers
def f : ℝ → ℝ := sorry

-- State the properties of f
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_pos : ∀ x > 0, f x = x * (1 - x)

-- Theorem to prove
theorem f_neg : ∀ x < 0, f x = x * (1 + x) := by sorry

end NUMINAMATH_CALUDE_f_neg_l1049_104973


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1049_104967

theorem polynomial_simplification (r : ℝ) :
  (2 * r^3 + 4 * r^2 + 5 * r - 3) - (r^3 + 6 * r^2 + 8 * r - 7) = r^3 - 2 * r^2 - 3 * r + 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1049_104967


namespace NUMINAMATH_CALUDE_both_selected_probability_l1049_104971

def prob_X : ℚ := 1/5
def prob_Y : ℚ := 2/7

theorem both_selected_probability : prob_X * prob_Y = 2/35 := by
  sorry

end NUMINAMATH_CALUDE_both_selected_probability_l1049_104971


namespace NUMINAMATH_CALUDE_remainder_73_power_73_plus_73_mod_137_l1049_104940

theorem remainder_73_power_73_plus_73_mod_137 :
  ∃ k : ℤ, 73^73 + 73 = 137 * k + 9 :=
by
  sorry

end NUMINAMATH_CALUDE_remainder_73_power_73_plus_73_mod_137_l1049_104940


namespace NUMINAMATH_CALUDE_fraction_problem_l1049_104983

theorem fraction_problem (x : ℝ) : 
  (x * 7000 - (1 / 1000) * 7000 = 700) ↔ (x = 0.101) :=
by sorry

end NUMINAMATH_CALUDE_fraction_problem_l1049_104983


namespace NUMINAMATH_CALUDE_cube_collinear_triples_l1049_104933

/-- Represents a point in a cube -/
inductive CubePoint
  | Vertex
  | EdgeMidpoint
  | FaceCenter
  | CubeCenter

/-- Represents a set of three collinear points in a cube -/
structure CollinearTriple where
  p1 : CubePoint
  p2 : CubePoint
  p3 : CubePoint

/-- The total number of points in the cube -/
def totalPoints : Nat := 27

/-- The number of vertices in the cube -/
def numVertices : Nat := 8

/-- The number of edge midpoints in the cube -/
def numEdgeMidpoints : Nat := 12

/-- The number of face centers in the cube -/
def numFaceCenters : Nat := 6

/-- The number of cube centers (always 1) -/
def numCubeCenters : Nat := 1

/-- Function to count the number of collinear triples in the cube -/
def countCollinearTriples : List CollinearTriple → Nat :=
  List.length

/-- Theorem: The number of sets of three collinear points in the cube is 49 -/
theorem cube_collinear_triples :
  ∃ (triples : List CollinearTriple),
    countCollinearTriples triples = 49 ∧
    totalPoints = numVertices + numEdgeMidpoints + numFaceCenters + numCubeCenters :=
  sorry

end NUMINAMATH_CALUDE_cube_collinear_triples_l1049_104933


namespace NUMINAMATH_CALUDE_systematic_sampling_selection_l1049_104955

/-- Represents a student number in the range [1, 1000] -/
def StudentNumber := Fin 1000

/-- The total number of students -/
def totalStudents : ℕ := 1000

/-- The number of students to be selected -/
def sampleSize : ℕ := 200

/-- The sample interval for systematic sampling -/
def sampleInterval : ℕ := totalStudents / sampleSize

/-- Predicate to check if a student number is selected in the systematic sampling -/
def isSelected (n : StudentNumber) : Prop :=
  n.val % sampleInterval = 122 % sampleInterval

theorem systematic_sampling_selection :
  isSelected ⟨121, by norm_num⟩ → isSelected ⟨926, by norm_num⟩ := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_selection_l1049_104955


namespace NUMINAMATH_CALUDE_problem_1_l1049_104949

theorem problem_1 (x : ℝ) (h1 : x ≠ 0) (h2 : x^2 - Real.sqrt 5 * x - x - 1 = 0) :
  x^2 + 1/x^2 = 8 + 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1049_104949


namespace NUMINAMATH_CALUDE_mary_needs_30_apples_l1049_104988

/-- Calculates the number of additional apples needed for baking pies -/
def additional_apples_needed (num_pies : ℕ) (apples_per_pie : ℕ) (apples_harvested : ℕ) : ℕ :=
  max ((num_pies * apples_per_pie) - apples_harvested) 0

/-- Proves that Mary needs to buy 30 more apples -/
theorem mary_needs_30_apples : additional_apples_needed 10 8 50 = 30 := by
  sorry

end NUMINAMATH_CALUDE_mary_needs_30_apples_l1049_104988


namespace NUMINAMATH_CALUDE_jane_earnings_l1049_104925

/-- Calculates the earnings from selling eggs over a given number of weeks -/
def egg_earnings (num_chickens : ℕ) (eggs_per_chicken : ℕ) (price_per_dozen : ℚ) (num_weeks : ℕ) : ℚ :=
  let eggs_per_week := num_chickens * eggs_per_chicken
  let dozens_per_week := eggs_per_week / 12
  let earnings_per_week := dozens_per_week * price_per_dozen
  earnings_per_week * num_weeks

/-- Proves that Jane's earnings from selling eggs over two weeks is $20 -/
theorem jane_earnings :
  egg_earnings 10 6 2 2 = 20 := by
  sorry

#eval egg_earnings 10 6 2 2

end NUMINAMATH_CALUDE_jane_earnings_l1049_104925


namespace NUMINAMATH_CALUDE_eliminate_denominator_l1049_104944

theorem eliminate_denominator (x : ℝ) : 
  (x + 1) / 3 - 3 = 2 * x + 7 → (x + 1) - 9 = 3 * (2 * x + 7) := by
  sorry

end NUMINAMATH_CALUDE_eliminate_denominator_l1049_104944


namespace NUMINAMATH_CALUDE_inequality_proof_l1049_104911

theorem inequality_proof (x y z : ℝ) : 
  x^2 / (x^2 + 2*y*z) + y^2 / (y^2 + 2*z*x) + z^2 / (z^2 + 2*x*y) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1049_104911


namespace NUMINAMATH_CALUDE_parallel_line_equation_l1049_104963

/-- Given a point and a line, find the equation of a parallel line passing through the point. -/
theorem parallel_line_equation (x₀ y₀ : ℝ) (a b c : ℝ) (h : a ≠ 0 ∨ b ≠ 0) :
  ∃ k : ℝ, ∀ x y : ℝ, (x = x₀ ∧ y = y₀) ∨ (a * x + b * y + k = 0) ↔ 
  (x = -3 ∧ y = -1) ∨ (x - 3 * y = 0) :=
sorry

end NUMINAMATH_CALUDE_parallel_line_equation_l1049_104963


namespace NUMINAMATH_CALUDE_smallest_common_factor_l1049_104981

theorem smallest_common_factor (n : ℕ) : 
  (∃ k : ℕ, k > 1 ∧ k ∣ (9*n - 2) ∧ k ∣ (7*n + 3)) → n ≥ 23 :=
sorry

end NUMINAMATH_CALUDE_smallest_common_factor_l1049_104981


namespace NUMINAMATH_CALUDE_democrat_ratio_is_one_third_l1049_104957

/-- Represents the number of participants in each category -/
structure Participants where
  total : ℕ
  female : ℕ
  male : ℕ
  femaleDemocrats : ℕ
  maleDemocrats : ℕ

/-- The ratio of democrats to total participants -/
def democratRatio (p : Participants) : ℚ :=
  (p.femaleDemocrats + p.maleDemocrats : ℚ) / p.total

theorem democrat_ratio_is_one_third (p : Participants) 
  (h1 : p.total = 660)
  (h2 : p.female + p.male = p.total)
  (h3 : p.femaleDemocrats = p.female / 2)
  (h4 : p.maleDemocrats = p.male / 4)
  (h5 : p.femaleDemocrats = 110) :
  democratRatio p = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_democrat_ratio_is_one_third_l1049_104957


namespace NUMINAMATH_CALUDE_colin_running_time_l1049_104952

theorem colin_running_time (total_miles : ℕ) (first_mile_time fourth_mile_time average_time : ℝ) :
  total_miles = 4 ∧
  first_mile_time = 6 ∧
  fourth_mile_time = 4 ∧
  average_time = 5 →
  ∃ middle_miles_time : ℝ,
    middle_miles_time = 5 ∧
    (first_mile_time + 2 * middle_miles_time + fourth_mile_time) / total_miles = average_time :=
by sorry

end NUMINAMATH_CALUDE_colin_running_time_l1049_104952


namespace NUMINAMATH_CALUDE_binomial_prob_three_l1049_104974

/-- A random variable following a binomial distribution B(n, p) -/
structure BinomialRV (n : ℕ) (p : ℝ) where
  prob : ℝ → ℝ

/-- The probability mass function of a binomial distribution -/
def binomial_pmf (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem binomial_prob_three (ξ : BinomialRV 5 (1/3)) :
  ξ.prob 3 = 40/243 := by sorry

end NUMINAMATH_CALUDE_binomial_prob_three_l1049_104974


namespace NUMINAMATH_CALUDE_wren_population_decline_l1049_104937

theorem wren_population_decline (n : ℕ) : (∀ k : ℕ, k < n → (0.7 : ℝ) ^ k ≥ 0.1) ∧ (0.7 : ℝ) ^ n < 0.1 → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_wren_population_decline_l1049_104937


namespace NUMINAMATH_CALUDE_students_playing_both_sports_l1049_104928

def total_students : ℕ := 460
def football_players : ℕ := 325
def cricket_players : ℕ := 175
def neither_players : ℕ := 50

theorem students_playing_both_sports : 
  total_students - neither_players = football_players + cricket_players - 90 := by
sorry

end NUMINAMATH_CALUDE_students_playing_both_sports_l1049_104928


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1049_104998

theorem polynomial_simplification (x : ℝ) :
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 = -x^2 + 23*x - 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1049_104998


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_two_ninths_l1049_104986

theorem sum_of_fractions_equals_two_ninths :
  (1 / (3 * 4 : ℚ)) + (1 / (4 * 5 : ℚ)) + (1 / (5 * 6 : ℚ)) +
  (1 / (6 * 7 : ℚ)) + (1 / (7 * 8 : ℚ)) + (1 / (8 * 9 : ℚ)) = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_two_ninths_l1049_104986


namespace NUMINAMATH_CALUDE_sacks_per_section_l1049_104913

/-- Given an orchard with 8 sections that produces 360 sacks of apples daily,
    prove that each section produces 45 sacks per day. -/
theorem sacks_per_section (sections : ℕ) (total_sacks : ℕ) (h1 : sections = 8) (h2 : total_sacks = 360) :
  total_sacks / sections = 45 := by
  sorry

end NUMINAMATH_CALUDE_sacks_per_section_l1049_104913


namespace NUMINAMATH_CALUDE_proportion_solution_l1049_104903

theorem proportion_solution : ∃ x : ℚ, (1 : ℚ) / 3 = (5 : ℚ) / (3 * x) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l1049_104903


namespace NUMINAMATH_CALUDE_naoh_combined_is_54_l1049_104905

/-- Represents the balanced chemical equation coefficients -/
structure BalancedEquation :=
  (naoh_coeff : ℕ)
  (h2so4_coeff : ℕ)
  (h2o_coeff : ℕ)

/-- Represents the given information about the reaction -/
structure ReactionInfo :=
  (h2so4_available : ℕ)
  (h2o_formed : ℕ)
  (equation : BalancedEquation)

/-- Calculates the number of moles of NaOH combined in the reaction -/
def naoh_combined (info : ReactionInfo) : ℕ :=
  info.h2o_formed * info.equation.naoh_coeff / info.equation.h2o_coeff

/-- Theorem stating that given the reaction information, 54 moles of NaOH were combined -/
theorem naoh_combined_is_54 (info : ReactionInfo) 
  (h_h2so4 : info.h2so4_available = 3)
  (h_h2o : info.h2o_formed = 54)
  (h_eq : info.equation = {naoh_coeff := 2, h2so4_coeff := 1, h2o_coeff := 2}) :
  naoh_combined info = 54 := by
  sorry

end NUMINAMATH_CALUDE_naoh_combined_is_54_l1049_104905


namespace NUMINAMATH_CALUDE_rate_of_change_kinetic_energy_l1049_104927

/-- The rate of change of kinetic energy for a system with increasing mass -/
theorem rate_of_change_kinetic_energy
  (M : ℝ)  -- Initial mass of the system
  (v : ℝ)  -- Constant velocity of the system
  (ρ : ℝ)  -- Rate of mass increase
  (h1 : M > 0)  -- Mass is positive
  (h2 : v ≠ 0)  -- Velocity is non-zero
  (h3 : ρ > 0)  -- Rate of mass increase is positive
  : 
  ∃ (K : ℝ → ℝ), -- Kinetic energy as a function of time
    (∀ t, K t = (1/2) * (M + ρ * t) * v^2) ∧ 
    (∀ t, deriv K t = (1/2) * ρ * v^2) :=
sorry

end NUMINAMATH_CALUDE_rate_of_change_kinetic_energy_l1049_104927


namespace NUMINAMATH_CALUDE_last_two_digits_of_sum_l1049_104987

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a number encoded as ZARAZA -/
structure Zaraza where
  z : Digit
  a : Digit
  r : Digit
  ne_za : z ≠ a
  ne_zr : z ≠ r
  ne_ar : a ≠ r

/-- Represents a number encoded as ALMAZ -/
structure Almaz where
  a : Digit
  l : Digit
  m : Digit
  z : Digit
  ne_al : a ≠ l
  ne_am : a ≠ m
  ne_az : a ≠ z
  ne_lm : l ≠ m
  ne_lz : l ≠ z
  ne_mz : m ≠ z

/-- Convert Zaraza to a natural number -/
def zarazaToNat (x : Zaraza) : ℕ :=
  x.z.val * 100000 + x.a.val * 10000 + x.r.val * 1000 + x.a.val * 100 + x.z.val * 10 + x.a.val

/-- Convert Almaz to a natural number -/
def almazToNat (x : Almaz) : ℕ :=
  x.a.val * 10000 + x.l.val * 1000 + x.m.val * 100 + x.a.val * 10 + x.z.val

/-- The main theorem -/
theorem last_two_digits_of_sum (zar : Zaraza) (alm : Almaz) 
    (h1 : zarazaToNat zar % 4 = 0)
    (h2 : almazToNat alm % 28 = 0)
    (h3 : zar.z = alm.z ∧ zar.a = alm.a) :
    (zarazaToNat zar + almazToNat alm) % 100 = 32 := by
  sorry


end NUMINAMATH_CALUDE_last_two_digits_of_sum_l1049_104987


namespace NUMINAMATH_CALUDE_linglings_spending_l1049_104906

theorem linglings_spending (x : ℝ) 
  (h1 : 720 = (1 - 1/3) * ((1 - 2/5) * x + 240)) : 
  ∃ (spent : ℝ), spent = 2/5 * x ∧ 
  720 = (1 - 1/3) * ((x - spent) + 240) := by
  sorry

end NUMINAMATH_CALUDE_linglings_spending_l1049_104906


namespace NUMINAMATH_CALUDE_meeting_time_on_circular_track_l1049_104930

/-- The time taken for two people to meet on a circular track -/
theorem meeting_time_on_circular_track 
  (track_circumference : ℝ)
  (speed1 : ℝ)
  (speed2 : ℝ)
  (h1 : track_circumference = 528)
  (h2 : speed1 = 4.5)
  (h3 : speed2 = 3.75) :
  (track_circumference / ((speed1 + speed2) * 1000 / 60)) = 3.84 := by
  sorry

end NUMINAMATH_CALUDE_meeting_time_on_circular_track_l1049_104930


namespace NUMINAMATH_CALUDE_talent_show_girls_l1049_104982

theorem talent_show_girls (total : ℕ) (difference : ℕ) (girls : ℕ) : 
  total = 34 → difference = 22 → girls = total - (total - difference) / 2 → girls = 28 := by
sorry

end NUMINAMATH_CALUDE_talent_show_girls_l1049_104982


namespace NUMINAMATH_CALUDE_simplify_expression_l1049_104960

theorem simplify_expression : 2 - (2 / (2 + 2 * Real.sqrt 2)) + (2 / (2 - 2 * Real.sqrt 2)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1049_104960


namespace NUMINAMATH_CALUDE_world_record_rates_l1049_104948

-- Define the world records
def hotdog_record : ℕ := 75
def hotdog_time : ℕ := 10
def hamburger_record : ℕ := 97
def hamburger_time : ℕ := 3
def cheesecake_record : ℚ := 11
def cheesecake_time : ℕ := 9

-- Define Lisa's progress
def lisa_hotdogs : ℕ := 20
def lisa_hotdog_time : ℕ := 5
def lisa_hamburgers : ℕ := 60
def lisa_hamburger_time : ℕ := 2
def lisa_cheesecake : ℚ := 5
def lisa_cheesecake_time : ℕ := 5

-- Define the theorem
theorem world_record_rates : 
  (((hotdog_record - lisa_hotdogs : ℚ) / (hotdog_time - lisa_hotdog_time)) = 11) ∧
  (((hamburger_record - lisa_hamburgers : ℚ) / (hamburger_time - lisa_hamburger_time)) = 37) ∧
  (((cheesecake_record - lisa_cheesecake) / (cheesecake_time - lisa_cheesecake_time)) = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_world_record_rates_l1049_104948


namespace NUMINAMATH_CALUDE_johns_payment_ratio_l1049_104914

/-- Proves that the ratio of John's payment to the total cost for the first year is 1/2 --/
theorem johns_payment_ratio (
  num_members : ℕ)
  (join_fee : ℕ)
  (monthly_cost : ℕ)
  (johns_payment : ℕ)
  (h1 : num_members = 4)
  (h2 : join_fee = 4000)
  (h3 : monthly_cost = 1000)
  (h4 : johns_payment = 32000)
  : johns_payment / (num_members * (join_fee + 12 * monthly_cost)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_johns_payment_ratio_l1049_104914


namespace NUMINAMATH_CALUDE_converse_proposition_l1049_104972

theorem converse_proposition : 
  (∀ x : ℝ, x = 3 → x^2 - 2*x - 3 = 0) ↔ 
  (∀ x : ℝ, x^2 - 2*x - 3 = 0 → x = 3) :=
by sorry

end NUMINAMATH_CALUDE_converse_proposition_l1049_104972


namespace NUMINAMATH_CALUDE_mary_balloons_l1049_104936

-- Define the number of Nancy's balloons
def nancy_balloons : ℕ := 7

-- Define the ratio of Mary's balloons to Nancy's
def mary_ratio : ℕ := 4

-- Theorem to prove
theorem mary_balloons : nancy_balloons * mary_ratio = 28 := by
  sorry

end NUMINAMATH_CALUDE_mary_balloons_l1049_104936


namespace NUMINAMATH_CALUDE_division_remainder_and_primality_l1049_104943

theorem division_remainder_and_primality : 
  let dividend := 5432109
  let divisor := 125
  let remainder := dividend % divisor
  (remainder = 84) ∧ ¬(Nat.Prime remainder) := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_and_primality_l1049_104943


namespace NUMINAMATH_CALUDE_arithmetic_sequence_specific_sum_l1049_104992

/-- An arithmetic sequence with sum S_n of its first n terms. -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0
  sum_formula : ∀ n : ℕ, S n = n * (a 0 + a (n - 1)) / 2

/-- Given an arithmetic sequence with specific sum values, prove n = 4. -/
theorem arithmetic_sequence_specific_sum (seq : ArithmeticSequence) 
  (h1 : seq.S 6 = 36)
  (h2 : seq.S 12 = 144)
  (h3 : ∃ n : ℕ, seq.S (6 * n) = 576) :
  ∃ n : ℕ, n = 4 ∧ seq.S (6 * n) = 576 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_specific_sum_l1049_104992


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l1049_104939

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 250 →
  train_speed_kmh = 60 →
  crossing_time = 20 →
  ∃ (bridge_length : ℝ), (abs (bridge_length - 83.4) < 0.1) ∧
    (bridge_length = train_speed_kmh * 1000 / 3600 * crossing_time - train_length) :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l1049_104939


namespace NUMINAMATH_CALUDE_parallel_line_existence_l1049_104985

-- Define the necessary structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define parallelism
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a ≠ 0 ∧ l1.b ≠ 0 ∧ l2.a ≠ 0 ∧ l2.b ≠ 0

-- Define a line passing through a point
def passes_through (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Theorem statement
theorem parallel_line_existence (A : Point) (l : Line) :
  ∃ (m : Line), passes_through m A ∧ parallel m l :=
sorry

end NUMINAMATH_CALUDE_parallel_line_existence_l1049_104985


namespace NUMINAMATH_CALUDE_max_min_difference_c_l1049_104916

theorem max_min_difference_c (a b c : ℝ) 
  (sum_eq : a + b + c = 6) 
  (sum_sq_eq : a^2 + b^2 + c^2 = 24) : 
  ∃ (c_max c_min : ℝ), 
    (∀ x : ℝ, (∃ y z : ℝ, y + z + x = 6 ∧ y^2 + z^2 + x^2 = 24) → x ≤ c_max ∧ x ≥ c_min) ∧
    c_max - c_min = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_min_difference_c_l1049_104916


namespace NUMINAMATH_CALUDE_sum_x_y_equals_five_l1049_104907

theorem sum_x_y_equals_five (x y : ℝ) 
  (eq1 : x + 3*y = 12) 
  (eq2 : 3*x + y = 8) : 
  x + y = 5 := by
sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_five_l1049_104907


namespace NUMINAMATH_CALUDE_range_of_c_sum_of_squares_inequality_l1049_104929

-- Part I
theorem range_of_c (c : ℝ) (h1 : c > 0) 
  (h2 : ∀ x : ℝ, x + |x - 2*c| ≥ 2) : c ≥ 1 := by
  sorry

-- Part II
theorem sum_of_squares_inequality (p q r : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) 
  (h_sum : p + q + r = 3) : p^2 + q^2 + r^2 ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_c_sum_of_squares_inequality_l1049_104929


namespace NUMINAMATH_CALUDE_sin_sum_to_product_l1049_104995

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (3 * x) + Real.sin (7 * x) = 2 * Real.sin (5 * x) * Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_to_product_l1049_104995


namespace NUMINAMATH_CALUDE_chef_nut_purchase_l1049_104908

/-- The weight of almonds bought by the chef in kilograms -/
def almond_weight : ℝ := 0.14

/-- The weight of pecans bought by the chef in kilograms -/
def pecan_weight : ℝ := 0.38

/-- The total weight of nuts bought by the chef in kilograms -/
def total_nut_weight : ℝ := almond_weight + pecan_weight

theorem chef_nut_purchase : total_nut_weight = 0.52 := by
  sorry

end NUMINAMATH_CALUDE_chef_nut_purchase_l1049_104908


namespace NUMINAMATH_CALUDE_sum_sequence_square_l1049_104997

theorem sum_sequence_square (n : ℕ) : 
  (List.range n).sum + n + (List.range n).reverse.sum = n^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_sequence_square_l1049_104997


namespace NUMINAMATH_CALUDE_pagoda_lights_l1049_104966

theorem pagoda_lights (n : ℕ) (total : ℕ) (h1 : n = 7) (h2 : total = 381) :
  ∃ (a : ℕ), 
    a * (1 - (1/2)^n) / (1 - 1/2) = total ∧ 
    a * (1/2)^(n-1) = 3 :=
sorry

end NUMINAMATH_CALUDE_pagoda_lights_l1049_104966


namespace NUMINAMATH_CALUDE_remainder_17_63_mod_7_l1049_104920

theorem remainder_17_63_mod_7 : 17^63 % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_17_63_mod_7_l1049_104920


namespace NUMINAMATH_CALUDE_coefficient_x_term_expansion_l1049_104921

theorem coefficient_x_term_expansion (x : ℝ) : 
  (∃ a b c d e : ℝ, (1 + x) * (2 - x)^4 = a*x^4 + b*x^3 + c*x^2 + d*x + e) → 
  (∃ a b c d e : ℝ, (1 + x) * (2 - x)^4 = a*x^4 + b*x^3 + c*x^2 + (-16)*x + e) :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x_term_expansion_l1049_104921


namespace NUMINAMATH_CALUDE_united_additional_charge_is_correct_l1049_104904

/-- The additional charge per minute for United Telephone -/
def united_additional_charge : ℚ := 1/4

/-- United Telephone's base rate -/
def united_base_rate : ℚ := 6

/-- Atlantic Call's base rate -/
def atlantic_base_rate : ℚ := 12

/-- Atlantic Call's additional charge per minute -/
def atlantic_additional_charge : ℚ := 1/5

/-- The number of minutes at which both companies' bills are equal -/
def equal_minutes : ℕ := 120

theorem united_additional_charge_is_correct : 
  united_base_rate + equal_minutes * united_additional_charge = 
  atlantic_base_rate + equal_minutes * atlantic_additional_charge :=
by sorry

end NUMINAMATH_CALUDE_united_additional_charge_is_correct_l1049_104904


namespace NUMINAMATH_CALUDE_intersection_on_unit_circle_l1049_104932

theorem intersection_on_unit_circle (k₁ k₂ : ℝ) (h : k₁ * k₂ + 1 = 0) :
  ∃ (x y : ℝ), (y = k₁ * x + 1) ∧ (y = k₂ * x - 1) ∧ (x^2 + y^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_intersection_on_unit_circle_l1049_104932


namespace NUMINAMATH_CALUDE_shared_focus_parabola_ellipse_l1049_104919

/-- Given a parabola and an ellipse that share a focus, prove the value of m in the ellipse equation -/
theorem shared_focus_parabola_ellipse (x y : ℝ) (m : ℝ) : 
  (x^2 = 2*y) →  -- Parabola equation
  (y^2/m + x^2/2 = 1) →  -- Ellipse equation
  (∃ f : ℝ × ℝ, f ∈ {p : ℝ × ℝ | p.1^2 = 2*p.2} ∩ {e : ℝ × ℝ | e.2^2/m + e.1^2/2 = 1}) →  -- Shared focus
  (m = 9/4) :=
by sorry

end NUMINAMATH_CALUDE_shared_focus_parabola_ellipse_l1049_104919


namespace NUMINAMATH_CALUDE_c_value_theorem_l1049_104991

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation for c
def c_equation (a b : ℕ+) : ℂ := (a + b * i) ^ 3 - 107 * i

-- State the theorem
theorem c_value_theorem (a b c : ℕ+) :
  (c_equation a b).re = c ∧ (c_equation a b).im = 0 → c = 198 := by
  sorry

end NUMINAMATH_CALUDE_c_value_theorem_l1049_104991


namespace NUMINAMATH_CALUDE_max_negative_integers_l1049_104954

theorem max_negative_integers (a b c d e f : ℤ) (h : a * b + c * d * e * f < 0) :
  (∃ neg_count : ℕ, neg_count ≤ 3 ∧
    neg_count = (if a < 0 then 1 else 0) + (if b < 0 then 1 else 0) +
                (if c < 0 then 1 else 0) + (if d < 0 then 1 else 0) +
                (if e < 0 then 1 else 0) + (if f < 0 then 1 else 0)) ∧
  ¬(∃ neg_count : ℕ, neg_count > 3 ∧
    neg_count = (if a < 0 then 1 else 0) + (if b < 0 then 1 else 0) +
                (if c < 0 then 1 else 0) + (if d < 0 then 1 else 0) +
                (if e < 0 then 1 else 0) + (if f < 0 then 1 else 0)) :=
by sorry

end NUMINAMATH_CALUDE_max_negative_integers_l1049_104954
