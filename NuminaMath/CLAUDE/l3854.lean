import Mathlib

namespace NUMINAMATH_CALUDE_michelle_crayons_l3854_385450

/-- The number of crayons Michelle has -/
def total_crayons (crayons_per_box : ℝ) (num_boxes : ℝ) : ℝ :=
  crayons_per_box * num_boxes

/-- Proof that Michelle has 7.0 crayons -/
theorem michelle_crayons :
  total_crayons 5.0 1.4 = 7.0 := by
  sorry

end NUMINAMATH_CALUDE_michelle_crayons_l3854_385450


namespace NUMINAMATH_CALUDE_total_students_is_240_l3854_385467

/-- The number of students from Know It All High School -/
def know_it_all_students : ℕ := 50

/-- The number of students from Karen High School -/
def karen_students : ℕ := (3 * know_it_all_students) / 5

/-- The combined number of students from Know It All High School and Karen High School -/
def combined_students : ℕ := know_it_all_students + karen_students

/-- The number of students from Novel Corona High School -/
def novel_corona_students : ℕ := 2 * combined_students

/-- The total number of students at the competition -/
def total_students : ℕ := combined_students + novel_corona_students

/-- Theorem stating that the total number of students at the competition is 240 -/
theorem total_students_is_240 : total_students = 240 := by
  sorry

end NUMINAMATH_CALUDE_total_students_is_240_l3854_385467


namespace NUMINAMATH_CALUDE_no_negative_roots_and_positive_root_exists_l3854_385471

def f (x : ℝ) : ℝ := x^6 - 3*x^5 - 6*x^3 - x + 8

theorem no_negative_roots_and_positive_root_exists :
  (∀ x < 0, f x ≠ 0) ∧ (∃ x > 0, f x = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_negative_roots_and_positive_root_exists_l3854_385471


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l3854_385417

/-- Given a parabola y = ax^2 - a (a ≠ 0) intersecting a line y = kx at points 
    with x-coordinates summing to less than 0, prove that the line y = ax + k 
    passes through the first and fourth quadrants. -/
theorem parabola_line_intersection (a k : ℝ) (h_a : a ≠ 0) :
  (∃ x₁ x₂ : ℝ, a * x₁^2 - a = k * x₁ ∧ 
               a * x₂^2 - a = k * x₂ ∧ 
               x₁ + x₂ < 0) →
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ y = a * x + k) ∧
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ y = a * x + k) :=
by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l3854_385417


namespace NUMINAMATH_CALUDE_randys_final_amount_l3854_385447

/-- Calculates Randy's remaining money after a series of transactions --/
def randys_remaining_money (initial_amount : ℝ) (smith_gift : ℝ) 
  (sally_percentage : ℝ) (stock_percentage : ℝ) (crypto_percentage : ℝ) : ℝ :=
  let new_total := initial_amount + smith_gift
  let after_sally := new_total * (1 - sally_percentage)
  let after_stocks := after_sally * (1 - stock_percentage)
  after_stocks * (1 - crypto_percentage)

/-- Theorem stating that Randy's remaining money is $1,008 --/
theorem randys_final_amount :
  randys_remaining_money 3000 200 0.25 0.40 0.30 = 1008 := by
  sorry

#eval randys_remaining_money 3000 200 0.25 0.40 0.30

end NUMINAMATH_CALUDE_randys_final_amount_l3854_385447


namespace NUMINAMATH_CALUDE_root_equation_implication_l3854_385483

theorem root_equation_implication (m : ℝ) : 
  m^2 - m - 3 = 0 → m^2 - m - 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_implication_l3854_385483


namespace NUMINAMATH_CALUDE_figure_division_l3854_385448

/-- A figure consisting of 24 cells can be divided into equal parts of specific sizes. -/
theorem figure_division (n : ℕ) : n ∣ 24 ∧ n ≠ 1 ↔ n ∈ ({2, 3, 4, 6, 8, 12, 24} : Finset ℕ) :=
sorry

end NUMINAMATH_CALUDE_figure_division_l3854_385448


namespace NUMINAMATH_CALUDE_james_carrot_sticks_l3854_385412

theorem james_carrot_sticks (before after total : ℕ) : 
  before = 22 → after = 15 → total = before + after → total = 37 := by sorry

end NUMINAMATH_CALUDE_james_carrot_sticks_l3854_385412


namespace NUMINAMATH_CALUDE_equilateral_triangle_expression_bound_l3854_385456

theorem equilateral_triangle_expression_bound (a : ℝ) (h : a > 0) : (3 * a^2) / (3 * a) > 0 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_expression_bound_l3854_385456


namespace NUMINAMATH_CALUDE_not_divisible_by_11599_l3854_385466

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def N : ℚ := (factorial 3400) / ((factorial 1700) ^ 2)

theorem not_divisible_by_11599 : ¬ (∃ (k : ℤ), N = k * (11599 : ℚ)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_11599_l3854_385466


namespace NUMINAMATH_CALUDE_polynomial_roots_product_l3854_385415

theorem polynomial_roots_product (b c : ℤ) : 
  (∀ r : ℝ, r^2 - r - 2 = 0 → r^5 - b*r - c = 0) → b*c = 110 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_product_l3854_385415


namespace NUMINAMATH_CALUDE_reverse_digit_integers_l3854_385490

theorem reverse_digit_integers (q r : ℕ) : 
  (10 ≤ q ∧ q < 100) →  -- q is a two-digit integer
  (10 ≤ r ∧ r < 100) →  -- r is a two-digit integer
  (∃ a b : ℕ, q = 10 * a + b ∧ r = 10 * b + a) →  -- q and r have reversed digits
  (q > r → q - r < 60) →  -- positive difference less than 60
  (r > q → r - q < 60) →  -- positive difference less than 60
  (∀ x y : ℕ, (10 ≤ x ∧ x < 100) → (10 ≤ y ∧ y < 100) → 
    (∃ c d : ℕ, x = 10 * c + d ∧ y = 10 * d + c) → 
    (x > y → x - y ≤ 54) ∧ (y > x → y - x ≤ 54)) →  -- greatest possible difference is 54
  (∃ a b : ℕ, q = 10 * a + b ∧ r = 10 * b + a ∧ a = b + 6) :=  -- conclusion: tens digit is 6 more than units digit
by sorry

end NUMINAMATH_CALUDE_reverse_digit_integers_l3854_385490


namespace NUMINAMATH_CALUDE_power_equation_solution_l3854_385473

theorem power_equation_solution (m : ℝ) : 2^m = (64 : ℝ)^(1/3) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l3854_385473


namespace NUMINAMATH_CALUDE_coin_difference_l3854_385429

def coin_values : List Nat := [5, 15, 20]

def target_amount : Nat := 50

def min_coins (values : List Nat) (target : Nat) : Nat :=
  sorry

def max_coins (values : List Nat) (target : Nat) : Nat :=
  sorry

theorem coin_difference :
  max_coins coin_values target_amount - min_coins coin_values target_amount = 6 :=
by sorry

end NUMINAMATH_CALUDE_coin_difference_l3854_385429


namespace NUMINAMATH_CALUDE_divisibility_of_A_l3854_385444

def A : ℕ := 2013 * (10^(4*165) - 1) / (10^4 - 1)

theorem divisibility_of_A : 2013^2 ∣ A := by sorry

end NUMINAMATH_CALUDE_divisibility_of_A_l3854_385444


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_equals_one_l3854_385489

theorem sum_of_reciprocals_equals_one (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y) : 
  1/x + 1/y = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_equals_one_l3854_385489


namespace NUMINAMATH_CALUDE_gizmo_production_l3854_385476

/-- Represents the production rate of gadgets per worker per hour -/
def gadget_rate : ℝ := 2

/-- Represents the production rate of gizmos per worker per hour -/
def gizmo_rate : ℝ := 1.5

/-- Represents the number of workers -/
def workers : ℕ := 40

/-- Represents the total working hours -/
def total_hours : ℝ := 6

/-- Represents the number of gadgets to be produced -/
def gadgets_to_produce : ℕ := 240

theorem gizmo_production :
  let hours_for_gadgets : ℝ := gadgets_to_produce / (workers * gadget_rate)
  let remaining_hours : ℝ := total_hours - hours_for_gadgets
  ↑workers * gizmo_rate * remaining_hours = 180 :=
sorry

end NUMINAMATH_CALUDE_gizmo_production_l3854_385476


namespace NUMINAMATH_CALUDE_expand_product_l3854_385445

theorem expand_product (x : ℝ) : (x^2 - 3*x + 3) * (x^2 + 3*x + 3) = x^4 - 3*x^2 + 9 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3854_385445


namespace NUMINAMATH_CALUDE_horner_method_evaluation_l3854_385487

/-- Horner's Method for polynomial evaluation -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => a + x * acc) 0

/-- The polynomial function -/
def f (x : ℝ) : ℝ :=
  1 + x + 0.5 * x^2 + 0.16667 * x^3 + 0.04167 * x^4 + 0.00833 * x^5

theorem horner_method_evaluation :
  let coeffs := [0.00833, 0.04167, 0.16667, 0.5, 1, 1]
  abs (horner_eval coeffs (-0.2) - f (-0.2)) < 1e-5 ∧
  abs (horner_eval coeffs (-0.2) - 0.00427) < 1e-5 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_evaluation_l3854_385487


namespace NUMINAMATH_CALUDE_sum_of_digits_of_greatest_prime_divisor_8191_l3854_385409

def greatest_prime_divisor (n : Nat) : Nat :=
  sorry

def sum_of_digits (n : Nat) : Nat :=
  sorry

theorem sum_of_digits_of_greatest_prime_divisor_8191 :
  sum_of_digits (greatest_prime_divisor 8191) = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_greatest_prime_divisor_8191_l3854_385409


namespace NUMINAMATH_CALUDE_eulerian_path_figures_l3854_385433

-- Define a structure for our figures
structure Figure where
  has_eulerian_path : Bool
  all_vertices_even_degree : Bool
  num_odd_degree_vertices : Nat

-- Define our theorem
theorem eulerian_path_figures :
  let figureA : Figure := { has_eulerian_path := true, all_vertices_even_degree := true, num_odd_degree_vertices := 0 }
  let figureB : Figure := { has_eulerian_path := true, all_vertices_even_degree := false, num_odd_degree_vertices := 0 }
  let figureC : Figure := { has_eulerian_path := false, all_vertices_even_degree := false, num_odd_degree_vertices := 3 }
  let figureD : Figure := { has_eulerian_path := true, all_vertices_even_degree := true, num_odd_degree_vertices := 0 }
  ∀ (f : Figure),
    (f.all_vertices_even_degree ∨ f.num_odd_degree_vertices = 2) ↔ f.has_eulerian_path :=
by
  sorry


end NUMINAMATH_CALUDE_eulerian_path_figures_l3854_385433


namespace NUMINAMATH_CALUDE_cone_height_l3854_385430

/-- The height of a cone given its slant height and lateral area -/
theorem cone_height (l : ℝ) (area : ℝ) (h : l = 13 ∧ area = 65 * Real.pi) :
  ∃ (r : ℝ), r > 0 ∧ area = Real.pi * r * l ∧ Real.sqrt (l^2 - r^2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_cone_height_l3854_385430


namespace NUMINAMATH_CALUDE_real_roots_condition_l3854_385455

theorem real_roots_condition (k : ℝ) : 
  (∃ x : ℝ, (k - 1) * x^2 + 2 * k * x + (k - 3) = 0) ↔ k ≥ 3/4 := by
sorry

end NUMINAMATH_CALUDE_real_roots_condition_l3854_385455


namespace NUMINAMATH_CALUDE_largest_n_divisibility_n_890_divisibility_n_890_largest_l3854_385410

theorem largest_n_divisibility : ∀ n : ℕ, n > 890 → ¬(n + 10 ∣ n^3 + 100) :=
by
  sorry

theorem n_890_divisibility : (890 + 10 ∣ 890^3 + 100) :=
by
  sorry

theorem n_890_largest : ∀ n : ℕ, n > 0 → (n + 10 ∣ n^3 + 100) → n ≤ 890 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_n_divisibility_n_890_divisibility_n_890_largest_l3854_385410


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3854_385482

theorem inequality_solution_set (x : ℝ) : -x^2 + 2*x + 3 ≥ 0 ↔ x ∈ Set.Icc (-1) 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3854_385482


namespace NUMINAMATH_CALUDE_root_sum_equals_three_l3854_385474

noncomputable section

-- Define the logarithm base 10 function
def log10 (x : ℝ) := Real.log x / Real.log 10

-- Define the equations for x₁ and x₂
def equation1 (x : ℝ) : Prop := x + log10 x = 3
def equation2 (x : ℝ) : Prop := x + 10^x = 3

-- State the theorem
theorem root_sum_equals_three 
  (x₁ x₂ : ℝ) 
  (h1 : equation1 x₁) 
  (h2 : equation2 x₂) : 
  x₁ + x₂ = 3 := by sorry

end

end NUMINAMATH_CALUDE_root_sum_equals_three_l3854_385474


namespace NUMINAMATH_CALUDE_two_color_distance_l3854_385425

/-- A type representing colors --/
inductive Color
| Red
| Blue

/-- A two-coloring of the plane --/
def Coloring := ℝ × ℝ → Color

/-- Predicate to check if both colors are used in a coloring --/
def BothColorsUsed (c : Coloring) : Prop :=
  (∃ p : ℝ × ℝ, c p = Color.Red) ∧ (∃ p : ℝ × ℝ, c p = Color.Blue)

/-- The main theorem --/
theorem two_color_distance (c : Coloring) (h : BothColorsUsed c) (a : ℝ) (ha : a > 0) :
  ∃ p q : ℝ × ℝ, c p ≠ c q ∧ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = a :=
sorry

end NUMINAMATH_CALUDE_two_color_distance_l3854_385425


namespace NUMINAMATH_CALUDE_constant_function_no_monotonicity_l3854_385407

open Function Set

theorem constant_function_no_monotonicity 
  {f : ℝ → ℝ} {I : Set ℝ} (hI : Interval I) :
  (∀ x ∈ I, HasDerivAt f (0 : ℝ) x) → 
  ∃ c, ∀ x ∈ I, f x = c :=
sorry

end NUMINAMATH_CALUDE_constant_function_no_monotonicity_l3854_385407


namespace NUMINAMATH_CALUDE_problem_solution_l3854_385405

theorem problem_solution : (2013^2 - 2013 - 1) / 2013 = 2012 - 1/2013 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3854_385405


namespace NUMINAMATH_CALUDE_complement_of_N_l3854_385431

-- Define the universal set M
def M : Set Nat := {1, 2, 3, 4, 5}

-- Define the set N
def N : Set Nat := {2, 4}

-- State the theorem
theorem complement_of_N : (M \ N) = {1, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_N_l3854_385431


namespace NUMINAMATH_CALUDE_smallest_term_at_four_l3854_385478

def a (n : ℕ+) : ℚ := (1 / 3) * n^3 - 13 * n

theorem smallest_term_at_four :
  ∀ k : ℕ+, a 4 ≤ a k := by sorry

end NUMINAMATH_CALUDE_smallest_term_at_four_l3854_385478


namespace NUMINAMATH_CALUDE_profit_distribution_l3854_385402

theorem profit_distribution (total_profit : ℝ) (num_employees : ℕ) (employee_share : ℝ) :
  total_profit = 50 →
  num_employees = 9 →
  employee_share = 5 →
  (total_profit - num_employees * employee_share) / total_profit * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_profit_distribution_l3854_385402


namespace NUMINAMATH_CALUDE_nested_G_evaluation_l3854_385497

def G (x : ℝ) : ℝ := (x - 2)^2 - 1

theorem nested_G_evaluation : G (G (G (G (G 2)))) = 1179395 := by
  sorry

end NUMINAMATH_CALUDE_nested_G_evaluation_l3854_385497


namespace NUMINAMATH_CALUDE_pebble_ratio_l3854_385438

def total_pebbles : ℕ := 30
def white_pebbles : ℕ := 20

def red_pebbles : ℕ := total_pebbles - white_pebbles

theorem pebble_ratio : 
  (red_pebbles : ℚ) / white_pebbles = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_pebble_ratio_l3854_385438


namespace NUMINAMATH_CALUDE_system_solution_l3854_385479

theorem system_solution (x y z u v : ℝ) : 
  (x + y + z + u = 5) ∧
  (y + z + u + v = 1) ∧
  (z + u + v + x = 2) ∧
  (u + v + x + y = 0) ∧
  (v + x + y + z = 4) →
  (v = -2 ∧ x = 2 ∧ y = 1 ∧ z = 3 ∧ u = -1) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3854_385479


namespace NUMINAMATH_CALUDE_set_operations_correctness_l3854_385424

variable {α : Type*}
variable (A B C : Set α)

theorem set_operations_correctness :
  (A ∪ B = B ∪ A) ∧
  (A ∪ (B ∪ C) = (A ∪ B) ∪ C) ∧
  (A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C)) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_correctness_l3854_385424


namespace NUMINAMATH_CALUDE_egg_production_increase_l3854_385427

theorem egg_production_increase (last_year_production this_year_production : ℕ) 
  (h1 : last_year_production = 1416)
  (h2 : this_year_production = 4636) :
  this_year_production - last_year_production = 3220 := by
  sorry

end NUMINAMATH_CALUDE_egg_production_increase_l3854_385427


namespace NUMINAMATH_CALUDE_bernoulli_misplacement_problem_l3854_385437

def D : ℕ → ℕ
  | 0 => 0
  | 1 => 0
  | 2 => 1
  | (n + 1) => n * (D n + D (n - 1))

theorem bernoulli_misplacement_problem :
  (D 4 : ℚ) / 24 = 3 / 8 ∧
  (6 * D 5 : ℚ) / 720 = 11 / 30 := by
  sorry


end NUMINAMATH_CALUDE_bernoulli_misplacement_problem_l3854_385437


namespace NUMINAMATH_CALUDE_tan_half_sum_l3854_385469

theorem tan_half_sum (a b : Real) 
  (h1 : Real.cos a + Real.cos b = 3/5)
  (h2 : Real.sin a + Real.sin b = 1/5) :
  Real.tan ((a + b) / 2) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_tan_half_sum_l3854_385469


namespace NUMINAMATH_CALUDE_triangle_cut_range_l3854_385461

/-- Given a triangle with side lengths 4, 5, and 6,
    if x is cut off from all sides resulting in an obtuse triangle,
    then 1 < x < 3 -/
theorem triangle_cut_range (x : ℝ) : 
  let a := 4 - x
  let b := 5 - x
  let c := 6 - x
  (0 < a ∧ 0 < b ∧ 0 < c) →
  (a + b > c ∧ b + c > a ∧ c + a > b) →
  (a^2 + b^2 - c^2) / (2 * a * b) < 0 →
  1 < x ∧ x < 3 :=
by sorry


end NUMINAMATH_CALUDE_triangle_cut_range_l3854_385461


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_negative_two_ninths_l3854_385496

-- Define the function f
def f (x : ℝ) : ℝ := (3*x)^2 + 2*(3*x) + 2

-- State the theorem
theorem sum_of_roots_equals_negative_two_ninths :
  ∃ (z₁ z₂ : ℝ), z₁ ≠ z₂ ∧ f z₁ = 10 ∧ f z₂ = 10 ∧ z₁ + z₂ = -2/9 :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_negative_two_ninths_l3854_385496


namespace NUMINAMATH_CALUDE_binomial_distribution_unique_parameters_l3854_385464

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expected value of a binomial distribution -/
def expectedValue (b : BinomialDistribution) : ℝ := b.n * b.p

/-- The variance of a binomial distribution -/
def variance (b : BinomialDistribution) : ℝ := b.n * b.p * (1 - b.p)

/-- Theorem: For a binomial distribution X ~ B(n, p) with E(X) = 3 and D(X) = 2,
    the values of n and p are 9 and 1/3 respectively -/
theorem binomial_distribution_unique_parameters :
  ∀ b : BinomialDistribution,
    expectedValue b = 3 →
    variance b = 2 →
    b.n = 9 ∧ b.p = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_binomial_distribution_unique_parameters_l3854_385464


namespace NUMINAMATH_CALUDE_xyz_value_l3854_385436

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12) :
  4 * x * y * z = 48 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l3854_385436


namespace NUMINAMATH_CALUDE_tom_spent_seven_tickets_on_hat_l3854_385411

/-- The number of tickets Tom spent on the hat -/
def tickets_spent_on_hat (whack_a_mole_tickets : ℕ) (skee_ball_tickets : ℕ) (remaining_tickets : ℕ) : ℕ :=
  whack_a_mole_tickets + skee_ball_tickets - remaining_tickets

/-- Proof that Tom spent 7 tickets on the hat -/
theorem tom_spent_seven_tickets_on_hat :
  tickets_spent_on_hat 32 25 50 = 7 := by
  sorry

end NUMINAMATH_CALUDE_tom_spent_seven_tickets_on_hat_l3854_385411


namespace NUMINAMATH_CALUDE_smallest_integer_above_sqrt_difference_power_l3854_385481

theorem smallest_integer_above_sqrt_difference_power :
  ∃ n : ℤ, (n = 9737 ∧ ∀ m : ℤ, (m > (Real.sqrt 5 - Real.sqrt 3)^8 → m ≥ n)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_above_sqrt_difference_power_l3854_385481


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l3854_385426

/-- Given a rhombus with diagonals of length 10 and 24, its perimeter is 52. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) : 
  let side := Real.sqrt ((d1/2)^2 + (d2/2)^2)
  4 * side = 52 := by
sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l3854_385426


namespace NUMINAMATH_CALUDE_age_equation_solution_l3854_385428

/-- Given a person's age and a number of years, this function represents the equation in the problem. -/
def ageEquation (A : ℕ) (x : ℕ) : Prop :=
  3 * (A + x) - 3 * (A - x) = A

/-- The theorem states that for an age of 30, the equation is satisfied when x is 5. -/
theorem age_equation_solution :
  ageEquation 30 5 := by
  sorry

end NUMINAMATH_CALUDE_age_equation_solution_l3854_385428


namespace NUMINAMATH_CALUDE_rhombus_area_l3854_385492

-- Define the vertices of the rhombus
def v1 : ℝ × ℝ := (1.2, 4.1)
def v2 : ℝ × ℝ := (7.3, 2.5)
def v3 : ℝ × ℝ := (1.2, -2.8)
def v4 : ℝ × ℝ := (-4.9, 2.5)

-- Define the vectors representing two adjacent sides of the rhombus
def vector1 : ℝ × ℝ := (v2.1 - v1.1, v2.2 - v1.2)
def vector2 : ℝ × ℝ := (v4.1 - v1.1, v4.2 - v1.2)

-- Theorem stating that the area of the rhombus is 19.52 square units
theorem rhombus_area : 
  abs ((vector1.1 * vector2.2) - (vector1.2 * vector2.1)) = 19.52 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l3854_385492


namespace NUMINAMATH_CALUDE_marble_bag_problem_l3854_385421

/-- Given a bag of black and white marbles, if removing one black marble
    results in 1/8 of the remaining marbles being black, and removing three
    white marbles results in 1/6 of the remaining marbles being black,
    then the initial number of marbles in the bag is 9. -/
theorem marble_bag_problem (x y : ℕ) : 
  x > 0 → y > 0 →
  (x - 1 : ℚ) / (x + y - 1 : ℚ) = 1 / 8 →
  x / (x + y - 3 : ℚ) = 1 / 6 →
  x + y = 9 :=
by sorry

end NUMINAMATH_CALUDE_marble_bag_problem_l3854_385421


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l3854_385486

noncomputable def f (x : ℝ) := x * Real.exp x

theorem tangent_line_at_one :
  ∃ (m b : ℝ), ∀ (x y : ℝ),
    y = m * x + b ↔ 
    (∃ (h : ℝ → ℝ), (∀ t, t ≠ 1 → (h t - f 1) / (t - 1) = (f t - f 1) / (t - 1)) ∧
                     (h 1 = f 1) ∧
                     y = h x) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l3854_385486


namespace NUMINAMATH_CALUDE_concert_ticket_revenue_l3854_385443

/-- Calculates the total revenue from concert ticket sales --/
theorem concert_ticket_revenue :
  let full_price : ℕ := 20
  let first_group_size : ℕ := 10
  let second_group_size : ℕ := 20
  let first_discount_percent : ℕ := 40
  let second_discount_percent : ℕ := 15
  let total_attendees : ℕ := 56

  let first_group_revenue := first_group_size * (full_price * (100 - first_discount_percent) / 100)
  let second_group_revenue := second_group_size * (full_price * (100 - second_discount_percent) / 100)
  let remaining_attendees := total_attendees - first_group_size - second_group_size
  let remaining_revenue := remaining_attendees * full_price

  let total_revenue := first_group_revenue + second_group_revenue + remaining_revenue

  total_revenue = 980 := by
    sorry

end NUMINAMATH_CALUDE_concert_ticket_revenue_l3854_385443


namespace NUMINAMATH_CALUDE_equation_solutions_l3854_385403

theorem equation_solutions :
  (∀ x : ℝ, 5 * x + 2 = 3 * x - 4 ↔ x = -3) ∧
  (∀ x : ℝ, 1.2 * (x + 4) = 3.6 * (x - 14) ↔ x = 23) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3854_385403


namespace NUMINAMATH_CALUDE_hyperbola_t_range_l3854_385465

-- Define the curve C
def curve_C (t : ℝ) := {(x, y) : ℝ × ℝ | x^2 / (4 - t) + y^2 / (t - 1) = 1}

-- Define what it means for a curve to be a hyperbola
def is_hyperbola (C : Set (ℝ × ℝ)) : Prop := sorry

-- State the theorem
theorem hyperbola_t_range (t : ℝ) :
  is_hyperbola (curve_C t) → t < 1 ∨ t > 4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_t_range_l3854_385465


namespace NUMINAMATH_CALUDE_original_recipe_flour_amount_l3854_385495

/-- Given a recipe that uses 8 ounces of butter for some amount of flour,
    and knowing that 12 ounces of butter is used for 56 cups of flour
    when the recipe is quadrupled, prove that the original recipe
    requires 37 cups of flour. -/
theorem original_recipe_flour_amount :
  ∀ (x : ℚ),
  (8 : ℚ) / x = (12 : ℚ) / (4 * 56) →
  x = 37 := by
sorry

end NUMINAMATH_CALUDE_original_recipe_flour_amount_l3854_385495


namespace NUMINAMATH_CALUDE_choose_four_from_fifteen_l3854_385440

theorem choose_four_from_fifteen (n : ℕ) (k : ℕ) : Nat.choose 15 4 = 1365 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_fifteen_l3854_385440


namespace NUMINAMATH_CALUDE_distribution_theorem_l3854_385435

-- Define the number of books and students
def num_books : ℕ := 5
def num_students : ℕ := 3

-- Define a function to calculate the number of distribution methods
def distribution_methods (n_books : ℕ) (n_students : ℕ) : ℕ :=
  -- Implementation details are not provided as per the instructions
  sorry

-- Theorem statement
theorem distribution_theorem :
  distribution_methods num_books num_students = 150 := by
  sorry

end NUMINAMATH_CALUDE_distribution_theorem_l3854_385435


namespace NUMINAMATH_CALUDE_percentage_relation_l3854_385408

theorem percentage_relation (A B C x y : ℝ) : 
  A > 0 ∧ B > 0 ∧ C > 0 →
  A = B * (1 + x / 100) →
  A = C * (1 - y / 100) →
  A = 120 →
  B = 100 →
  C = 150 →
  x = 20 ∧ y = 20 := by
sorry

end NUMINAMATH_CALUDE_percentage_relation_l3854_385408


namespace NUMINAMATH_CALUDE_ceiling_neg_sqrt_64_over_9_l3854_385420

theorem ceiling_neg_sqrt_64_over_9 : ⌈-Real.sqrt (64/9)⌉ = -2 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_neg_sqrt_64_over_9_l3854_385420


namespace NUMINAMATH_CALUDE_B_power_150_is_identity_l3854_385406

def B : Matrix (Fin 3) (Fin 3) ℝ := !![0, 1, 0; 0, 0, 1; 1, 0, 0]

theorem B_power_150_is_identity :
  B^150 = (1 : Matrix (Fin 3) (Fin 3) ℝ) := by
  sorry

end NUMINAMATH_CALUDE_B_power_150_is_identity_l3854_385406


namespace NUMINAMATH_CALUDE_stealth_fighter_most_suitable_for_census_l3854_385413

/-- Represents a survey option -/
structure SurveyOption where
  name : String
  population_size : Nat
  feasibility_of_comprehensive_testing : Nat

/-- Defines the criteria for a survey to be suitable for a comprehensive survey (census) -/
def is_suitable_for_census (s : SurveyOption) : Prop :=
  s.population_size ≤ 1000 ∧ 
  s.importance_of_individual ≥ 9 ∧ 
  s.feasibility_of_comprehensive_testing ≥ 9

/-- The four survey options -/
def survey_options : List SurveyOption := [
  { name := "Car crash resistance", population_size := 10000, importance_of_individual := 5, feasibility_of_comprehensive_testing := 2 },
  { name := "Traffic regulation awareness", population_size := 1000000, importance_of_individual := 3, feasibility_of_comprehensive_testing := 1 },
  { name := "Light bulb service life", population_size := 100000, importance_of_individual := 2, feasibility_of_comprehensive_testing := 3 },
  { name := "Stealth fighter components", population_size := 100, importance_of_individual := 10, feasibility_of_comprehensive_testing := 10 }
]

/-- Theorem stating that the stealth fighter components survey is the most suitable for a comprehensive survey -/
theorem stealth_fighter_most_suitable_for_census :
  ∃ (s : SurveyOption), s ∈ survey_options ∧ 
  s.name = "Stealth fighter components" ∧
  is_suitable_for_census s ∧
  ∀ (t : SurveyOption), t ∈ survey_options → t.name ≠ "Stealth fighter components" → ¬(is_suitable_for_census t) :=
sorry

end NUMINAMATH_CALUDE_stealth_fighter_most_suitable_for_census_l3854_385413


namespace NUMINAMATH_CALUDE_trig_problem_l3854_385494

theorem trig_problem (α : Real) (h1 : 0 < α) (h2 : α < Real.pi / 2) 
  (h3 : Real.sin (α + Real.pi / 6) = 3 / 5) : 
  Real.cos (2 * α - Real.pi / 6) = 24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_trig_problem_l3854_385494


namespace NUMINAMATH_CALUDE_count_valid_n_l3854_385441

theorem count_valid_n : ∃! (s : Finset ℕ), 
  (∀ n ∈ s, 0 < n ∧ n < 35 ∧ ∃ k : ℕ, k > 0 ∧ n = k * (35 - n)) ∧ 
  s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_count_valid_n_l3854_385441


namespace NUMINAMATH_CALUDE_min_triangle_area_l3854_385499

/-- A point in the 2D plane with integer coordinates -/
structure IntPoint where
  x : ℤ
  y : ℤ

/-- The rectangle OABC -/
structure Rectangle where
  O : IntPoint
  B : IntPoint

/-- Checks if a point is inside the rectangle -/
def isInside (r : Rectangle) (p : IntPoint) : Prop :=
  0 ≤ p.x ∧ p.x ≤ r.B.x ∧ 0 ≤ p.y ∧ p.y ≤ r.B.y

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : IntPoint) : ℚ :=
  (1 / 2 : ℚ) * |p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)|

/-- The main theorem -/
theorem min_triangle_area (r : Rectangle) :
  r.O = ⟨0, 0⟩ → r.B = ⟨11, 8⟩ →
  ∃ (X : IntPoint), isInside r X ∧
    ∀ (Y : IntPoint), isInside r Y →
      triangleArea r.O r.B X ≤ triangleArea r.O r.B Y ∧
      triangleArea r.O r.B X = (1 / 2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_min_triangle_area_l3854_385499


namespace NUMINAMATH_CALUDE_direct_variation_problem_l3854_385484

/-- z varies directly as w -/
def direct_variation (z w : ℝ) := ∃ k : ℝ, z = k * w

theorem direct_variation_problem (z w : ℝ → ℝ) :
  (∀ x, direct_variation (z x) (w x)) →  -- z varies directly as w
  z 5 = 10 →                             -- z = 10 when w = 5
  w 5 = 5 →                              -- w = 5 when z = 10
  w (-15) = -15 →                        -- w = -15
  z (-15) = -30                          -- z = -30 when w = -15
  := by sorry

end NUMINAMATH_CALUDE_direct_variation_problem_l3854_385484


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3854_385451

theorem algebraic_expression_value (x : ℝ) (h : x^2 + x - 5 = 0) :
  (x - 1)^2 - x*(x - 3) + (x + 2)*(x - 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3854_385451


namespace NUMINAMATH_CALUDE_geometric_series_sum_l3854_385446

/-- The sum of a geometric series with given parameters -/
theorem geometric_series_sum (a₁ : ℝ) (q : ℝ) (aₙ : ℝ) (h₁ : a₁ = 100) (h₂ : q = 1/10) (h₃ : aₙ = 0.01) :
  (a₁ - aₙ * q) / (1 - q) = (10^5 - 1) / 900 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l3854_385446


namespace NUMINAMATH_CALUDE_log_inequality_l3854_385454

theorem log_inequality : ∀ x : ℝ, x > 0 → x - 1 ≥ Real.log x ∧ (x - 1 = Real.log x ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l3854_385454


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3854_385442

theorem sqrt_equation_solution :
  ∀ x : ℝ, (Real.sqrt ((1 + Real.sqrt 2) ^ x) + Real.sqrt ((1 - Real.sqrt 2) ^ x) = 2) ↔ (x = 0) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3854_385442


namespace NUMINAMATH_CALUDE_vector_relation_l3854_385452

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable (P A B C : V)

theorem vector_relation (h : (A - P) + 2 • (B - P) + 3 • (C - P) = 0) :
  P - A = (1/3) • (B - A) + (1/2) • (C - A) := by sorry

end NUMINAMATH_CALUDE_vector_relation_l3854_385452


namespace NUMINAMATH_CALUDE_increasing_seq_with_properties_is_geometric_l3854_385418

-- Define the sequence type
def Sequence := ℕ → ℝ

-- Define the properties
def Property1 (a : Sequence) : Prop :=
  ∀ i j, i > j → ∃ m, a i ^ 2 / a j = a m

def Property2 (a : Sequence) : Prop :=
  ∀ n, n ≥ 3 → ∃ k l, k > l ∧ a n = a k ^ 2 / a l

-- Define increasing sequence
def IncreasingSeq (a : Sequence) : Prop :=
  ∀ n m, n < m → a n < a m

-- Define geometric sequence
def GeometricSeq (a : Sequence) : Prop :=
  ∃ r, ∀ n, a (n + 1) = r * a n

-- State the theorem
theorem increasing_seq_with_properties_is_geometric (a : Sequence) 
  (h_inc : IncreasingSeq a) 
  (h1 : Property1 a) 
  (h2 : Property2 a) : 
  GeometricSeq a := by
  sorry

end NUMINAMATH_CALUDE_increasing_seq_with_properties_is_geometric_l3854_385418


namespace NUMINAMATH_CALUDE_twentieth_fisherman_catch_l3854_385432

theorem twentieth_fisherman_catch (total_fishermen : ℕ) (total_fish : ℕ) (fish_per_nineteen : ℕ) 
  (h1 : total_fishermen = 20)
  (h2 : total_fish = 10000)
  (h3 : fish_per_nineteen = 400) :
  total_fish - (total_fishermen - 1) * fish_per_nineteen = 2400 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_fisherman_catch_l3854_385432


namespace NUMINAMATH_CALUDE_paths_equal_combinations_correct_number_of_paths_l3854_385468

/-- The number of paths from (0,0) to (8,8) on an 8x8 grid -/
def number_of_paths : ℕ := 12870

/-- The size of the grid -/
def grid_size : ℕ := 8

/-- The total number of steps required to reach from (0,0) to (8,8) -/
def total_steps : ℕ := 16

/-- The number of right steps required -/
def right_steps : ℕ := 8

/-- The number of up steps required -/
def up_steps : ℕ := 8

/-- Theorem stating that the number of paths from (0,0) to (8,8) on an 8x8 grid
    is equal to the number of ways to choose 8 up steps out of 16 total steps -/
theorem paths_equal_combinations :
  number_of_paths = Nat.choose total_steps up_steps :=
sorry

/-- Theorem stating that the number of paths is correct -/
theorem correct_number_of_paths :
  number_of_paths = 12870 :=
sorry

end NUMINAMATH_CALUDE_paths_equal_combinations_correct_number_of_paths_l3854_385468


namespace NUMINAMATH_CALUDE_fraction_dislike_but_interested_l3854_385472

/-- Represents the student population at Novo Middle School -/
structure SchoolPopulation where
  total : ℕ
  artInterested : ℕ
  artUninterested : ℕ
  interestedLike : ℕ
  interestedDislike : ℕ
  uninterestedLike : ℕ
  uninterestedDislike : ℕ

/-- Theorem about the fraction of students who dislike art but are interested -/
theorem fraction_dislike_but_interested (pop : SchoolPopulation) : 
  pop.total = 200 ∧ 
  pop.artInterested = 150 ∧ 
  pop.artUninterested = 50 ∧
  pop.interestedLike = 105 ∧
  pop.interestedDislike = 45 ∧
  pop.uninterestedLike = 10 ∧
  pop.uninterestedDislike = 40 →
  (pop.interestedDislike : ℚ) / (pop.interestedDislike + pop.uninterestedDislike) = 9/17 := by
  sorry

#check fraction_dislike_but_interested

end NUMINAMATH_CALUDE_fraction_dislike_but_interested_l3854_385472


namespace NUMINAMATH_CALUDE_log_product_equality_l3854_385462

theorem log_product_equality : (Real.log 9 / Real.log 2) * (Real.log 4 / Real.log 3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equality_l3854_385462


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l3854_385453

theorem min_sum_of_squares (a b c t : ℝ) (h : a + b + c = t) :
  ∃ (m : ℝ), m = t^2 / 3 ∧ ∀ (x y z : ℝ), x + y + z = t → x^2 + y^2 + z^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l3854_385453


namespace NUMINAMATH_CALUDE_six_right_triangles_with_smallest_perimeter_l3854_385457

/-- A structure representing a triangle with integer sides -/
structure Triangle where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Check if a triangle is a right triangle -/
def is_right_triangle (t : Triangle) : Prop :=
  t.a ^ 2 + t.b ^ 2 = t.c ^ 2

/-- Calculate the perimeter of a triangle -/
def perimeter (t : Triangle) : ℕ :=
  t.a + t.b + t.c

/-- The set of six triangles with their side lengths -/
def six_triangles : List Triangle :=
  [⟨120, 288, 312⟩, ⟨144, 270, 306⟩, ⟨72, 320, 328⟩,
   ⟨45, 336, 339⟩, ⟨80, 315, 325⟩, ⟨180, 240, 300⟩]

/-- Theorem: There exist 6 rational right triangles with the same smallest possible perimeter of 720 -/
theorem six_right_triangles_with_smallest_perimeter :
  (∀ t ∈ six_triangles, is_right_triangle t) ∧
  (∀ t ∈ six_triangles, perimeter t = 720) ∧
  (∀ t : Triangle, is_right_triangle t → perimeter t < 720 → t ∉ six_triangles) :=
sorry

end NUMINAMATH_CALUDE_six_right_triangles_with_smallest_perimeter_l3854_385457


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3854_385449

-- Define the quadratic function
def f (x : ℝ) : ℝ := 4 * x^2 - x - 5

-- Define the solution set
def S : Set ℝ := { x | -1 ≤ x ∧ x ≤ 5/4 }

-- Theorem statement
theorem solution_set_of_inequality :
  { x : ℝ | f x ≤ 0 } = S :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3854_385449


namespace NUMINAMATH_CALUDE_num_correct_statements_is_zero_l3854_385485

/-- Definition of a frustum -/
structure Frustum where
  has_parallel_bases : Bool
  lateral_edges_converge : Bool

/-- The three statements about frustums -/
def statement1 (f : Frustum) : Prop :=
  true -- We don't need to define this precisely as it's always false

def statement2 (f : Frustum) : Prop :=
  f.has_parallel_bases

def statement3 (f : Frustum) : Prop :=
  f.has_parallel_bases

/-- Theorem: The number of correct statements is 0 -/
theorem num_correct_statements_is_zero : 
  (∀ f : Frustum, ¬statement1 f) ∧ 
  (∀ f : Frustum, f.has_parallel_bases ∧ f.lateral_edges_converge → statement2 f) ∧
  (∀ f : Frustum, f.has_parallel_bases ∧ f.lateral_edges_converge → statement3 f) →
  (¬∃ f : Frustum, statement1 f) ∧ 
  (¬∃ f : Frustum, statement2 f) ∧ 
  (¬∃ f : Frustum, statement3 f) :=
by
  sorry

#check num_correct_statements_is_zero

end NUMINAMATH_CALUDE_num_correct_statements_is_zero_l3854_385485


namespace NUMINAMATH_CALUDE_greatest_n_with_222_digits_l3854_385422

def a (n : ℕ) : ℚ := (2 * 10^(n+1) - 20 - 18*n) / 81

def number_of_digits (q : ℚ) : ℕ := sorry

theorem greatest_n_with_222_digits : 
  ∃ (n : ℕ), (∀ m : ℕ, number_of_digits (a m) = 222 → m ≤ n) ∧ 
  number_of_digits (a n) = 222 ∧ n = 222 := by sorry

end NUMINAMATH_CALUDE_greatest_n_with_222_digits_l3854_385422


namespace NUMINAMATH_CALUDE_population_growth_duration_l3854_385400

/-- Proves that given specific population growth rates and a total net increase,
    the duration of the period is 24 hours. -/
theorem population_growth_duration :
  let birth_rate : ℕ := 3  -- people per second
  let death_rate : ℕ := 1  -- people per second
  let net_increase_rate : ℕ := birth_rate - death_rate
  let total_net_increase : ℕ := 172800
  let duration_seconds : ℕ := total_net_increase / net_increase_rate
  let seconds_per_hour : ℕ := 3600
  duration_seconds / seconds_per_hour = 24 := by
  sorry

end NUMINAMATH_CALUDE_population_growth_duration_l3854_385400


namespace NUMINAMATH_CALUDE_quadratic_equality_existence_l3854_385477

theorem quadratic_equality_existence (P : ℝ → ℝ) (h : ∃ a b c : ℝ, ∀ x, P x = a * x^2 + b * x + c ∧ a ≠ 0) :
  ∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    P (b + c) = P a ∧ P (c + a) = P b ∧ P (a + b) = P c :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equality_existence_l3854_385477


namespace NUMINAMATH_CALUDE_factorial_difference_l3854_385434

theorem factorial_difference : Nat.factorial 10 - Nat.factorial 9 = 3265920 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_l3854_385434


namespace NUMINAMATH_CALUDE_excluded_students_average_mark_l3854_385439

theorem excluded_students_average_mark 
  (N : ℕ) 
  (A : ℚ) 
  (E : ℕ) 
  (A_remaining : ℚ) 
  (h1 : N = 25)
  (h2 : A = 80)
  (h3 : E = 5)
  (h4 : A_remaining = 95) :
  let A_excluded := ((N : ℚ) * A - (N - E : ℚ) * A_remaining) / E
  A_excluded = 20 := by
sorry

end NUMINAMATH_CALUDE_excluded_students_average_mark_l3854_385439


namespace NUMINAMATH_CALUDE_middle_group_frequency_l3854_385414

/-- Represents a frequency distribution histogram with 5 rectangles. -/
structure Histogram where
  rectangles : Fin 5 → ℝ
  total_sample : ℝ
  middle_equals_sum : rectangles 2 = (rectangles 0) + (rectangles 1) + (rectangles 3) + (rectangles 4)
  sample_size : total_sample = 100

/-- The frequency of the middle group in the histogram is 50. -/
theorem middle_group_frequency (h : Histogram) : h.rectangles 2 = 50 := by
  sorry

end NUMINAMATH_CALUDE_middle_group_frequency_l3854_385414


namespace NUMINAMATH_CALUDE_largest_angle_in_triangle_l3854_385458

theorem largest_angle_in_triangle (a b c : ℝ) (A : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  a + a * c + b * c = 2 * b →
  a - a * c + b * c = 2 * c →
  a = b + c + 2 * b * c * Real.cos A →
  A = 2 * π / 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_in_triangle_l3854_385458


namespace NUMINAMATH_CALUDE_max_angle_is_90_deg_l3854_385423

/-- A regular quadrilateral prism with height half the side length of its base -/
structure RegularQuadPrism where
  base_side : ℝ
  height : ℝ
  height_eq_half_base : height = base_side / 2

/-- A point on the edge AB of the prism -/
def PointOnAB (prism : RegularQuadPrism) := {x : ℝ // 0 ≤ x ∧ x ≤ prism.base_side}

/-- The angle A₁MC₁ where M is a point on AB -/
def angleA1MC1 (prism : RegularQuadPrism) (m : PointOnAB prism) : ℝ := sorry

/-- The maximum value of angle A₁MC₁ is 90° -/
theorem max_angle_is_90_deg (prism : RegularQuadPrism) :
  ∃ (m : PointOnAB prism), angleA1MC1 prism m = π / 2 ∧
  ∀ (m' : PointOnAB prism), angleA1MC1 prism m' ≤ π / 2 :=
sorry

end NUMINAMATH_CALUDE_max_angle_is_90_deg_l3854_385423


namespace NUMINAMATH_CALUDE_banana_distribution_l3854_385493

theorem banana_distribution (total_bananas : ℕ) : 
  (∀ (children : ℕ), 
    (children * 2 = total_bananas) →
    ((children - 160) * 4 = total_bananas)) →
  ∃ (actual_children : ℕ), actual_children = 320 := by
  sorry

end NUMINAMATH_CALUDE_banana_distribution_l3854_385493


namespace NUMINAMATH_CALUDE_sum_reciprocals_equals_negative_five_l3854_385498

theorem sum_reciprocals_equals_negative_five (x y : ℝ) 
  (eq1 : x^2 + Real.sqrt 3 * y = 4)
  (eq2 : y^2 + Real.sqrt 3 * x = 4)
  (neq : x ≠ y) :
  y / x + x / y = -5 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_equals_negative_five_l3854_385498


namespace NUMINAMATH_CALUDE_triangle_area_l3854_385480

/-- The area of a triangle with vertices at (0,0), (0,5), and (7,12) is 17.5 square units. -/
theorem triangle_area : 
  let v1 : ℝ × ℝ := (0, 0)
  let v2 : ℝ × ℝ := (0, 5)
  let v3 : ℝ × ℝ := (7, 12)
  (1/2 : ℝ) * |v2.2 - v1.2| * |v3.1 - v1.1| = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3854_385480


namespace NUMINAMATH_CALUDE_assignment_schemes_l3854_385404

def total_students : ℕ := 6
def selected_students : ℕ := 4
def restricted_students : ℕ := 2
def restricted_tasks : ℕ := 1

theorem assignment_schemes :
  (total_students.factorial / (total_students - selected_students).factorial) -
  (restricted_students * (total_students - 1).factorial / (total_students - selected_students).factorial) = 240 :=
sorry

end NUMINAMATH_CALUDE_assignment_schemes_l3854_385404


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3854_385463

theorem geometric_sequence_ratio (a₁ : ℝ) (q : ℝ) :
  (a₁ + a₁ * q + a₁ * q^2 = 3 * a₁) → (q = -2 ∨ q = 1) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3854_385463


namespace NUMINAMATH_CALUDE_pics_per_album_l3854_385459

-- Define the given conditions
def phone_pics : ℕ := 35
def camera_pics : ℕ := 5
def num_albums : ℕ := 5

-- Define the total number of pictures
def total_pics : ℕ := phone_pics + camera_pics

-- Theorem to prove
theorem pics_per_album : total_pics / num_albums = 8 := by
  sorry

end NUMINAMATH_CALUDE_pics_per_album_l3854_385459


namespace NUMINAMATH_CALUDE_parabola_equation_l3854_385419

/-- A parabola with axis of symmetry x = -2 has the standard form equation y² = 8x -/
theorem parabola_equation (p : ℝ) (h : p > 0) :
  (∀ x y : ℝ, y^2 = 2*p*x) → -- Standard form of parabola
  (-p/2 = -2) →             -- Axis of symmetry
  (∀ x y : ℝ, y^2 = 8*x) :=  -- Resulting equation
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l3854_385419


namespace NUMINAMATH_CALUDE_trapezoid_diagonal_midpoint_segment_length_l3854_385488

/-- A trapezoid with upper base length L and midline length m -/
structure Trapezoid (L m : ℝ) where
  upper_base : ℝ := L
  midline : ℝ := m

/-- The length of the segment connecting the midpoints of the two diagonals in a trapezoid -/
def diagonal_midpoint_segment_length (T : Trapezoid L m) : ℝ :=
  T.midline - T.upper_base

theorem trapezoid_diagonal_midpoint_segment_length (L m : ℝ) (T : Trapezoid L m) :
  diagonal_midpoint_segment_length T = m - L := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_diagonal_midpoint_segment_length_l3854_385488


namespace NUMINAMATH_CALUDE_f_unique_zero_x1_minus_2x2_bound_l3854_385470

noncomputable section

variables (a : ℝ) (h : a ≥ 0)

def f (x : ℝ) : ℝ := a * (Real.exp x - x - 1) - Real.log (x + 1) + x

def g (x : ℝ) : ℝ := a * Real.exp x + x

theorem f_unique_zero :
  ∃! x, f a x = 0 :=
sorry

theorem x1_minus_2x2_bound (x₁ x₂ : ℝ) 
  (h₁ : x₁ > -1) (h₂ : x₂ > -1) 
  (h₃ : f a x₁ = g a x₁ - g a x₂) :
  x₁ - 2 * x₂ ≥ 1 - 2 * Real.log 2 :=
sorry

end NUMINAMATH_CALUDE_f_unique_zero_x1_minus_2x2_bound_l3854_385470


namespace NUMINAMATH_CALUDE_pet_store_birds_l3854_385475

/-- Calculates the total number of birds in a pet store given the number of cages and birds per cage. -/
def total_birds (num_cages : ℕ) (parrots_per_cage : ℕ) (parakeets_per_cage : ℕ) : ℕ :=
  num_cages * (parrots_per_cage + parakeets_per_cage)

/-- Proves that the pet store has 72 birds in total. -/
theorem pet_store_birds : total_birds 9 2 6 = 72 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_birds_l3854_385475


namespace NUMINAMATH_CALUDE_correct_system_l3854_385416

/-- Represents the length of rope needed to go around the tree once. -/
def y : ℝ := sorry

/-- Represents the total length of the rope. -/
def x : ℝ := sorry

/-- The condition that when the rope goes around the tree 3 times, there will be an extra 5 feet of rope left. -/
axiom three_wraps : 3 * y + 5 = x

/-- The condition that when the rope goes around the tree 4 times, there will be 2 feet less of rope left. -/
axiom four_wraps : 4 * y - 2 = x

/-- Theorem stating that the system of equations correctly represents the problem. -/
theorem correct_system : (3 * y + 5 = x) ∧ (4 * y - 2 = x) := by sorry

end NUMINAMATH_CALUDE_correct_system_l3854_385416


namespace NUMINAMATH_CALUDE_negative_division_equals_positive_division_negative_three_hundred_by_negative_twenty_five_l3854_385491

theorem negative_division_equals_positive (x y : ℤ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (-x) / (-y) = x / y :=
sorry

theorem division_negative_three_hundred_by_negative_twenty_five :
  (-300) / (-25) = 12 :=
sorry

end NUMINAMATH_CALUDE_negative_division_equals_positive_division_negative_three_hundred_by_negative_twenty_five_l3854_385491


namespace NUMINAMATH_CALUDE_remainder_10_pow_23_minus_7_mod_6_l3854_385401

theorem remainder_10_pow_23_minus_7_mod_6 :
  (10^23 - 7) % 6 = 3 := by sorry

end NUMINAMATH_CALUDE_remainder_10_pow_23_minus_7_mod_6_l3854_385401


namespace NUMINAMATH_CALUDE_function_inequality_l3854_385460

theorem function_inequality (a : ℝ) : 
  (∀ x : ℝ, x > 0 → x * Real.log x - a * x ≥ -x^2 - 2) → a ≤ -2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3854_385460
