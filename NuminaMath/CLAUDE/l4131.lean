import Mathlib

namespace NUMINAMATH_CALUDE_tyrone_eric_marbles_l4131_413128

theorem tyrone_eric_marbles (tyrone_initial : ℕ) (eric_initial : ℕ) 
  (h1 : tyrone_initial = 97) 
  (h2 : eric_initial = 11) : 
  ∃ (marbles_given : ℕ), 
    marbles_given = 25 ∧ 
    (tyrone_initial - marbles_given = 2 * (eric_initial + marbles_given)) := by
  sorry

end NUMINAMATH_CALUDE_tyrone_eric_marbles_l4131_413128


namespace NUMINAMATH_CALUDE_total_pencils_l4131_413135

/-- The number of colors in a rainbow -/
def rainbow_colors : ℕ := 7

/-- The number of people who bought color boxes -/
def number_of_people : ℕ := 3

/-- The number of pencils in each color box -/
def pencils_per_box : ℕ := rainbow_colors

theorem total_pencils :
  rainbow_colors * number_of_people = 21 :=
sorry

end NUMINAMATH_CALUDE_total_pencils_l4131_413135


namespace NUMINAMATH_CALUDE_angle_d_is_190_l4131_413148

/-- A quadrilateral with angles A, B, C, and D. -/
structure Quadrilateral where
  angleA : Real
  angleB : Real
  angleC : Real
  angleD : Real
  sum_360 : angleA + angleB + angleC + angleD = 360

/-- Theorem: In a quadrilateral ABCD, if ∠A = 70°, ∠B = 60°, and ∠C = 40°, then ∠D = 190°. -/
theorem angle_d_is_190 (q : Quadrilateral) 
  (hA : q.angleA = 70)
  (hB : q.angleB = 60)
  (hC : q.angleC = 40) : 
  q.angleD = 190 := by
  sorry

end NUMINAMATH_CALUDE_angle_d_is_190_l4131_413148


namespace NUMINAMATH_CALUDE_quadratic_roots_ratio_l4131_413187

theorem quadratic_roots_ratio (q : ℝ) : 
  (∃ r s : ℝ, r ≠ 0 ∧ s ≠ 0 ∧ r / s = 3 ∧ 
   r + s = -8 ∧ r * s = q ∧ 
   ∀ x : ℝ, x^2 + 8*x + q = 0 ↔ (x = r ∨ x = s)) → 
  q = 12 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_ratio_l4131_413187


namespace NUMINAMATH_CALUDE_smaller_root_of_equation_l4131_413146

theorem smaller_root_of_equation :
  let f : ℝ → ℝ := λ x => (x - 1/3)^2 + (x - 1/3)*(x + 1/6)
  (f (1/12) = 0) ∧ (∀ y < 1/12, f y ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_smaller_root_of_equation_l4131_413146


namespace NUMINAMATH_CALUDE_odd_function_property_l4131_413145

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_property (f : ℝ → ℝ) (h_odd : IsOdd f) (h_sum : ∀ x, f (x + 1) + f x = 0) :
  f 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l4131_413145


namespace NUMINAMATH_CALUDE_problem_1_l4131_413184

theorem problem_1 (x : ℝ) (a : ℝ) : x - 1/x = 3 → a = x^2 + 1/x^2 → a = 11 := by
  sorry


end NUMINAMATH_CALUDE_problem_1_l4131_413184


namespace NUMINAMATH_CALUDE_total_amount_192_rupees_l4131_413158

/-- Represents the denominations of rupee notes -/
inductive Denomination
  | One
  | Five
  | Ten

/-- Calculates the value of a single note of a given denomination -/
def noteValue (d : Denomination) : Nat :=
  match d with
  | Denomination.One => 1
  | Denomination.Five => 5
  | Denomination.Ten => 10

/-- Represents the collection of notes -/
structure NoteCollection where
  totalNotes : Nat
  denominations : List Denomination
  equalDenominations : List.length denominations = 3
  equalDistribution : totalNotes % (List.length denominations) = 0

/-- Theorem stating that a collection of 36 notes equally distributed among
    one-rupee, five-rupee, and ten-rupee denominations totals 192 rupees -/
theorem total_amount_192_rupees (nc : NoteCollection)
    (h1 : nc.totalNotes = 36)
    (h2 : nc.denominations = [Denomination.One, Denomination.Five, Denomination.Ten]) :
    (nc.totalNotes / 3) * (noteValue Denomination.One +
                           noteValue Denomination.Five +
                           noteValue Denomination.Ten) = 192 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_192_rupees_l4131_413158


namespace NUMINAMATH_CALUDE_arithmetic_sequence_triangle_cos_identity_l4131_413191

theorem arithmetic_sequence_triangle_cos_identity (A B C : Real) (a b c : Real) :
  -- Triangle conditions
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  -- Arithmetic sequence condition
  2 * b = a + c ∧
  -- Side-angle relationships (law of sines)
  a / (Real.sin A) = b / (Real.sin B) ∧
  b / (Real.sin B) = c / (Real.sin C) →
  -- Theorem to prove
  5 * (Real.cos A) - 4 * (Real.cos A) * (Real.cos C) + 5 * (Real.cos C) = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_triangle_cos_identity_l4131_413191


namespace NUMINAMATH_CALUDE_complex_imaginary_part_l4131_413160

theorem complex_imaginary_part (z : ℂ) : (3 - 4*I) * z = Complex.abs (4 + 3*I) → Complex.im z = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_complex_imaginary_part_l4131_413160


namespace NUMINAMATH_CALUDE_max_profit_at_max_price_l4131_413174

-- Define the problem parameters
def raw_material_price : ℝ := 30
def min_selling_price : ℝ := 30
def max_selling_price : ℝ := 60
def additional_cost : ℝ := 450

-- Define the sales volume function
def sales_volume (x : ℝ) : ℝ := -2 * x + 200

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - raw_material_price) * sales_volume x - additional_cost

-- State the theorem
theorem max_profit_at_max_price :
  ∀ x ∈ Set.Icc min_selling_price max_selling_price,
    profit x ≤ profit max_selling_price ∧
    profit max_selling_price = 1950 :=
by sorry

end NUMINAMATH_CALUDE_max_profit_at_max_price_l4131_413174


namespace NUMINAMATH_CALUDE_restaurant_production_l4131_413103

/-- Represents a restaurant's daily production of pizzas and hot dogs -/
structure Restaurant where
  hotdogs : ℕ
  pizza_excess : ℕ

/-- Calculates the total number of pizzas and hot dogs made in a given number of days -/
def total_production (r : Restaurant) (days : ℕ) : ℕ :=
  (r.hotdogs + (r.hotdogs + r.pizza_excess)) * days

/-- Theorem stating that a restaurant making 40 more pizzas than hot dogs daily,
    and 60 hot dogs per day, will produce 4800 pizzas and hot dogs in 30 days -/
theorem restaurant_production :
  ∀ (r : Restaurant),
    r.hotdogs = 60 →
    r.pizza_excess = 40 →
    total_production r 30 = 4800 :=
by
  sorry

end NUMINAMATH_CALUDE_restaurant_production_l4131_413103


namespace NUMINAMATH_CALUDE_bank_layoff_optimization_l4131_413113

/-- Represents the economic benefit function for the bank -/
def economic_benefit (x : ℕ) : ℚ :=
  (320 - x) * (20 + 0.2 * x) - 6 * x

/-- Represents the constraint on the number of employees that can be laid off -/
def valid_layoff (x : ℕ) : Prop :=
  x ≤ 80

theorem bank_layoff_optimization :
  ∃ (x : ℕ), valid_layoff x ∧
    (∀ (y : ℕ), valid_layoff y → economic_benefit x ≥ economic_benefit y) ∧
    economic_benefit x = 9160 :=
sorry

end NUMINAMATH_CALUDE_bank_layoff_optimization_l4131_413113


namespace NUMINAMATH_CALUDE_homework_theorem_l4131_413178

def homework_problem (total_time math_percentage other_time : ℝ) : Prop :=
  let math_time := math_percentage * total_time
  let science_time := total_time - math_time - other_time
  (science_time / total_time) * 100 = 40

theorem homework_theorem :
  ∀ (total_time math_percentage other_time : ℝ),
    total_time = 150 →
    math_percentage = 0.3 →
    other_time = 45 →
    homework_problem total_time math_percentage other_time :=
by sorry

end NUMINAMATH_CALUDE_homework_theorem_l4131_413178


namespace NUMINAMATH_CALUDE_system_solution_l4131_413133

theorem system_solution : 
  let x : ℚ := -135/41
  let y : ℚ := 192/41
  (7 * x = -9 - 3 * y) ∧ (2 * x = 5 * y - 30) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l4131_413133


namespace NUMINAMATH_CALUDE_least_m_is_207_l4131_413167

def x : ℕ → ℚ
  | 0 => 7
  | n + 1 => (x n ^ 2 + 6 * x n + 8) / (x n + 7)

def is_least_m (m : ℕ) : Prop :=
  x m ≤ 5 + 1 / 2^15 ∧ ∀ k < m, x k > 5 + 1 / 2^15

theorem least_m_is_207 : is_least_m 207 := by
  sorry

end NUMINAMATH_CALUDE_least_m_is_207_l4131_413167


namespace NUMINAMATH_CALUDE_cosine_identity_from_system_l4131_413115

theorem cosine_identity_from_system (A B C a b c : ℝ) 
  (eq1 : a = b * Real.cos C + c * Real.cos B)
  (eq2 : b = c * Real.cos A + a * Real.cos C)
  (eq3 : c = a * Real.cos B + b * Real.cos A)
  (h : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) : 
  (Real.cos A)^2 + (Real.cos B)^2 + (Real.cos C)^2 + 2 * Real.cos A * Real.cos B * Real.cos C = 1 := by
sorry

end NUMINAMATH_CALUDE_cosine_identity_from_system_l4131_413115


namespace NUMINAMATH_CALUDE_rachel_age_when_emily_half_l4131_413112

theorem rachel_age_when_emily_half (emily_age rachel_age : ℕ) : 
  rachel_age = emily_age + 4 → 
  ∃ (x : ℕ), x = rachel_age ∧ x / 2 = x - 4 → 
  x = 8 := by sorry

end NUMINAMATH_CALUDE_rachel_age_when_emily_half_l4131_413112


namespace NUMINAMATH_CALUDE_negative_1651_mod_9_l4131_413117

theorem negative_1651_mod_9 : ∃! n : ℤ, 0 ≤ n ∧ n < 9 ∧ -1651 ≡ n [ZMOD 9] ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_negative_1651_mod_9_l4131_413117


namespace NUMINAMATH_CALUDE_alex_score_l4131_413129

theorem alex_score (total_students : ℕ) (graded_students : ℕ) (initial_average : ℚ) (final_average : ℚ) :
  total_students = 20 →
  graded_students = 19 →
  initial_average = 72 →
  final_average = 74 →
  (graded_students * initial_average + (total_students - graded_students) * 
    ((total_students * final_average - graded_students * initial_average) / (total_students - graded_students))) / total_students = final_average →
  (total_students * final_average - graded_students * initial_average) = 112 := by
  sorry

end NUMINAMATH_CALUDE_alex_score_l4131_413129


namespace NUMINAMATH_CALUDE_count_congruent_is_71_l4131_413119

/-- The number of positive integers less than 500 that are congruent to 3 (mod 7) -/
def count_congruent : ℕ :=
  (Finset.filter (fun n => n % 7 = 3) (Finset.range 500)).card

/-- Theorem: The count of positive integers less than 500 that are congruent to 3 (mod 7) is 71 -/
theorem count_congruent_is_71 : count_congruent = 71 := by
  sorry

end NUMINAMATH_CALUDE_count_congruent_is_71_l4131_413119


namespace NUMINAMATH_CALUDE_composition_equation_solution_l4131_413189

theorem composition_equation_solution :
  let δ : ℝ → ℝ := λ x ↦ 5 * x + 9
  let φ : ℝ → ℝ := λ x ↦ 7 * x + 6
  ∃ x : ℝ, δ (φ x) = -4 ∧ x = -43/35 := by
  sorry

end NUMINAMATH_CALUDE_composition_equation_solution_l4131_413189


namespace NUMINAMATH_CALUDE_geometry_relations_l4131_413164

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (parallel : Plane → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem geometry_relations 
  (l m : Line) (α β : Plane) 
  (h1 : perpendicular l α) 
  (h2 : contains β m) :
  (parallel α β → perpendicular_lines l m) ∧
  (parallel_lines l m → perpendicular_planes α β) ∧
  ¬(perpendicular_planes α β → parallel_lines l m) ∧
  ¬(perpendicular_lines l m → parallel α β) :=
by sorry

end NUMINAMATH_CALUDE_geometry_relations_l4131_413164


namespace NUMINAMATH_CALUDE_band_arrangement_l4131_413153

theorem band_arrangement (total_members : Nat) (min_row : Nat) (max_row : Nat) : 
  total_members = 108 → min_row = 10 → max_row = 18 → 
  (∃! n : Nat, n = (Finset.filter (λ x : Nat => min_row ≤ x ∧ x ≤ max_row ∧ total_members % x = 0) 
    (Finset.range (max_row - min_row + 1))).card ∧ n = 2) := by
  sorry

end NUMINAMATH_CALUDE_band_arrangement_l4131_413153


namespace NUMINAMATH_CALUDE_alexa_lemonade_profit_l4131_413139

/-- Calculates the profit from a lemonade stand given the price per cup,
    cost of ingredients, and number of cups sold. -/
def lemonade_profit (price_per_cup : ℕ) (ingredient_cost : ℕ) (cups_sold : ℕ) : ℕ :=
  price_per_cup * cups_sold - ingredient_cost

/-- Proves that given the specific conditions of Alexa's lemonade stand,
    her desired profit is $80. -/
theorem alexa_lemonade_profit :
  lemonade_profit 2 20 50 = 80 := by
  sorry

end NUMINAMATH_CALUDE_alexa_lemonade_profit_l4131_413139


namespace NUMINAMATH_CALUDE_octagon_side_length_l4131_413102

/-- Given a regular pentagon with side length 16 cm, prove that if the same total length of yarn
    is used to make a regular octagon, then the length of one side of the octagon is 10 cm. -/
theorem octagon_side_length (pentagon_side : ℝ) (octagon_side : ℝ) : 
  pentagon_side = 16 → 5 * pentagon_side = 8 * octagon_side → octagon_side = 10 := by
  sorry

end NUMINAMATH_CALUDE_octagon_side_length_l4131_413102


namespace NUMINAMATH_CALUDE_unique_integer_satisfying_conditions_l4131_413105

theorem unique_integer_satisfying_conditions :
  ∃! (x : ℤ), 1 < x ∧ x < 9 ∧ 2 < x ∧ x < 15 ∧ -1 < x ∧ x < 7 ∧ 0 < x ∧ x < 4 ∧ x + 1 < 5 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_satisfying_conditions_l4131_413105


namespace NUMINAMATH_CALUDE_function_property_l4131_413100

/-- Piecewise function f(x) as described in the problem -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  if x < 3 then a * x + b else 7 - 2 * x

/-- The main theorem to prove -/
theorem function_property (a b : ℝ) :
  (∀ x, f a b (f a b x) = x) → a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l4131_413100


namespace NUMINAMATH_CALUDE_stock_price_increase_l4131_413161

/-- Given a stock price that decreased by 8% in the first year and had a net percentage change of 1.20% over two years, the percentage increase in the second year was 10%. -/
theorem stock_price_increase (initial_price : ℝ) (h : initial_price > 0) :
  let price_after_decrease := initial_price * (1 - 0.08)
  let final_price := initial_price * (1 + 0.012)
  let increase_percentage := (final_price / price_after_decrease - 1) * 100
  increase_percentage = 10 := by sorry

end NUMINAMATH_CALUDE_stock_price_increase_l4131_413161


namespace NUMINAMATH_CALUDE_work_completion_time_l4131_413185

theorem work_completion_time 
  (x : ℝ) 
  (hx : x > 0) 
  (h_combined : 1/x + 1/8 = 3/16) : x = 16 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l4131_413185


namespace NUMINAMATH_CALUDE_line_equation_correct_l4131_413180

/-- A line in 2D space --/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space --/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A vector in 2D space --/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line --/
def pointOnLine (l : Line2D) (p : Point2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if a vector is parallel to a line --/
def vectorParallelToLine (l : Line2D) (v : Vector2D) : Prop :=
  l.a * v.y = l.b * v.x

/-- The main theorem --/
theorem line_equation_correct (l : Line2D) (p : Point2D) (v : Vector2D) : 
  l.a = 1 ∧ l.b = 2 ∧ l.c = -1 ∧
  p.x = 1 ∧ p.y = 0 ∧
  v.x = 2 ∧ v.y = -1 →
  pointOnLine l p ∧ vectorParallelToLine l v := by
  sorry

end NUMINAMATH_CALUDE_line_equation_correct_l4131_413180


namespace NUMINAMATH_CALUDE_distribution_property_l4131_413179

-- Define a type for our distribution
def Distribution (α : Type*) := α → ℝ

-- Define properties of our distribution
def IsSymmetric (f : Distribution ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

def StandardDeviationProperty (f : Distribution ℝ) (a d : ℝ) : Prop :=
  ∫ x in Set.Icc (a - d) (a + d), f x = 0.68

-- Main theorem
theorem distribution_property (f : Distribution ℝ) (a d : ℝ) 
  (h_symmetric : IsSymmetric f a) 
  (h_std_dev : StandardDeviationProperty f a d) :
  ∫ x in Set.Iic (a + d), f x = 0.84 := by
  sorry

end NUMINAMATH_CALUDE_distribution_property_l4131_413179


namespace NUMINAMATH_CALUDE_recliner_sales_increase_l4131_413126

theorem recliner_sales_increase 
  (price_decrease : ℝ) 
  (gross_increase : ℝ) 
  (h1 : price_decrease = 0.20) 
  (h2 : gross_increase = 0.20000000000000014) : 
  (1 + gross_increase) / (1 - price_decrease) - 1 = 0.5 := by sorry

end NUMINAMATH_CALUDE_recliner_sales_increase_l4131_413126


namespace NUMINAMATH_CALUDE_sum_of_distinct_prime_factors_l4131_413116

def expression : ℕ := 7^7 - 7^3

theorem sum_of_distinct_prime_factors : 
  (Finset.sum (Finset.filter Nat.Prime (Finset.range (expression + 1)))
    (λ p => if p ∣ expression then p else 0)) = 17 := by sorry

end NUMINAMATH_CALUDE_sum_of_distinct_prime_factors_l4131_413116


namespace NUMINAMATH_CALUDE_no_valid_base_for_122_square_l4131_413123

theorem no_valid_base_for_122_square : ¬ ∃ (b : ℕ), b > 1 ∧ ∃ (n : ℕ), b^2 + 2*b + 2 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_base_for_122_square_l4131_413123


namespace NUMINAMATH_CALUDE_quadratic_root_sum_l4131_413131

theorem quadratic_root_sum (m : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 - (2*m - 2)*x + (m^2 - 2*m) = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ + x₂ = 10 →
  m = 6 := by sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_l4131_413131


namespace NUMINAMATH_CALUDE_gcf_of_270_and_180_l4131_413163

theorem gcf_of_270_and_180 : Nat.gcd 270 180 = 90 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_270_and_180_l4131_413163


namespace NUMINAMATH_CALUDE_chef_total_plates_l4131_413121

theorem chef_total_plates (lobster_rolls spicy_hot_noodles seafood_noodles : ℕ) 
  (h1 : lobster_rolls = 25)
  (h2 : spicy_hot_noodles = 14)
  (h3 : seafood_noodles = 16) :
  lobster_rolls + spicy_hot_noodles + seafood_noodles = 55 := by
  sorry

end NUMINAMATH_CALUDE_chef_total_plates_l4131_413121


namespace NUMINAMATH_CALUDE_inequality_proof_l4131_413141

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  1 ≤ ((x + y) * (x^3 + y^3)) / ((x^2 + y^2)^2) ∧ 
  ((x + y) * (x^3 + y^3)) / ((x^2 + y^2)^2) ≤ 9/8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4131_413141


namespace NUMINAMATH_CALUDE_triangle_proof_l4131_413172

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that under certain conditions, BD = b and cos(ABC) = 7/12 -/
theorem triangle_proof (a b c : ℝ) (A B C : ℝ) (D : ℝ × ℝ) :
  b^2 = a * c →
  D.1 ≥ 0 ∧ D.1 ≤ c →  -- D lies on AC
  b * Real.sin B = a * Real.sin C →
  2 * (c - D.1) = D.1 →  -- AD = 2DC
  (b = Real.sqrt (a * c)) ∧
  (Real.cos B = 7 / 12) :=
by sorry

end NUMINAMATH_CALUDE_triangle_proof_l4131_413172


namespace NUMINAMATH_CALUDE_smallest_coin_set_l4131_413110

/-- Represents the different types of coins available --/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter
  | HalfDollar

/-- The value of each coin in cents --/
def coinValue : Coin → ℕ
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25
  | Coin.HalfDollar => 50

/-- A function that checks if a given set of coins can pay any amount from 1 to n cents --/
def canPayAllAmounts (coins : List Coin) (n : ℕ) : Prop :=
  ∀ (amount : ℕ), 1 ≤ amount ∧ amount ≤ n →
    ∃ (subset : List Coin), subset ⊆ coins ∧ (subset.map coinValue).sum = amount

/-- The main theorem stating that 10 is the smallest number of coins needed --/
theorem smallest_coin_set :
  ∃ (coins : List Coin),
    coins.length = 10 ∧
    canPayAllAmounts coins 149 ∧
    ∀ (other_coins : List Coin),
      canPayAllAmounts other_coins 149 →
      other_coins.length ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_smallest_coin_set_l4131_413110


namespace NUMINAMATH_CALUDE_citrus_yield_probability_l4131_413177

/-- Represents the yield recovery rates in the first year -/
def first_year_rates : List ℝ := [1.0, 0.9, 0.8]

/-- Represents the probabilities of yield recovery rates in the first year -/
def first_year_probs : List ℝ := [0.2, 0.4, 0.4]

/-- Represents the growth rates in the second year -/
def second_year_rates : List ℝ := [1.5, 1.25, 1.0]

/-- Represents the probabilities of growth rates in the second year -/
def second_year_probs : List ℝ := [0.3, 0.3, 0.4]

/-- Calculates the probability of reaching exactly the pre-disaster yield after two years -/
def probability_pre_disaster_yield (f_rates : List ℝ) (f_probs : List ℝ) (s_rates : List ℝ) (s_probs : List ℝ) : ℝ :=
  sorry

theorem citrus_yield_probability :
  probability_pre_disaster_yield first_year_rates first_year_probs second_year_rates second_year_probs = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_citrus_yield_probability_l4131_413177


namespace NUMINAMATH_CALUDE_power_sum_equals_two_l4131_413183

theorem power_sum_equals_two : (-1 : ℝ)^2 + (1/3 : ℝ)^0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equals_two_l4131_413183


namespace NUMINAMATH_CALUDE_base_b_cube_iff_six_l4131_413130

/-- Represents a number in base b --/
def base_b_number (b : ℕ) : ℕ := b^2 + 4*b + 4

/-- Checks if a natural number is a perfect cube --/
def is_cube (n : ℕ) : Prop := ∃ m : ℕ, m^3 = n

/-- The main theorem: 144 in base b is a cube iff b = 6 --/
theorem base_b_cube_iff_six (b : ℕ) : 
  (b > 0) → (is_cube (base_b_number b) ↔ b = 6) := by
sorry

end NUMINAMATH_CALUDE_base_b_cube_iff_six_l4131_413130


namespace NUMINAMATH_CALUDE_triangle_angle_ratio_range_l4131_413157

theorem triangle_angle_ratio_range (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 →
  A + B + C = π →
  S = (1/2) * b * c * Real.sin A →
  a^2 = 2*S + (b-c)^2 →
  1 - (1/2) * Real.sin A = (b^2 + c^2 - a^2) / (2*b*c) →
  ∃ (l u : ℝ), l = 2 * Real.sqrt 2 ∧ u = 59/15 ∧
    (∀ x, l ≤ x ∧ x < u ↔ 
      ∃ (B' C' : ℝ), 0 < B' ∧ B' < π/2 ∧ 0 < C' ∧ C' < π/2 ∧
        x = (2 * Real.sin B'^2 + Real.sin C'^2) / (Real.sin B' * Real.sin C')) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_ratio_range_l4131_413157


namespace NUMINAMATH_CALUDE_kid_ticket_cost_prove_kid_ticket_cost_l4131_413125

theorem kid_ticket_cost (adult_price : ℝ) (total_tickets : ℕ) (total_profit : ℝ) (kid_tickets : ℕ) : ℝ :=
  let adult_tickets := total_tickets - kid_tickets
  let adult_profit := adult_tickets * adult_price
  let kid_profit := total_profit - adult_profit
  let kid_price := kid_profit / kid_tickets
  kid_price

theorem prove_kid_ticket_cost :
  kid_ticket_cost 6 175 750 75 = 2 := by
  sorry

end NUMINAMATH_CALUDE_kid_ticket_cost_prove_kid_ticket_cost_l4131_413125


namespace NUMINAMATH_CALUDE_geometry_class_eligibility_l4131_413104

def minimum_score (q1 q2 q3 : ℚ) : ℚ :=
  4 * (85 : ℚ) / 100 - (q1 + q2 + q3)

theorem geometry_class_eligibility 
  (q1 q2 q3 : ℚ) 
  (h1 : q1 = 85 / 100) 
  (h2 : q2 = 80 / 100) 
  (h3 : q3 = 90 / 100) : 
  minimum_score q1 q2 q3 = 85 / 100 := by
sorry

end NUMINAMATH_CALUDE_geometry_class_eligibility_l4131_413104


namespace NUMINAMATH_CALUDE_solution_set_for_a_3_f_geq_1_iff_a_leq_1_or_geq_3_l4131_413199

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := |x| + 2*|x + 2 - a|

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := g a (x - 2)

-- Theorem for part (1)
theorem solution_set_for_a_3 :
  {x : ℝ | g 3 x ≤ 4} = Set.Icc (-2/3) 2 := by sorry

-- Theorem for part (2)
theorem f_geq_1_iff_a_leq_1_or_geq_3 :
  (∀ x, f a x ≥ 1) ↔ (a ≤ 1 ∨ a ≥ 3) := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_3_f_geq_1_iff_a_leq_1_or_geq_3_l4131_413199


namespace NUMINAMATH_CALUDE_cosine_C_value_l4131_413181

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to angles A, B, C respectively

-- State the theorem
theorem cosine_C_value (t : Triangle) 
  (h1 : t.c = 2 * t.a)  -- Given condition: c = 2a
  (h2 : Real.sin t.A / Real.sin t.B = 2/3)  -- Given condition: sin A / sin B = 2/3
  : Real.cos t.C = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_C_value_l4131_413181


namespace NUMINAMATH_CALUDE_copy_paper_purchase_solution_l4131_413155

/-- Represents the purchase of copy papers -/
structure CopyPaperPurchase where
  white : ℕ
  colored : ℕ

/-- The total cost of the purchase in yuan -/
def total_cost (p : CopyPaperPurchase) : ℕ := 80 * p.white + 180 * p.colored

/-- The relationship between white and colored paper quantities -/
def quantity_relation (p : CopyPaperPurchase) : Prop :=
  p.white = 5 * p.colored - 3

/-- The main theorem stating the solution to the problem -/
theorem copy_paper_purchase_solution :
  ∃ (p : CopyPaperPurchase),
    total_cost p = 2660 ∧
    quantity_relation p ∧
    p.white = 22 ∧
    p.colored = 5 := by
  sorry

end NUMINAMATH_CALUDE_copy_paper_purchase_solution_l4131_413155


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l4131_413169

theorem simplify_fraction_product : 8 * (15 / 4) * (-40 / 45) = -2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l4131_413169


namespace NUMINAMATH_CALUDE_x_to_y_equals_negative_eight_l4131_413114

theorem x_to_y_equals_negative_eight (x y : ℝ) (h : |x + 2| + (y - 3)^2 = 0) : x^y = -8 := by
  sorry

end NUMINAMATH_CALUDE_x_to_y_equals_negative_eight_l4131_413114


namespace NUMINAMATH_CALUDE_systematic_sampling_l4131_413182

/-- Systematic sampling problem -/
theorem systematic_sampling
  (total_employees : ℕ)
  (sample_size : ℕ)
  (fifth_group_number : ℕ)
  (h1 : total_employees = 200)
  (h2 : sample_size = 40)
  (h3 : fifth_group_number = 22) :
  let first_group_number := 2
  let group_difference := (fifth_group_number - first_group_number) / 4
  (9 * group_difference + first_group_number) = 47 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_l4131_413182


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l4131_413188

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, |x - 1| - |x + 1| ≤ 3) ↔ (∃ x₀ : ℝ, |x₀ - 1| - |x₀ + 1| > 3) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l4131_413188


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l4131_413159

/-- A line in 2D space represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A point lies on a line if it satisfies the line's equation -/
def on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

theorem perpendicular_line_equation (given_line : Line) (point : Point) :
  given_line = Line.mk 1 (-2) (-3) →
  point = Point.mk 3 1 →
  ∃ (result_line : Line),
    perpendicular given_line result_line ∧
    on_line point result_line ∧
    result_line = Line.mk 2 1 (-7) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l4131_413159


namespace NUMINAMATH_CALUDE_trevors_brother_age_l4131_413111

/-- Trevor's age a decade ago -/
def trevors_age_decade_ago : ℕ := 16

/-- Current year -/
def current_year : ℕ := 2023

/-- Trevor's current age -/
def trevors_current_age : ℕ := trevors_age_decade_ago + 10

/-- Trevor's age 20 years ago -/
def trevors_age_20_years_ago : ℕ := trevors_current_age - 20

/-- Trevor's brother's age 20 years ago -/
def brothers_age_20_years_ago : ℕ := 2 * trevors_age_20_years_ago

/-- Trevor's brother's current age -/
def brothers_current_age : ℕ := brothers_age_20_years_ago + 20

theorem trevors_brother_age : brothers_current_age = 32 := by
  sorry

end NUMINAMATH_CALUDE_trevors_brother_age_l4131_413111


namespace NUMINAMATH_CALUDE_percentage_difference_in_gain_l4131_413144

/-- Given an article with cost price, and two selling prices, calculate the percentage difference in gain -/
theorem percentage_difference_in_gain 
  (cost_price : ℝ) 
  (selling_price1 : ℝ) 
  (selling_price2 : ℝ) 
  (h1 : cost_price = 250) 
  (h2 : selling_price1 = 350) 
  (h3 : selling_price2 = 340) : 
  (selling_price1 - cost_price - (selling_price2 - cost_price)) / (selling_price2 - cost_price) * 100 = 100 / 9 := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_in_gain_l4131_413144


namespace NUMINAMATH_CALUDE_unique_solution_for_P_squared_prime_l4131_413195

/-- The polynomial P(n) = n^3 - n^2 - 5n + 2 -/
def P (n : ℤ) : ℤ := n^3 - n^2 - 5*n + 2

/-- A predicate to check if a number is prime -/
def isPrime (p : ℤ) : Prop := Nat.Prime p.natAbs

theorem unique_solution_for_P_squared_prime :
  ∃! n : ℤ, ∃ p : ℤ, isPrime p ∧ (P n)^2 = p^2 ∧ n = -3 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_for_P_squared_prime_l4131_413195


namespace NUMINAMATH_CALUDE_blueberry_lake_fish_count_l4131_413156

/-- The number of fish associated with each white duck -/
def white_duck_fish : ℕ := 8

/-- The number of fish associated with each black duck -/
def black_duck_fish : ℕ := 15

/-- The number of fish associated with each multicolor duck -/
def multicolor_duck_fish : ℕ := 20

/-- The number of fish associated with each golden duck -/
def golden_duck_fish : ℕ := 25

/-- The number of fish associated with each teal duck -/
def teal_duck_fish : ℕ := 30

/-- The number of white ducks in Blueberry Lake -/
def white_ducks : ℕ := 10

/-- The number of black ducks in Blueberry Lake -/
def black_ducks : ℕ := 12

/-- The number of multicolor ducks in Blueberry Lake -/
def multicolor_ducks : ℕ := 8

/-- The number of golden ducks in Blueberry Lake -/
def golden_ducks : ℕ := 6

/-- The number of teal ducks in Blueberry Lake -/
def teal_ducks : ℕ := 14

/-- The total number of fish in Blueberry Lake -/
def total_fish : ℕ := white_duck_fish * white_ducks + 
                      black_duck_fish * black_ducks + 
                      multicolor_duck_fish * multicolor_ducks + 
                      golden_duck_fish * golden_ducks + 
                      teal_duck_fish * teal_ducks

theorem blueberry_lake_fish_count : total_fish = 990 := by
  sorry

end NUMINAMATH_CALUDE_blueberry_lake_fish_count_l4131_413156


namespace NUMINAMATH_CALUDE_probability_sum_three_two_dice_l4131_413166

theorem probability_sum_three_two_dice : 
  let total_outcomes : ℕ := 6 * 6
  let favorable_outcomes : ℕ := 2
  favorable_outcomes / total_outcomes = (1 : ℚ) / 18 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_three_two_dice_l4131_413166


namespace NUMINAMATH_CALUDE_melissa_pencils_count_l4131_413134

/-- The number of pencils Melissa wants to buy -/
def melissa_pencils : ℕ := 2

/-- The price of one pencil in cents -/
def pencil_price : ℕ := 20

/-- The number of pencils Tolu wants to buy -/
def tolu_pencils : ℕ := 3

/-- The number of pencils Robert wants to buy -/
def robert_pencils : ℕ := 5

/-- The total amount spent by all students in cents -/
def total_spent : ℕ := 200

theorem melissa_pencils_count :
  melissa_pencils * pencil_price + tolu_pencils * pencil_price + robert_pencils * pencil_price = total_spent :=
by sorry

end NUMINAMATH_CALUDE_melissa_pencils_count_l4131_413134


namespace NUMINAMATH_CALUDE_double_length_isosceles_triangle_base_length_l4131_413198

/-- A triangle with one side length being twice the length of another side is called a "double-length triangle". -/
def is_double_length_triangle (a b c : ℝ) : Prop :=
  a = 2*b ∨ a = 2*c ∨ b = 2*a ∨ b = 2*c ∨ c = 2*a ∨ c = 2*b

/-- An isosceles triangle is a triangle with at least two equal sides. -/
def is_isosceles_triangle (a b c : ℝ) : Prop :=
  a = b ∨ a = c ∨ b = c

theorem double_length_isosceles_triangle_base_length 
  (a b c : ℝ) 
  (h_isosceles : is_isosceles_triangle a b c) 
  (h_double_length : is_double_length_triangle a b c) 
  (h_side_length : a = 6) : 
  (a = b ∧ a = 2*c ∧ c = 3) ∨ (a = c ∧ a = 2*b ∧ b = 3) :=
sorry

end NUMINAMATH_CALUDE_double_length_isosceles_triangle_base_length_l4131_413198


namespace NUMINAMATH_CALUDE_simplify_cube_roots_l4131_413168

theorem simplify_cube_roots : 
  (1 + 27) ^ (1/3) * (1 + 27 ^ (1/3)) ^ (1/3) = 28 ^ (1/3) * 4 ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_simplify_cube_roots_l4131_413168


namespace NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l4131_413118

theorem largest_n_satisfying_inequality :
  ∀ n : ℕ, (1/4 : ℚ) + (n : ℚ)/8 < 2 ↔ n ≤ 13 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l4131_413118


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_8_and_6_l4131_413176

theorem smallest_common_multiple_of_8_and_6 : 
  ∃ n : ℕ, n > 0 ∧ 8 ∣ n ∧ 6 ∣ n ∧ ∀ m : ℕ, m > 0 → 8 ∣ m → 6 ∣ m → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_8_and_6_l4131_413176


namespace NUMINAMATH_CALUDE_is_vertex_of_parabola_l4131_413197

/-- The parabola equation -/
def parabola_equation (x : ℝ) : ℝ := -2 * x^2 - 20 * x - 50

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (-5, 0)

/-- Theorem stating that the given point is the vertex of the parabola -/
theorem is_vertex_of_parabola :
  let (m, n) := vertex
  ∀ x : ℝ, parabola_equation x ≤ parabola_equation m :=
by sorry

end NUMINAMATH_CALUDE_is_vertex_of_parabola_l4131_413197


namespace NUMINAMATH_CALUDE_least_number_with_divisibility_conditions_l4131_413132

theorem least_number_with_divisibility_conditions : ∃ n : ℕ, 
  (∀ k : ℕ, 2 ≤ k → k ≤ 7 → n % k = 1) ∧ 
  (n % 8 = 0) ∧
  (∀ m : ℕ, m < n → ¬((∀ k : ℕ, 2 ≤ k → k ≤ 7 → m % k = 1) ∧ (m % 8 = 0))) ∧
  n = 1681 := by
sorry

end NUMINAMATH_CALUDE_least_number_with_divisibility_conditions_l4131_413132


namespace NUMINAMATH_CALUDE_cubic_function_properties_l4131_413138

-- Define the function f
def f (m n x : ℝ) : ℝ := m * x^3 + n * x^2

-- Define the derivative of f
def f' (m n x : ℝ) : ℝ := 3 * m * x^2 + 2 * n * x

-- Theorem statement
theorem cubic_function_properties (m : ℝ) (h : m ≠ 0) :
  ∃ n : ℝ,
    f' m n 2 = 0 ∧
    n = -3 * m ∧
    (∀ x : ℝ, m > 0 → (x < 0 ∨ x > 2) → (f' m n x > 0)) ∧
    (∀ x : ℝ, m < 0 → (x > 0 ∧ x < 2) → (f' m n x > 0)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l4131_413138


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l4131_413150

theorem polynomial_division_theorem (x : ℝ) : 
  ∃ (q r : ℝ → ℝ), 
    (x^4 - 3*x^2 + 1 = (x^2 - x + 1) * q x + r x) ∧ 
    (∀ y, r y = -3*y^2 + y + 1) ∧
    (∀ z, z^2 - z + 1 = 0 → r z = 0) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l4131_413150


namespace NUMINAMATH_CALUDE_difference_divisible_by_99_l4131_413140

/-- Represents a three-digit number with digits a, b, and c -/
structure ThreeDigitNumber where
  a : ℕ
  b : ℕ
  c : ℕ
  h1 : a ≥ 1 ∧ a ≤ 9
  h2 : b ≥ 0 ∧ b ≤ 9
  h3 : c ≥ 0 ∧ c ≤ 9

/-- The value of a three-digit number -/
def value (n : ThreeDigitNumber) : ℕ :=
  100 * n.a + 10 * n.b + n.c

/-- The value of a three-digit number with hundreds and units digits exchanged -/
def exchangedValue (n : ThreeDigitNumber) : ℕ :=
  100 * n.c + 10 * n.b + n.a

/-- The difference between the exchanged value and the original value -/
def difference (n : ThreeDigitNumber) : ℤ :=
  (exchangedValue n : ℤ) - (value n : ℤ)

/-- Theorem stating that the difference is always divisible by 99 -/
theorem difference_divisible_by_99 (n : ThreeDigitNumber) :
  ∃ k : ℤ, difference n = 99 * k := by
  sorry


end NUMINAMATH_CALUDE_difference_divisible_by_99_l4131_413140


namespace NUMINAMATH_CALUDE_digital_earth_storage_technologies_l4131_413192

-- Define the set of all possible technologies
inductive Technology
| Nano
| LaserHolographic
| Protein
| Distributed
| Virtual
| Spatial
| Visualization

-- Define the property of contributing to digital Earth data storage
def contributesToDigitalEarthStorage (tech : Technology) : Prop :=
  match tech with
  | Technology.Nano => true
  | Technology.LaserHolographic => true
  | Technology.Protein => true
  | Technology.Distributed => true
  | _ => false

-- Define the set of technologies that contribute to digital Earth storage
def contributingTechnologies : Set Technology :=
  {tech | contributesToDigitalEarthStorage tech}

-- Theorem statement
theorem digital_earth_storage_technologies :
  contributingTechnologies = {Technology.Nano, Technology.LaserHolographic, Technology.Protein, Technology.Distributed} :=
by sorry

end NUMINAMATH_CALUDE_digital_earth_storage_technologies_l4131_413192


namespace NUMINAMATH_CALUDE_group_size_calculation_l4131_413106

/-- Given a group of people where:
  1. The average weight increase is 1.5 kg
  2. The total weight increase is 12 kg (77 kg - 65 kg)
  3. The total weight increase equals the average weight increase multiplied by the number of people
  Prove that the number of people in the group is 8. -/
theorem group_size_calculation (avg_increase : ℝ) (total_increase : ℝ) :
  avg_increase = 1.5 →
  total_increase = 12 →
  total_increase = avg_increase * 8 :=
by sorry

end NUMINAMATH_CALUDE_group_size_calculation_l4131_413106


namespace NUMINAMATH_CALUDE_f_geq_a_iff_a_in_range_l4131_413120

/-- The function f(x) defined as x^2 - 2ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 2

/-- The theorem stating the equivalence between the condition and the range of a -/
theorem f_geq_a_iff_a_in_range (a : ℝ) :
  (∀ x ≥ -1, f a x ≥ a) ↔ a ∈ Set.Icc (-3) 1 :=
by sorry

#check f_geq_a_iff_a_in_range

end NUMINAMATH_CALUDE_f_geq_a_iff_a_in_range_l4131_413120


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l4131_413193

def A : Set ℤ := {0, 1, 2}

def U : Set ℤ := {z | ∃ x y, x ∈ A ∧ y ∈ A ∧ z = x - y}

theorem complement_of_A_in_U :
  (A : Set ℤ)ᶜ ∩ U = {-2, -1} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l4131_413193


namespace NUMINAMATH_CALUDE_isabellas_total_items_l4131_413171

/-- Given that Alexis bought 3 times more pants and dresses than Isabella,
    prove that Isabella bought 13 items in total. -/
theorem isabellas_total_items
  (alexis_pants : ℕ)
  (alexis_dresses : ℕ)
  (h1 : alexis_pants = 21)
  (h2 : alexis_dresses = 18)
  (h3 : ∃ (k : ℕ), k > 0 ∧ alexis_pants = 3 * k ∧ alexis_dresses = 3 * (alexis_dresses / 3)) :
  alexis_pants / 3 + alexis_dresses / 3 = 13 :=
by sorry

end NUMINAMATH_CALUDE_isabellas_total_items_l4131_413171


namespace NUMINAMATH_CALUDE_taller_tree_height_l4131_413109

/-- Given two trees where one is 20 feet taller than the other and their heights
    are in the ratio 2:3, prove that the height of the taller tree is 60 feet. -/
theorem taller_tree_height (h : ℝ) (h_pos : h > 0) : 
  (h - 20) / h = 2 / 3 → h = 60 := by
  sorry

end NUMINAMATH_CALUDE_taller_tree_height_l4131_413109


namespace NUMINAMATH_CALUDE_sum_odd_numbers_l4131_413165

/-- Sum of first n natural numbers -/
def sum_n (n : ℕ) : ℕ := n * (n + 1)

/-- Sum of first n odd numbers -/
def sum_odd (n : ℕ) : ℕ := n^2

/-- The 35th odd number -/
def last_odd : ℕ := 69

/-- Number of odd numbers up to 69 -/
def num_odds : ℕ := (last_odd + 1) / 2

theorem sum_odd_numbers :
  3 * (sum_odd num_odds) = 3675 :=
sorry

end NUMINAMATH_CALUDE_sum_odd_numbers_l4131_413165


namespace NUMINAMATH_CALUDE_clothing_distribution_l4131_413136

def total_clothing : ℕ := 135
def first_load : ℕ := 29
def num_small_loads : ℕ := 7

theorem clothing_distribution :
  (total_clothing - first_load) / num_small_loads = 15 :=
by sorry

end NUMINAMATH_CALUDE_clothing_distribution_l4131_413136


namespace NUMINAMATH_CALUDE_right_triangle_log_identity_l4131_413151

theorem right_triangle_log_identity 
  (a b c : ℝ) 
  (h_right_triangle : a^2 + b^2 = c^2) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle_inequality : c > b) :
  Real.log a / Real.log (b + c) + Real.log a / Real.log (c - b) = 
  2 * (Real.log a / Real.log (c + b)) * (Real.log a / Real.log (c - b)) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_log_identity_l4131_413151


namespace NUMINAMATH_CALUDE_extreme_values_cubic_l4131_413186

/-- Given a cubic function with extreme values at x=1 and x=2, prove that b=4 -/
theorem extreme_values_cubic (a b : ℝ) : 
  let f := fun x : ℝ => 2 * x^3 + 3 * a * x^2 + 3 * b * x
  let f' := fun x : ℝ => 6 * x^2 + 6 * a * x + 3 * b
  (f' 1 = 0 ∧ f' 2 = 0) → b = 4 := by
sorry

end NUMINAMATH_CALUDE_extreme_values_cubic_l4131_413186


namespace NUMINAMATH_CALUDE_interest_rate_is_six_percent_l4131_413147

/-- Calculates the simple interest rate given the principal, final amount, and time period. -/
def simple_interest_rate (principal : ℚ) (final_amount : ℚ) (time : ℚ) : ℚ :=
  ((final_amount - principal) * 100) / (principal * time)

/-- Theorem stating that for the given conditions, the simple interest rate is 6% -/
theorem interest_rate_is_six_percent :
  simple_interest_rate 12500 15500 4 = 6 := by
  sorry

#eval simple_interest_rate 12500 15500 4

end NUMINAMATH_CALUDE_interest_rate_is_six_percent_l4131_413147


namespace NUMINAMATH_CALUDE_proportion_theorem_l4131_413194

theorem proportion_theorem (y : ℝ) : 
  (0.75 : ℝ) / 0.9 = y / 6 → y = 5 := by
  sorry

end NUMINAMATH_CALUDE_proportion_theorem_l4131_413194


namespace NUMINAMATH_CALUDE_product_price_l4131_413173

/-- Given that m kilograms of a product costs 9 yuan, 
    prove that n kilograms of the same product costs (9n/m) yuan. -/
theorem product_price (m n : ℝ) (hm : m > 0) : 
  (9 : ℝ) / m * n = 9 * n / m := by sorry

end NUMINAMATH_CALUDE_product_price_l4131_413173


namespace NUMINAMATH_CALUDE_sum_of_fractions_inequality_l4131_413142

theorem sum_of_fractions_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (b + c)) + (b / (c + a)) + (c / (a + b)) ≥ (3 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_inequality_l4131_413142


namespace NUMINAMATH_CALUDE_function_divisibility_condition_l4131_413190

theorem function_divisibility_condition (f : ℕ+ → ℕ+) :
  (∀ n m : ℕ+, (n + f m) ∣ (f n + n * f m)) →
  (∀ n : ℕ+, f n = n ^ 2 ∨ f n = 1) :=
by sorry

end NUMINAMATH_CALUDE_function_divisibility_condition_l4131_413190


namespace NUMINAMATH_CALUDE_product_mod_five_l4131_413101

theorem product_mod_five : (2023 * 2024 * 2025 * 2026) % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_five_l4131_413101


namespace NUMINAMATH_CALUDE_stock_price_loss_l4131_413152

theorem stock_price_loss (n : ℕ) (P : ℝ) (h : P > 0) : 
  P * (1.1 ^ n) * (0.9 ^ n) < P := by
  sorry

#check stock_price_loss

end NUMINAMATH_CALUDE_stock_price_loss_l4131_413152


namespace NUMINAMATH_CALUDE_number_division_problem_l4131_413124

theorem number_division_problem (n : ℕ) : 
  n % 37 = 26 ∧ n / 37 = 2 → 48 - n / 4 = 23 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l4131_413124


namespace NUMINAMATH_CALUDE_reciprocal_inequality_l4131_413170

theorem reciprocal_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 1 / a > 1 / b := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_inequality_l4131_413170


namespace NUMINAMATH_CALUDE_base_2_digit_difference_l4131_413122

theorem base_2_digit_difference : ∀ (n m : ℕ), n = 300 → m = 1500 → 
  (Nat.log 2 m + 1) - (Nat.log 2 n + 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_base_2_digit_difference_l4131_413122


namespace NUMINAMATH_CALUDE_tommy_order_cost_and_percentages_l4131_413154

/-- Represents the weight of each fruit in kilograms -/
structure FruitOrder where
  apples : ℝ
  oranges : ℝ
  grapes : ℝ
  strawberries : ℝ
  bananas : ℝ
  pineapples : ℝ

/-- Represents the price of each fruit per kilogram -/
structure FruitPrices where
  apples : ℝ
  oranges : ℝ
  grapes : ℝ
  strawberries : ℝ
  bananas : ℝ
  pineapples : ℝ

def totalWeight (order : FruitOrder) : ℝ :=
  order.apples + order.oranges + order.grapes + order.strawberries + order.bananas + order.pineapples

def totalCost (order : FruitOrder) (prices : FruitPrices) : ℝ :=
  order.apples * prices.apples +
  order.oranges * prices.oranges +
  order.grapes * prices.grapes +
  order.strawberries * prices.strawberries +
  order.bananas * prices.bananas +
  order.pineapples * prices.pineapples

theorem tommy_order_cost_and_percentages 
  (order : FruitOrder)
  (prices : FruitPrices)
  (h1 : totalWeight order = 20)
  (h2 : order.apples = 4)
  (h3 : order.oranges = 2)
  (h4 : order.grapes = 4)
  (h5 : order.strawberries = 3)
  (h6 : order.bananas = 1)
  (h7 : order.pineapples = 3)
  (h8 : prices.apples = 2)
  (h9 : prices.oranges = 3)
  (h10 : prices.grapes = 2.5)
  (h11 : prices.strawberries = 4)
  (h12 : prices.bananas = 1.5)
  (h13 : prices.pineapples = 3.5) :
  totalCost order prices = 48 ∧
  order.apples / totalWeight order = 0.2 ∧
  order.oranges / totalWeight order = 0.1 ∧
  order.grapes / totalWeight order = 0.2 ∧
  order.strawberries / totalWeight order = 0.15 ∧
  order.bananas / totalWeight order = 0.05 ∧
  order.pineapples / totalWeight order = 0.15 := by
  sorry


end NUMINAMATH_CALUDE_tommy_order_cost_and_percentages_l4131_413154


namespace NUMINAMATH_CALUDE_profit_of_c_l4131_413108

def total_profit : ℕ := 56700
def ratio_a : ℕ := 8
def ratio_b : ℕ := 9
def ratio_c : ℕ := 10

theorem profit_of_c :
  let total_ratio := ratio_a + ratio_b + ratio_c
  let part_value := total_profit / total_ratio
  part_value * ratio_c = 21000 := by sorry

end NUMINAMATH_CALUDE_profit_of_c_l4131_413108


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l4131_413143

theorem min_value_reciprocal_sum (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x * y * z = 1) :
  1 / (x + y) + 1 / (y + z) + 1 / (z + x) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l4131_413143


namespace NUMINAMATH_CALUDE_min_bushes_cover_alley_l4131_413175

/-- The length of the alley in meters -/
def alley_length : ℝ := 400

/-- The radius of scent spread for each lily of the valley bush in meters -/
def scent_radius : ℝ := 20

/-- The minimum number of bushes needed to cover the alley with scent -/
def min_bushes : ℕ := 10

/-- Theorem stating that the minimum number of bushes needed to cover the alley is correct -/
theorem min_bushes_cover_alley :
  ∀ (n : ℕ), n ≥ min_bushes → n * (2 * scent_radius) ≥ alley_length :=
by sorry

end NUMINAMATH_CALUDE_min_bushes_cover_alley_l4131_413175


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_k_equals_five_l4131_413107

/-- Given vectors in ℝ², prove that if (a - c) is parallel to b, then k = 5 --/
theorem parallel_vectors_imply_k_equals_five (a b c : ℝ × ℝ) (k : ℝ) :
  a = (3, 1) →
  b = (1, 3) →
  c = (k, 7) →
  ∃ (t : ℝ), (a.1 - c.1, a.2 - c.2) = (t * b.1, t * b.2) →
  k = 5 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_k_equals_five_l4131_413107


namespace NUMINAMATH_CALUDE_square_difference_equals_150_l4131_413127

theorem square_difference_equals_150 : (15 + 5)^2 - (5^2 + 15^2) = 150 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equals_150_l4131_413127


namespace NUMINAMATH_CALUDE_sunday_letters_zero_l4131_413149

/-- Represents the number of letters written on each day of the week -/
structure WeeklyLetters where
  sunday : ℕ
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ

/-- The average number of letters written per day -/
def averageLettersPerDay : ℕ := 9

/-- The total number of days in a week -/
def daysInWeek : ℕ := 7

/-- Calculates the total number of letters written in a week -/
def totalLetters (w : WeeklyLetters) : ℕ :=
  w.sunday + w.monday + w.tuesday + w.wednesday + w.thursday + w.friday + w.saturday

/-- States that the total number of letters written in a week equals the average per day times the number of days -/
axiom total_letters_axiom (w : WeeklyLetters) :
  totalLetters w = averageLettersPerDay * daysInWeek

/-- Defines the known number of letters written on specific days -/
def knownLetters (w : WeeklyLetters) : Prop :=
  w.wednesday ≥ 13 ∧ w.thursday ≥ 12 ∧ w.friday ≥ 9 ∧ w.saturday ≥ 7

/-- Theorem stating that given the conditions, the number of letters written on Sunday must be zero -/
theorem sunday_letters_zero (w : WeeklyLetters) 
  (h : knownLetters w) : w.sunday = 0 := by
  sorry


end NUMINAMATH_CALUDE_sunday_letters_zero_l4131_413149


namespace NUMINAMATH_CALUDE_X_equals_Y_l4131_413137

def X : Set ℝ := {x | ∃ n : ℤ, x = (2 * n + 1) * Real.pi}
def Y : Set ℝ := {y | ∃ k : ℤ, y = (4 * k + 1) * Real.pi ∨ y = (4 * k - 1) * Real.pi}

theorem X_equals_Y : X = Y := by sorry

end NUMINAMATH_CALUDE_X_equals_Y_l4131_413137


namespace NUMINAMATH_CALUDE_inverse_of_B_cubed_l4131_413162

def B_inv : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; -2, -3]

theorem inverse_of_B_cubed (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B⁻¹ = B_inv) : 
  (B^3)⁻¹ = B⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_B_cubed_l4131_413162


namespace NUMINAMATH_CALUDE_prob_rolling_doubles_l4131_413196

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The number of dice being rolled -/
def numDice : ℕ := 3

/-- The number of favorable outcomes (rolling the same number on all dice) -/
def favorableOutcomes : ℕ := numSides

/-- The total number of possible outcomes when rolling the dice -/
def totalOutcomes : ℕ := numSides ^ numDice

/-- The probability of rolling doubles with three six-sided dice -/
theorem prob_rolling_doubles : 
  (favorableOutcomes : ℚ) / totalOutcomes = 1 / 36 := by sorry

end NUMINAMATH_CALUDE_prob_rolling_doubles_l4131_413196
