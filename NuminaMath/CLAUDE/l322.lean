import Mathlib

namespace NUMINAMATH_CALUDE_square_perimeter_sum_l322_32227

theorem square_perimeter_sum (x y : ℝ) (h1 : x^2 + y^2 = 85) (h2 : x^2 - y^2 = 45) :
  4*x + 4*y = 4*Real.sqrt 65 + 8*Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_sum_l322_32227


namespace NUMINAMATH_CALUDE_system_solution_l322_32274

theorem system_solution :
  let f (x y : ℝ) := 7 * x^2 + 7 * y^2 - 3 * x^2 * y^2
  let g (x y : ℝ) := x^4 + y^4 - x^2 * y^2
  ∀ x y : ℝ, (f x y = 7 ∧ g x y = 37) ↔
    ((x = Real.sqrt 7 ∧ (y = Real.sqrt 3 ∨ y = -Real.sqrt 3)) ∨
     (x = -Real.sqrt 7 ∧ (y = Real.sqrt 3 ∨ y = -Real.sqrt 3)) ∨
     (x = Real.sqrt 3 ∧ (y = Real.sqrt 7 ∨ y = -Real.sqrt 7)) ∨
     (x = -Real.sqrt 3 ∧ (y = Real.sqrt 7 ∨ y = -Real.sqrt 7))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l322_32274


namespace NUMINAMATH_CALUDE_intersection_implies_a_leq_neg_one_l322_32254

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (x^2 - 2*x - 3)}
def B (a : ℝ) : Set ℝ := {x | ∃ y, y = Real.sqrt (a - x)}

-- State the theorem
theorem intersection_implies_a_leq_neg_one (a : ℝ) : A ∩ B a = B a → a ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_leq_neg_one_l322_32254


namespace NUMINAMATH_CALUDE_quadratic_root_on_line_l322_32271

/-- A root of a quadratic equation lies on a corresponding line in the p-q plane. -/
theorem quadratic_root_on_line (p q x₀ : ℝ) : 
  x₀^2 + p * x₀ + q = 0 → q = -x₀ * p - x₀^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_on_line_l322_32271


namespace NUMINAMATH_CALUDE_tournament_committee_count_l322_32235

/-- The number of teams in the league -/
def num_teams : ℕ := 4

/-- The number of members in each team -/
def team_size : ℕ := 8

/-- The number of members selected from the winning team -/
def winning_team_selection : ℕ := 3

/-- The number of members selected from each non-winning team -/
def other_team_selection : ℕ := 2

/-- The total number of members in the tournament committee -/
def committee_size : ℕ := 9

/-- The number of possible tournament committees -/
def num_committees : ℕ := 4917248

theorem tournament_committee_count :
  num_committees = 
    num_teams * (Nat.choose team_size winning_team_selection) * 
    (Nat.choose team_size other_team_selection) ^ (num_teams - 1) := by
  sorry

end NUMINAMATH_CALUDE_tournament_committee_count_l322_32235


namespace NUMINAMATH_CALUDE_students_just_passed_l322_32216

theorem students_just_passed (total : ℕ) (first_div_percent : ℚ) (second_div_percent : ℚ) :
  total = 300 →
  first_div_percent = 29 / 100 →
  second_div_percent = 54 / 100 →
  (total : ℚ) * (1 - first_div_percent - second_div_percent) = 51 := by
sorry

end NUMINAMATH_CALUDE_students_just_passed_l322_32216


namespace NUMINAMATH_CALUDE_derivative_at_one_l322_32279

theorem derivative_at_one (f : ℝ → ℝ) (f' : ℝ → ℝ) :
  (∀ x, f x = x^2 + 3*x*(f' 1)) →
  (∀ x, HasDerivAt f (f' x) x) →
  f' 1 = -1 := by
sorry

end NUMINAMATH_CALUDE_derivative_at_one_l322_32279


namespace NUMINAMATH_CALUDE_sum_of_products_l322_32253

theorem sum_of_products (a₁ a₂ a₃ d₁ d₂ d₃ : ℝ) 
  (h : ∀ x : ℝ, x^6 + 2*x^5 + x^4 + x^3 + x^2 + 2*x + 1 = 
    (x^2 + a₁*x + d₁) * (x^2 + a₂*x + d₂) * (x^2 + a₃*x + d₃)) : 
  a₁*d₁ + a₂*d₂ + a₃*d₃ = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_products_l322_32253


namespace NUMINAMATH_CALUDE_function_inequality_l322_32289

open Set
open Real

theorem function_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ) (h1 : ∀ x, HasDerivAt f (f' x) x) 
  (h2 : ∀ x, f x + f' x > 2) (h3 : f 0 = 2021) :
  ∀ x, f x > 2 + 2019 / exp x ↔ x > 0 := by
sorry

end NUMINAMATH_CALUDE_function_inequality_l322_32289


namespace NUMINAMATH_CALUDE_f_properties_l322_32223

noncomputable section

def e : ℝ := Real.exp 1

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < e then -x^3 + x^2 else a * Real.log x

theorem f_properties (a : ℝ) (h : a ≠ 0) :
  (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧ ∀ z : ℝ, f a z = 0 → z = x ∨ z = y) ∧
  (∃ M N : ℝ × ℝ,
    let (xM, yM) := M
    let (xN, yN) := N
    xM > 0 ∧ xN < 0 ∧
    yM = f a xM ∧ yN = f a (-xN) ∧
    xM * xN + yM * yN = 0 ∧
    (xM - xN) * yM = xM * (yM - yN) ∧
    0 < a ∧ a ≤ 1 / (e + 1)) ∧
  (∀ a' : ℝ, a' ≤ 0 ∨ a' > 1 / (e + 1) →
    ¬∃ M N : ℝ × ℝ,
      let (xM, yM) := M
      let (xN, yN) := N
      xM > 0 ∧ xN < 0 ∧
      yM = f a' xM ∧ yN = f a' (-xN) ∧
      xM * xN + yM * yN = 0 ∧
      (xM - xN) * yM = xM * (yM - yN)) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l322_32223


namespace NUMINAMATH_CALUDE_factor_expression_l322_32242

theorem factor_expression (a : ℝ) : 45 * a^2 + 135 * a + 90 = 45 * a * (a + 5) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l322_32242


namespace NUMINAMATH_CALUDE_line_sum_m_b_l322_32268

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) with slope m and y-intercept b -/
structure Line where
  x₁ : ℚ
  y₁ : ℚ
  x₂ : ℚ
  y₂ : ℚ
  m : ℚ
  b : ℚ
  eq₁ : y₁ = m * x₁ + b
  eq₂ : y₂ = m * x₂ + b

/-- Theorem: For a line passing through (2, -1) and (5, 3), m + b = -7/3 -/
theorem line_sum_m_b :
  ∀ l : Line,
    l.x₁ = 2 ∧ l.y₁ = -1 ∧ l.x₂ = 5 ∧ l.y₂ = 3 →
    l.m + l.b = -7/3 := by
  sorry

end NUMINAMATH_CALUDE_line_sum_m_b_l322_32268


namespace NUMINAMATH_CALUDE_total_books_calculation_l322_32269

theorem total_books_calculation (darryl_books : ℕ) (lamont_books : ℕ) (loris_books : ℕ) (danielle_books : ℕ) : 
  darryl_books = 20 →
  lamont_books = 2 * darryl_books →
  loris_books + 3 = lamont_books →
  danielle_books = lamont_books + darryl_books + 10 →
  darryl_books + lamont_books + loris_books + danielle_books = 167 := by
sorry

end NUMINAMATH_CALUDE_total_books_calculation_l322_32269


namespace NUMINAMATH_CALUDE_quadratic_factorization_l322_32290

theorem quadratic_factorization (x : ℝ) : x^2 - 7*x + 10 = (x - 2) * (x - 5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l322_32290


namespace NUMINAMATH_CALUDE_science_homework_duration_l322_32282

/-- Calculates the time remaining for science homework given the total time and time spent on other subjects. -/
def science_homework_time (total_time math_time english_time history_time project_time : ℕ) : ℕ :=
  total_time - (math_time + english_time + history_time + project_time)

/-- Proves that given the specified times for total work and other subjects, the remaining time for science homework is 50 minutes. -/
theorem science_homework_duration :
  science_homework_time 180 45 30 25 30 = 50 := by
  sorry

end NUMINAMATH_CALUDE_science_homework_duration_l322_32282


namespace NUMINAMATH_CALUDE_rectangle_side_relationship_l322_32230

/-- Represents a rectangle with sides x and y -/
structure Rectangle where
  x : ℝ
  y : ℝ

/-- The perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.x + r.y)

/-- Theorem: For a rectangle with perimeter 50 cm, y = 25 - x -/
theorem rectangle_side_relationship (r : Rectangle) 
  (h : perimeter r = 50) : r.y = 25 - r.x := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_relationship_l322_32230


namespace NUMINAMATH_CALUDE_quadratic_set_equality_l322_32213

theorem quadratic_set_equality (p : ℝ) : 
  ({x : ℝ | x^2 - 5*x + p ≥ 0} = {x : ℝ | x ≤ -1 ∨ x ≥ 6}) → p = -6 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_set_equality_l322_32213


namespace NUMINAMATH_CALUDE_pure_imaginary_product_l322_32263

theorem pure_imaginary_product (b : ℝ) : 
  let z₁ : ℂ := 1 + Complex.I
  let z₂ : ℂ := 1 - b * Complex.I
  (z₁ * z₂).re = 0 ∧ (z₁ * z₂).im ≠ 0 → b = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_l322_32263


namespace NUMINAMATH_CALUDE_solution_set_equality_l322_32244

-- Define the set S
def S : Set ℝ := {x | |x + 2| + |x - 1| ≤ 4}

-- State the theorem
theorem solution_set_equality : S = Set.Icc (-5/2) (3/2) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equality_l322_32244


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l322_32296

/-- A perfect square trinomial in the form x^2 + ax + 4 -/
def is_perfect_square_trinomial (a : ℝ) : Prop :=
  ∃ b : ℝ, ∀ x : ℝ, x^2 + a*x + 4 = (x + b)^2

/-- If x^2 + ax + 4 is a perfect square trinomial, then a = ±4 -/
theorem perfect_square_trinomial_condition (a : ℝ) :
  is_perfect_square_trinomial a → a = 4 ∨ a = -4 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l322_32296


namespace NUMINAMATH_CALUDE_floor_of_e_l322_32260

theorem floor_of_e : ⌊Real.exp 1⌋ = 2 := by sorry

end NUMINAMATH_CALUDE_floor_of_e_l322_32260


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l322_32284

/-- An arithmetic sequence with common ratio q ≠ 1 -/
def ArithmeticSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q ≠ 1 ∧ ∀ n : ℕ, a (n + 1) - a n = q * (a n - a (n - 1))

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  ArithmeticSequence a q →
  (a 1 + a 2 + a 3 + a 4 + a 5 = 6) →
  (a 1^2 + a 2^2 + a 3^2 + a 4^2 + a 5^2 = 18) →
  a 1 - a 2 + a 3 - a 4 + a 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l322_32284


namespace NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l322_32218

theorem product_of_difference_and_sum_of_squares (a b : ℝ) 
  (h1 : a - b = 3) 
  (h2 : a^2 + b^2 = 25) : 
  a * b = 8 := by
sorry

end NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l322_32218


namespace NUMINAMATH_CALUDE_pet_store_inventory_l322_32243

/-- Represents the number of birds of each type in a cage -/
structure BirdCage where
  parrots : ℕ
  parakeets : ℕ
  canaries : ℕ
  cockatiels : ℕ
  lovebirds : ℕ
  finches : ℕ

/-- The pet store inventory -/
def petStore : List BirdCage :=
  (List.replicate 7 ⟨3, 5, 4, 0, 0, 0⟩) ++
  (List.replicate 6 ⟨0, 0, 0, 2, 3, 1⟩) ++
  (List.replicate 2 ⟨0, 0, 0, 0, 0, 0⟩)

/-- Calculate the total number of birds of each type -/
def totalBirds (store : List BirdCage) : BirdCage :=
  store.foldl (fun acc cage =>
    ⟨acc.parrots + cage.parrots,
     acc.parakeets + cage.parakeets,
     acc.canaries + cage.canaries,
     acc.cockatiels + cage.cockatiels,
     acc.lovebirds + cage.lovebirds,
     acc.finches + cage.finches⟩)
    ⟨0, 0, 0, 0, 0, 0⟩

theorem pet_store_inventory :
  totalBirds petStore = ⟨21, 35, 28, 12, 18, 6⟩ := by
  sorry

end NUMINAMATH_CALUDE_pet_store_inventory_l322_32243


namespace NUMINAMATH_CALUDE_vector_equality_transitivity_l322_32228

variable {V : Type*} [AddCommGroup V]

theorem vector_equality_transitivity (a b c : V) :
  a = b → b = c → a = c := by
  sorry

end NUMINAMATH_CALUDE_vector_equality_transitivity_l322_32228


namespace NUMINAMATH_CALUDE_area_of_triangle_ABC_l322_32233

-- Define the points
variable (A B C D : ℝ × ℝ)

-- Define the distances
def AC : ℝ := 10
def AB : ℝ := 17
def DC : ℝ := 6

-- Define coplanarity
def coplanar (A B C D : ℝ × ℝ) : Prop := sorry

-- Define right angle
def right_angle (A B C : ℝ × ℝ) : Prop := sorry

-- Define area of a triangle
def triangle_area (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem area_of_triangle_ABC (h1 : coplanar A B C D) 
                             (h2 : right_angle A D C) :
  triangle_area A B C = 84 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_ABC_l322_32233


namespace NUMINAMATH_CALUDE_ticket_cost_proof_l322_32246

def adult_price : ℕ := 12
def child_price : ℕ := 10
def senior_price : ℕ := 8
def student_price : ℕ := 9

def num_parents : ℕ := 2
def num_grandparents : ℕ := 2
def num_sisters : ℕ := 3
def num_cousins : ℕ := 1
def num_uncle_aunt : ℕ := 2

def total_cost : ℕ :=
  num_parents * adult_price +
  num_grandparents * senior_price +
  num_sisters * child_price +
  num_cousins * student_price +
  num_uncle_aunt * adult_price

theorem ticket_cost_proof : total_cost = 103 := by
  sorry

end NUMINAMATH_CALUDE_ticket_cost_proof_l322_32246


namespace NUMINAMATH_CALUDE_derivative_of_f_l322_32292

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) / x

theorem derivative_of_f (x : ℝ) (h : x ≠ 0) :
  deriv f x = (-x * Real.sin x - Real.cos x) / (x^2) := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_f_l322_32292


namespace NUMINAMATH_CALUDE_ln_range_l322_32273

open Real

theorem ln_range (f : ℝ → ℝ) (x : ℝ) :
  (∀ y, f y = log y) →
  f (x - 1) < 1 →
  1 < x ∧ x < exp 1 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ln_range_l322_32273


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l322_32297

theorem quadratic_expression_value (x y : ℝ) 
  (h1 : 2 * x + y = 11) 
  (h2 : x + 2 * y = 13) : 
  10 * x^2 - 6 * x * y + y^2 = 530 := by
sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l322_32297


namespace NUMINAMATH_CALUDE_midpoint_calculation_l322_32202

/-- Given two points A and B in a 2D plane, prove that 3x - 5y = -13.5,
    where (x, y) is the midpoint of segment AB. -/
theorem midpoint_calculation (A B : ℝ × ℝ) (h : A = (20, 12) ∧ B = (-4, 3)) :
  let C : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  3 * C.1 - 5 * C.2 = -13.5 := by
sorry

end NUMINAMATH_CALUDE_midpoint_calculation_l322_32202


namespace NUMINAMATH_CALUDE_matrix_multiplication_l322_32224

def A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; 4, -2]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![5, -3; 2, 4]

theorem matrix_multiplication :
  A * B = !![17, -5; 16, -20] := by sorry

end NUMINAMATH_CALUDE_matrix_multiplication_l322_32224


namespace NUMINAMATH_CALUDE_reservoir_capacity_difference_l322_32231

/-- Represents the properties of a reservoir --/
structure Reservoir where
  current_level : ℝ
  normal_level : ℝ
  total_capacity : ℝ
  evaporation_rate : ℝ

/-- Theorem about the difference between total capacity and normal level after evaporation --/
theorem reservoir_capacity_difference (r : Reservoir) 
  (h1 : r.current_level = 14)
  (h2 : r.current_level = 2 * r.normal_level)
  (h3 : r.current_level = 0.7 * r.total_capacity)
  (h4 : r.evaporation_rate = 0.1) :
  r.total_capacity - (r.normal_level * (1 - r.evaporation_rate)) = 13.7 := by
  sorry

end NUMINAMATH_CALUDE_reservoir_capacity_difference_l322_32231


namespace NUMINAMATH_CALUDE_jen_triple_flips_l322_32206

/-- Represents the number of flips in a specific type of flip. -/
def flips_per_type (flip_type : String) : ℕ :=
  if flip_type = "double" then 2 else 3

/-- Represents the total number of flips performed by a gymnast. -/
def total_flips (completed_flips : ℕ) (flip_type : String) : ℕ :=
  completed_flips * flips_per_type flip_type

theorem jen_triple_flips (tyler_double_flips : ℕ) (h1 : tyler_double_flips = 12) :
  let tyler_total_flips := total_flips tyler_double_flips "double"
  let jen_total_flips := 2 * tyler_total_flips
  jen_total_flips / flips_per_type "triple" = 16 := by
  sorry

end NUMINAMATH_CALUDE_jen_triple_flips_l322_32206


namespace NUMINAMATH_CALUDE_congruence_solution_count_l322_32266

theorem congruence_solution_count :
  ∃! (x : ℕ), x > 0 ∧ x < 50 ∧ (x + 20) % 45 = 70 % 45 :=
by sorry

end NUMINAMATH_CALUDE_congruence_solution_count_l322_32266


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_ratio_l322_32239

theorem arithmetic_geometric_mean_ratio (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : (a + b) / 2 = 3 * Real.sqrt (a * b)) : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |a / b - 34| < ε :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_ratio_l322_32239


namespace NUMINAMATH_CALUDE_quadratic_point_ordering_l322_32208

/-- A quadratic function f(x) = (x+1)² + 1 -/
def f (x : ℝ) : ℝ := (x + 1)^2 + 1

/-- Point A on the graph of f -/
def A : ℝ × ℝ := (-3, f (-3))

/-- Point B on the graph of f -/
def B : ℝ × ℝ := (0, f 0)

/-- Point C on the graph of f -/
def C : ℝ × ℝ := (2, f 2)

theorem quadratic_point_ordering :
  B.2 < A.2 ∧ A.2 < C.2 := by sorry

end NUMINAMATH_CALUDE_quadratic_point_ordering_l322_32208


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l322_32275

theorem least_subtraction_for_divisibility (x : ℕ) : 
  (x = 26 ∧ (12702 - x) % 99 = 0) ∧ 
  ∀ y : ℕ, y < x → (12702 - y) % 99 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l322_32275


namespace NUMINAMATH_CALUDE_right_triangle_sets_l322_32240

theorem right_triangle_sets : ∃! (a b c : ℕ), 
  ((a = 3 ∧ b = 4 ∧ c = 5) ∨
   (a = 6 ∧ b = 8 ∧ c = 10) ∨
   (a = 5 ∧ b = 2 ∧ c = 5) ∨
   (a = 5 ∧ b = 12 ∧ c = 13)) ∧
  ¬(a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l322_32240


namespace NUMINAMATH_CALUDE_trig_identity_l322_32203

theorem trig_identity (α : Real) (h : Real.sin α + Real.cos α = 1/5) :
  (Real.sin α - Real.cos α)^2 = 49/25 ∧ Real.sin α^3 + Real.cos α^3 = 37/125 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l322_32203


namespace NUMINAMATH_CALUDE_roots_equation_l322_32226

theorem roots_equation (x y : ℝ) (h1 : x + y = 10) (h2 : (x - y) * (x + y) = 48) :
  x^2 - 10*x + 19.24 = 0 ∧ y^2 - 10*y + 19.24 = 0 := by
  sorry

end NUMINAMATH_CALUDE_roots_equation_l322_32226


namespace NUMINAMATH_CALUDE_equilateral_triangle_min_rotation_angle_l322_32285

/-- An equilateral triangle with rotational symmetry -/
class EquilateralTriangle :=
  (rotation_symmetry : Bool)
  (is_equilateral : Bool)

/-- The minimum rotation angle (in degrees) for a shape with rotational symmetry -/
def min_rotation_angle (shape : EquilateralTriangle) : ℝ :=
  sorry

/-- Theorem: The minimum rotation angle for an equilateral triangle with rotational symmetry is 120 degrees -/
theorem equilateral_triangle_min_rotation_angle (t : EquilateralTriangle)
  (h1 : t.rotation_symmetry = true)
  (h2 : t.is_equilateral = true) :
  min_rotation_angle t = 120 :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_min_rotation_angle_l322_32285


namespace NUMINAMATH_CALUDE_certain_number_proof_l322_32261

theorem certain_number_proof : ∃! x : ℕ, (x - 16) % 37 = 0 ∧ (x - 16) / 37 = 23 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l322_32261


namespace NUMINAMATH_CALUDE_max_vector_norm_l322_32211

theorem max_vector_norm (θ : ℝ) : 
  (‖(2 * Real.cos θ - Real.sqrt 3, 2 * Real.sin θ + 1)‖ : ℝ) ≤ 4 ∧ 
  ∃ θ₀ : ℝ, ‖(2 * Real.cos θ₀ - Real.sqrt 3, 2 * Real.sin θ₀ + 1)‖ = 4 :=
sorry

end NUMINAMATH_CALUDE_max_vector_norm_l322_32211


namespace NUMINAMATH_CALUDE_unique_circle_from_hexagon_vertices_l322_32298

/-- A regular hexagon in a 2D plane -/
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ
  is_regular : ∀ (i j : Fin 6), dist (vertices i) (vertices j) = dist (vertices 0) (vertices 1)

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The theorem stating that there's only one unique circle with diameter endpoints at hexagon vertices -/
theorem unique_circle_from_hexagon_vertices (P : RegularHexagon) : 
  ∃! (c : Circle), ∃ (i j : Fin 6), 
    i ≠ j ∧ 
    dist (P.vertices i) (P.vertices j) = 2 * c.radius ∧
    dist ((P.vertices i).1 + (P.vertices j).1, (P.vertices i).2 + (P.vertices j).2) c.center = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_circle_from_hexagon_vertices_l322_32298


namespace NUMINAMATH_CALUDE_no_solution_and_inequality_solution_l322_32201

theorem no_solution_and_inequality_solution :
  (∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 → (x + 1) / (x - 1) + 4 / (1 - x^2) ≠ 1) ∧
  (∀ x : ℝ, 2 * (x - 1) ≥ x + 1 ∧ x - 2 > (2 * x - 1) / 3 ↔ x > 5) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_and_inequality_solution_l322_32201


namespace NUMINAMATH_CALUDE_jims_remaining_distance_l322_32286

/-- Given a total journey distance and the distance already driven, 
    calculate the remaining distance to drive. -/
def remaining_distance (total : ℕ) (driven : ℕ) : ℕ :=
  total - driven

/-- Theorem: For Jim's journey of 1200 miles, having driven 923 miles, 
    the remaining distance is 277 miles. -/
theorem jims_remaining_distance : 
  remaining_distance 1200 923 = 277 := by
  sorry

end NUMINAMATH_CALUDE_jims_remaining_distance_l322_32286


namespace NUMINAMATH_CALUDE_range_of_2a_plus_c_l322_32221

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_inequality_ab : a < b + c
  triangle_inequality_bc : b < a + c
  triangle_inequality_ca : c < a + b

-- State the theorem
theorem range_of_2a_plus_c (t : Triangle) 
  (h1 : t.a^2 + t.c^2 - t.b^2 = t.a * t.c)
  (h2 : t.b = Real.sqrt 3) :
  Real.sqrt 3 < 2 * t.a + t.c ∧ 2 * t.a + t.c ≤ 2 * Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_range_of_2a_plus_c_l322_32221


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l322_32293

-- Define the given line
def given_line (x y : ℝ) : Prop := x + y - 5 = 0

-- Define the point that the perpendicular line passes through
def point : ℝ × ℝ := (2, -1)

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := x - y - 3 = 0

-- Theorem statement
theorem perpendicular_line_equation :
  ∃ (m b : ℝ),
    (∀ x y, perpendicular_line x y ↔ y = m * x + b) ∧
    perpendicular_line point.1 point.2 ∧
    (∀ x₁ y₁ x₂ y₂, given_line x₁ y₁ → given_line x₂ y₂ → x₁ ≠ x₂ →
      (y₂ - y₁) / (x₂ - x₁) * m = -1) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l322_32293


namespace NUMINAMATH_CALUDE_tournament_games_theorem_l322_32267

/-- Represents a single-elimination tournament. -/
structure Tournament :=
  (num_teams : ℕ)
  (no_ties : Bool)

/-- Calculates the number of games needed to determine a winner in a single-elimination tournament. -/
def games_to_win (t : Tournament) : ℕ :=
  t.num_teams - 1

/-- Theorem stating that a single-elimination tournament with 23 teams and no ties requires 22 games to determine a winner. -/
theorem tournament_games_theorem (t : Tournament) (h1 : t.num_teams = 23) (h2 : t.no_ties = true) : 
  games_to_win t = 22 := by
  sorry

end NUMINAMATH_CALUDE_tournament_games_theorem_l322_32267


namespace NUMINAMATH_CALUDE_box_stacking_comparison_l322_32214

/-- Represents the height of a stack of boxes -/
def stack_height (box_height : ℝ) (num_floors : ℕ) : ℝ :=
  box_height * (num_floors : ℝ)

/-- The problem statement -/
theorem box_stacking_comparison : 
  let box_a_height : ℝ := 3
  let box_b_height : ℝ := 3.5
  let taehyung_floors : ℕ := 16
  let yoongi_floors : ℕ := 14
  
  stack_height box_b_height yoongi_floors - stack_height box_a_height taehyung_floors = 1 := by
  sorry

end NUMINAMATH_CALUDE_box_stacking_comparison_l322_32214


namespace NUMINAMATH_CALUDE_max_area_of_nonoverlapping_triangle_l322_32262

/-- A triangle on a coordinate plane -/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

/-- Check if two triangles overlap -/
def overlap (t1 t2 : Triangle) : Prop := sorry

/-- Translation of a triangle by an integer vector -/
def translate (t : Triangle) (v : ℤ × ℤ) : Triangle := sorry

/-- Area of a triangle -/
def area (t : Triangle) : ℝ := sorry

/-- A triangle is valid if its translations by integer vectors do not overlap -/
def valid_triangle (t : Triangle) : Prop :=
  ∀ v : ℤ × ℤ, ¬(overlap t (translate t v))

theorem max_area_of_nonoverlapping_triangle :
  ∃ (t : Triangle), valid_triangle t ∧ area t = 2/3 ∧
  ∀ (t' : Triangle), valid_triangle t' → area t' ≤ 2/3 := by sorry

end NUMINAMATH_CALUDE_max_area_of_nonoverlapping_triangle_l322_32262


namespace NUMINAMATH_CALUDE_incorrect_multiplication_result_l322_32237

theorem incorrect_multiplication_result (x : ℕ) : 
  x * 153 = 109395 → x * 152 = 108680 := by
sorry

end NUMINAMATH_CALUDE_incorrect_multiplication_result_l322_32237


namespace NUMINAMATH_CALUDE_total_trees_is_86_l322_32255

/-- Calculates the number of trees that can be planted on a street --/
def treesOnStreet (length : ℕ) (spacing : ℕ) : ℕ :=
  (length / spacing) + 1

/-- The total number of trees that can be planted on all five streets --/
def totalTrees : ℕ :=
  treesOnStreet 151 14 + treesOnStreet 210 18 + treesOnStreet 275 12 +
  treesOnStreet 345 20 + treesOnStreet 475 22

theorem total_trees_is_86 : totalTrees = 86 := by
  sorry

end NUMINAMATH_CALUDE_total_trees_is_86_l322_32255


namespace NUMINAMATH_CALUDE_nicky_received_card_value_l322_32209

/-- The value of a card Nicky received in a trade, given the value of cards he traded and his profit -/
def card_value (traded_card_value : ℕ) (num_traded_cards : ℕ) (profit : ℕ) : ℕ :=
  traded_card_value * num_traded_cards + profit

/-- Theorem stating the value of the card Nicky received from Jill -/
theorem nicky_received_card_value :
  card_value 8 2 5 = 21 := by
  sorry

end NUMINAMATH_CALUDE_nicky_received_card_value_l322_32209


namespace NUMINAMATH_CALUDE_weekly_earnings_correct_l322_32264

/-- Represents the weekly earnings of Jake, Jacob, and Jim --/
structure WeeklyEarnings where
  jacob : ℕ
  jake : ℕ
  jim : ℕ

/-- Calculates the weekly earnings based on the given conditions --/
def calculateWeeklyEarnings : WeeklyEarnings :=
  let jacobWeekdayRate := 6
  let jacobWeekendRate := 8
  let weekdayHours := 8
  let weekendHours := 5
  let weekdays := 5
  let weekendDays := 2

  let jacobWeekdayEarnings := jacobWeekdayRate * weekdayHours * weekdays
  let jacobWeekendEarnings := jacobWeekendRate * weekendHours * weekendDays
  let jacobTotal := jacobWeekdayEarnings + jacobWeekendEarnings

  let jakeWeekdayRate := 3 * jacobWeekdayRate
  let jakeWeekdayEarnings := jakeWeekdayRate * weekdayHours * weekdays
  let jakeWeekendEarnings := jacobWeekendEarnings
  let jakeTotal := jakeWeekdayEarnings + jakeWeekendEarnings

  let jimWeekdayRate := 2 * jakeWeekdayRate
  let jimWeekdayEarnings := jimWeekdayRate * weekdayHours * weekdays
  let jimWeekendEarnings := jacobWeekendEarnings
  let jimTotal := jimWeekdayEarnings + jimWeekendEarnings

  { jacob := jacobTotal, jake := jakeTotal, jim := jimTotal }

/-- Theorem stating that the calculated weekly earnings match the expected values --/
theorem weekly_earnings_correct : 
  let earnings := calculateWeeklyEarnings
  earnings.jacob = 320 ∧ earnings.jake = 800 ∧ earnings.jim = 1520 := by
  sorry

end NUMINAMATH_CALUDE_weekly_earnings_correct_l322_32264


namespace NUMINAMATH_CALUDE_expected_black_pairs_standard_deck_l322_32288

/-- A standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (black_cards : ℕ)
  (red_cards : ℕ)
  (h_total : total_cards = black_cards + red_cards)

/-- A standard 104-card deck -/
def standard_deck : Deck :=
  { total_cards := 104,
    black_cards := 52,
    red_cards := 52,
    h_total := rfl }

/-- The expected number of pairs of adjacent black cards when dealt in a line -/
def expected_black_pairs (d : Deck) : ℚ :=
  (d.black_cards - 1 : ℚ) * (d.black_cards - 1) / (d.total_cards - 1)

theorem expected_black_pairs_standard_deck :
  expected_black_pairs standard_deck = 2601 / 103 :=
sorry

end NUMINAMATH_CALUDE_expected_black_pairs_standard_deck_l322_32288


namespace NUMINAMATH_CALUDE_sin_75_times_sin_15_l322_32200

theorem sin_75_times_sin_15 : Real.sin (75 * π / 180) * Real.sin (15 * π / 180) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_75_times_sin_15_l322_32200


namespace NUMINAMATH_CALUDE_triangle_problem_l322_32299

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  b = 7 →
  Real.sqrt 3 * b * Real.sin A - a * Real.cos B = 2 * a →
  (1 / 2) * a * c * Real.sin B = (15 * Real.sqrt 3) / 4 →
  B = (2 * Real.pi) / 3 ∧ a + c = 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l322_32299


namespace NUMINAMATH_CALUDE_decimal_23_to_binary_l322_32252

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec to_binary_helper (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else to_binary_helper (m / 2) ((m % 2) :: acc)
    to_binary_helper n []

theorem decimal_23_to_binary :
  decimal_to_binary 23 = [1, 0, 1, 1, 1] := by
  sorry

end NUMINAMATH_CALUDE_decimal_23_to_binary_l322_32252


namespace NUMINAMATH_CALUDE_problem_solution_l322_32258

theorem problem_solution (p q r : ℝ) 
  (h1 : p / q = 5 / 4)
  (h2 : p = r^2)
  (h3 : Real.sin r = 3 / 5) : 
  2 * p + q = 44.8 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l322_32258


namespace NUMINAMATH_CALUDE_semicircles_area_ratio_l322_32217

theorem semicircles_area_ratio (r : ℝ) (h : r > 0) :
  let circle_area := π * r^2
  let semicircle1_area := (π * r^2) / 2
  let semicircle2_area := (π * (r/2)^2) / 2
  (semicircle1_area + semicircle2_area) / circle_area = 5/8 := by
sorry

end NUMINAMATH_CALUDE_semicircles_area_ratio_l322_32217


namespace NUMINAMATH_CALUDE_inspector_rejection_l322_32219

-- Define the rejection rate
def rejection_rate : ℝ := 0.15

-- Define the number of meters examined
def meters_examined : ℝ := 66.67

-- Define the function to calculate the number of rejected meters
def rejected_meters (rate : ℝ) (total : ℝ) : ℝ := rate * total

-- Theorem statement
theorem inspector_rejection :
  rejected_meters rejection_rate meters_examined = 10 := by
  sorry

end NUMINAMATH_CALUDE_inspector_rejection_l322_32219


namespace NUMINAMATH_CALUDE_video_game_lives_l322_32238

theorem video_game_lives (initial_lives lost_lives gained_lives : ℕ) 
  (h1 : initial_lives = 43)
  (h2 : lost_lives = 14)
  (h3 : gained_lives = 27) :
  initial_lives - lost_lives + gained_lives = 56 :=
by sorry

end NUMINAMATH_CALUDE_video_game_lives_l322_32238


namespace NUMINAMATH_CALUDE_min_sum_with_reciprocal_constraint_l322_32278

theorem min_sum_with_reciprocal_constraint (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : 1 / a + 2 / b = 2) : 
  a + b ≥ (3 + 2 * Real.sqrt 2) / 2 ∧ 
  ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ 1 / a₀ + 2 / b₀ = 2 ∧ a₀ + b₀ = (3 + 2 * Real.sqrt 2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_with_reciprocal_constraint_l322_32278


namespace NUMINAMATH_CALUDE_intersection_locus_is_ellipse_l322_32215

/-- The locus of points (x, y) satisfying a system of equations forms an ellipse -/
theorem intersection_locus_is_ellipse :
  ∀ (s x y : ℝ), 
  (2 * s * x - 3 * y - 4 * s = 0) → 
  (x - 3 * s * y + 4 = 0) → 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_intersection_locus_is_ellipse_l322_32215


namespace NUMINAMATH_CALUDE_range_of_t_l322_32257

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def cubic_for_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x ≥ 0, f x = x^3

theorem range_of_t (f : ℝ → ℝ) (t : ℝ) :
  is_even_function f →
  cubic_for_nonneg f →
  (∀ x ∈ Set.Icc (2*t - 1) (2*t + 3), f (3*x - t) ≥ 8 * f x) →
  t ∈ Set.Iic (-3) ∪ {0} ∪ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_t_l322_32257


namespace NUMINAMATH_CALUDE_simplify_expression_l322_32265

theorem simplify_expression (m n : ℝ) : -2 * (m - n) = -2 * m + 2 * n := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l322_32265


namespace NUMINAMATH_CALUDE_set_intersection_and_union_l322_32294

theorem set_intersection_and_union (a : ℝ) : 
  let A : Set ℝ := {2, 3, a^2 + 4*a + 2}
  let B : Set ℝ := {0, 7, 2 - a, a^2 + 4*a - 2}
  (A ∩ B = {3, 7}) → 
  (a = 1 ∧ A ∪ B = {0, 1, 2, 3, 7}) :=
by sorry

end NUMINAMATH_CALUDE_set_intersection_and_union_l322_32294


namespace NUMINAMATH_CALUDE_M_properties_l322_32229

-- Define the operation M
def M : ℚ → ℚ
| n => if (↑n : ℚ).den = 1 
       then (↑n : ℚ).num - 3 
       else -(1 / ((↑n : ℚ).den ^ 2))

-- Theorem statement
theorem M_properties : 
  (M 28 * M (1/5) = -1) ∧ 
  (-1 / M 39 / (-M (1/6)) = -1) := by
  sorry

end NUMINAMATH_CALUDE_M_properties_l322_32229


namespace NUMINAMATH_CALUDE_expression_evaluation_l322_32234

theorem expression_evaluation : 
  |((4:ℝ)^2 - 8*((3:ℝ)^2 - 12))^2| - |Real.sin (5*π/6) - Real.cos (11*π/3)| = 1600 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l322_32234


namespace NUMINAMATH_CALUDE_cone_height_l322_32270

/-- The height of a cone with base area π and slant height 2 is √3 -/
theorem cone_height (base_area : Real) (slant_height : Real) :
  base_area = Real.pi → slant_height = 2 → ∃ (height : Real), height = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_height_l322_32270


namespace NUMINAMATH_CALUDE_triangle_formation_l322_32248

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_formation :
  can_form_triangle 13 12 20 ∧
  ¬ can_form_triangle 8 7 15 ∧
  ¬ can_form_triangle 5 5 11 ∧
  ¬ can_form_triangle 3 4 8 :=
by sorry

end NUMINAMATH_CALUDE_triangle_formation_l322_32248


namespace NUMINAMATH_CALUDE_counterfeit_coin_identifiable_l322_32245

/-- Represents the type of coin -/
inductive CoinType
| Gold
| Silver

/-- Represents a coin with its type and whether it's counterfeit -/
structure Coin where
  type : CoinType
  isCounterfeit : Bool

/-- Represents the result of a weighing -/
inductive WeighingResult
| Equal
| LeftHeavier
| RightHeavier

/-- Represents a group of coins -/
def CoinGroup := List Coin

/-- Represents a weighing action -/
def Weighing := CoinGroup → CoinGroup → WeighingResult

/-- The total number of coins -/
def totalCoins : Nat := 27

/-- The number of gold coins -/
def goldCoins : Nat := 13

/-- The number of silver coins -/
def silverCoins : Nat := 14

/-- The maximum number of weighings allowed -/
def maxWeighings : Nat := 3

/-- Axiom: There is exactly one counterfeit coin -/
axiom one_counterfeit (coins : List Coin) : 
  coins.length = totalCoins → ∃! c, c ∈ coins ∧ c.isCounterfeit

/-- Axiom: Counterfeit gold coin is lighter than real gold coins -/
axiom counterfeit_gold_lighter (w : Weighing) (c1 c2 : Coin) :
  c1.type = CoinType.Gold ∧ c2.type = CoinType.Gold ∧ c1.isCounterfeit ∧ ¬c2.isCounterfeit →
  w [c1] [c2] = WeighingResult.RightHeavier

/-- Axiom: Counterfeit silver coin is heavier than real silver coins -/
axiom counterfeit_silver_heavier (w : Weighing) (c1 c2 : Coin) :
  c1.type = CoinType.Silver ∧ c2.type = CoinType.Silver ∧ c1.isCounterfeit ∧ ¬c2.isCounterfeit →
  w [c1] [c2] = WeighingResult.LeftHeavier

/-- Axiom: Real coins of the same type have equal weight -/
axiom real_coins_equal_weight (w : Weighing) (c1 c2 : Coin) :
  c1.type = c2.type ∧ ¬c1.isCounterfeit ∧ ¬c2.isCounterfeit →
  w [c1] [c2] = WeighingResult.Equal

/-- The main theorem: It's possible to identify the counterfeit coin in at most three weighings -/
theorem counterfeit_coin_identifiable (coins : List Coin) (w : Weighing) :
  coins.length = totalCoins →
  ∃ (strategy : List (CoinGroup × CoinGroup)), 
    strategy.length ≤ maxWeighings ∧
    ∃ (counterfeit : Coin), counterfeit ∈ coins ∧ counterfeit.isCounterfeit ∧
    ∀ (c : Coin), c ∈ coins ∧ c.isCounterfeit → c = counterfeit :=
  sorry


end NUMINAMATH_CALUDE_counterfeit_coin_identifiable_l322_32245


namespace NUMINAMATH_CALUDE_school_gender_ratio_l322_32249

theorem school_gender_ratio (num_girls : ℕ) (ratio_boys : ℕ) (ratio_girls : ℕ) :
  num_girls = 1200 →
  ratio_boys = 5 →
  ratio_girls = 4 →
  (ratio_boys : ℚ) / ratio_girls * num_girls = 1500 :=
by sorry

end NUMINAMATH_CALUDE_school_gender_ratio_l322_32249


namespace NUMINAMATH_CALUDE_sequence_equals_index_l322_32272

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def sequence_property (a : ℕ → ℕ) : Prop :=
  (∀ n : ℕ, n ≥ 1 → a n > 0) ∧
  (∀ n m : ℕ, n ≥ 1 → m ≥ 1 → n < m → a n < a m) ∧
  (∀ n : ℕ, n ≥ 1 → a (2*n) = a n + n) ∧
  (∀ n : ℕ, n ≥ 1 → is_prime (a n) → is_prime n)

theorem sequence_equals_index (a : ℕ → ℕ) (h : sequence_property a) :
  ∀ n : ℕ, n ≥ 1 → a n = n :=
sorry

end NUMINAMATH_CALUDE_sequence_equals_index_l322_32272


namespace NUMINAMATH_CALUDE_existence_of_x_y_for_power_of_two_l322_32291

theorem existence_of_x_y_for_power_of_two (n : ℕ) (h : n ≥ 3) :
  ∃ x y : ℕ+, 2^n = 7 * x^2 + y^2 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_x_y_for_power_of_two_l322_32291


namespace NUMINAMATH_CALUDE_cryptarithm_solution_l322_32259

/-- Represents a digit in base 9 --/
def Digit := Fin 9

/-- Represents the cryptarithm LAKE + KALE + LEAK = KLAE in base 9 --/
def Cryptarithm (L A K E : Digit) : Prop :=
  (L.val + K.val + L.val) % 9 = K.val ∧
  (A.val + A.val + E.val) % 9 = L.val ∧
  (K.val + L.val + A.val) % 9 = A.val ∧
  (E.val + E.val + K.val) % 9 = E.val

/-- All digits are distinct --/
def DistinctDigits (L A K E : Digit) : Prop :=
  L ≠ A ∧ L ≠ K ∧ L ≠ E ∧ A ≠ K ∧ A ≠ E ∧ K ≠ E

theorem cryptarithm_solution :
  ∃ (L A K E : Digit),
    Cryptarithm L A K E ∧
    DistinctDigits L A K E ∧
    L.val = 0 ∧ E.val = 8 ∧ K.val = 4 ∧ (A.val = 1 ∨ A.val = 2 ∨ A.val = 3 ∨ A.val = 5 ∨ A.val = 6 ∨ A.val = 7) :=
by sorry

end NUMINAMATH_CALUDE_cryptarithm_solution_l322_32259


namespace NUMINAMATH_CALUDE_periodic_decimal_sum_l322_32204

/-- The sum of 0.3̅, 0.0̅4̅, and 0.0̅0̅5̅ is equal to 14/37 -/
theorem periodic_decimal_sum : 
  (1 / 3 : ℚ) + (4 / 99 : ℚ) + (5 / 999 : ℚ) = 14 / 37 := by sorry

end NUMINAMATH_CALUDE_periodic_decimal_sum_l322_32204


namespace NUMINAMATH_CALUDE_highway_extension_remaining_miles_l322_32222

/-- Proves that given the highway extension conditions, 250 miles still need to be added -/
theorem highway_extension_remaining_miles 
  (current_length : ℝ) 
  (final_length : ℝ) 
  (first_day_miles : ℝ) 
  (second_day_multiplier : ℝ) :
  current_length = 200 →
  final_length = 650 →
  first_day_miles = 50 →
  second_day_multiplier = 3 →
  final_length - current_length - first_day_miles - (second_day_multiplier * first_day_miles) = 250 := by
  sorry

#check highway_extension_remaining_miles

end NUMINAMATH_CALUDE_highway_extension_remaining_miles_l322_32222


namespace NUMINAMATH_CALUDE_third_digit_is_one_l322_32277

/-- A self-descriptive 7-digit number -/
structure SelfDescriptiveNumber where
  digits : Fin 7 → Fin 7
  sum_is_seven : (Finset.sum Finset.univ (λ i => digits i)) = 7
  first_digit : digits 0 = 3
  second_digit : digits 1 = 2
  fourth_digit : digits 3 = 1
  fifth_digit : digits 4 = 0

/-- The third digit of a self-descriptive number is 1 -/
theorem third_digit_is_one (n : SelfDescriptiveNumber) : n.digits 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_third_digit_is_one_l322_32277


namespace NUMINAMATH_CALUDE_smallest_non_square_product_of_four_primes_l322_32241

/-- A function that checks if a number is a perfect square --/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- A function that checks if a number is prime --/
def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

/-- A function that checks if a number is the product of four primes --/
def is_product_of_four_primes (n : ℕ) : Prop :=
  ∃ p q r s : ℕ, is_prime p ∧ is_prime q ∧ is_prime r ∧ is_prime s ∧ n = p * q * r * s

theorem smallest_non_square_product_of_four_primes :
  (∀ m : ℕ, m < 24 → ¬(is_product_of_four_primes m ∧ ¬is_perfect_square m)) ∧
  (is_product_of_four_primes 24 ∧ ¬is_perfect_square 24) :=
sorry

end NUMINAMATH_CALUDE_smallest_non_square_product_of_four_primes_l322_32241


namespace NUMINAMATH_CALUDE_find_multiple_of_q_l322_32236

theorem find_multiple_of_q (q : ℤ) (m : ℤ) : 
  let x := 55 + 2*q
  let y := m*q + 41
  (q = 7 → x = y) → m = 4 := by
sorry

end NUMINAMATH_CALUDE_find_multiple_of_q_l322_32236


namespace NUMINAMATH_CALUDE_larger_number_proof_l322_32212

theorem larger_number_proof (x y : ℝ) (sum_eq : x + y = 30) (diff_eq : x - y = 4) : 
  max x y = 17 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l322_32212


namespace NUMINAMATH_CALUDE_dangerous_animals_remaining_is_231_l322_32287

/-- The number of dangerous animals remaining in the swamp after some animals migrate --/
def dangerous_animals_remaining : ℕ :=
  let initial_crocodiles : ℕ := 42
  let initial_alligators : ℕ := 35
  let initial_vipers : ℕ := 10
  let initial_water_moccasins : ℕ := 28
  let initial_cottonmouth_snakes : ℕ := 15
  let initial_piranha_fish : ℕ := 120
  let migrating_crocodiles : ℕ := 9
  let migrating_alligators : ℕ := 7
  let migrating_vipers : ℕ := 3
  let total_initial := initial_crocodiles + initial_alligators + initial_vipers + 
                       initial_water_moccasins + initial_cottonmouth_snakes + initial_piranha_fish
  let total_migrating := migrating_crocodiles + migrating_alligators + migrating_vipers
  total_initial - total_migrating

/-- Theorem stating that the number of dangerous animals remaining in the swamp is 231 --/
theorem dangerous_animals_remaining_is_231 : dangerous_animals_remaining = 231 := by
  sorry

end NUMINAMATH_CALUDE_dangerous_animals_remaining_is_231_l322_32287


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l322_32283

/-- The area of a circle with circumference 24 cm is 144/π square centimeters. -/
theorem circle_area_from_circumference :
  ∀ (r : ℝ), 2 * π * r = 24 → π * r^2 = 144 / π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l322_32283


namespace NUMINAMATH_CALUDE_divisible_by_forty_l322_32225

theorem divisible_by_forty (n : ℕ) 
  (h1 : ∃ k : ℕ, 2 * n + 1 = k ^ 2) 
  (h2 : ∃ m : ℕ, 3 * n + 1 = m ^ 2) : 
  40 ∣ n := by
sorry

end NUMINAMATH_CALUDE_divisible_by_forty_l322_32225


namespace NUMINAMATH_CALUDE_conference_handshakes_l322_32205

/-- The number of handshakes in a conference of n people where each person
    shakes hands exactly once with every other person. -/
def handshakes (n : ℕ) : ℕ := n.choose 2

/-- Theorem stating that in a conference of 10 people where each person
    shakes hands exactly once with every other person, the total number
    of handshakes is 45. -/
theorem conference_handshakes :
  handshakes 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l322_32205


namespace NUMINAMATH_CALUDE_convergence_iff_cauchy_l322_32295

/-- A sequence of real numbers -/
def RealSequence := ℕ → ℝ

/-- Convergence of a sequence -/
def converges (x : RealSequence) : Prop :=
  ∃ (l : ℝ), ∀ ε > 0, ∃ N, ∀ n ≥ N, |x n - l| < ε

/-- Cauchy criterion for a sequence -/
def is_cauchy (x : RealSequence) : Prop :=
  ∀ ε > 0, ∃ N, ∀ m n, m ≥ N → n ≥ N → |x m - x n| < ε

/-- Theorem: A sequence of real numbers converges if and only if it satisfies the Cauchy criterion -/
theorem convergence_iff_cauchy (x : RealSequence) :
  converges x ↔ is_cauchy x :=
sorry

end NUMINAMATH_CALUDE_convergence_iff_cauchy_l322_32295


namespace NUMINAMATH_CALUDE_cubic_sum_reciprocal_l322_32281

theorem cubic_sum_reciprocal (x : ℝ) (h : x^3 + 1/x^3 = 110) : x + 1/x = 5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_reciprocal_l322_32281


namespace NUMINAMATH_CALUDE_adults_attending_play_l322_32251

/-- Proves the number of adults attending a play given ticket prices and total receipts --/
theorem adults_attending_play (adult_price children_price total_receipts total_attendance : ℕ) 
  (h1 : adult_price = 25)
  (h2 : children_price = 15)
  (h3 : total_receipts = 7200)
  (h4 : total_attendance = 400) :
  ∃ (adults children : ℕ), 
    adults + children = total_attendance ∧ 
    adult_price * adults + children_price * children = total_receipts ∧
    adults = 120 := by
  sorry


end NUMINAMATH_CALUDE_adults_attending_play_l322_32251


namespace NUMINAMATH_CALUDE_triangle_theorem_l322_32247

noncomputable section

open Real

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def condition (t : Triangle) : Prop :=
  (t.c / (t.a + t.b)) + (sin t.A / (sin t.B + sin t.C)) = 1

-- Define the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : condition t) 
  (h2 : t.b = sqrt 2) : 
  t.B = π/3 ∧ (∀ (x y : ℝ), x^2 + y^2 ≤ 4) ∧ (∃ (x y : ℝ), x^2 + y^2 = 4) :=
sorry

end

end NUMINAMATH_CALUDE_triangle_theorem_l322_32247


namespace NUMINAMATH_CALUDE_ice_cream_problem_l322_32250

def pennies : ℕ := 123
def nickels : ℕ := 85
def dimes : ℕ := 35
def quarters : ℕ := 26
def double_scoop_cost : ℚ := 3
def leftover : ℚ := 48/100

def total_amount : ℚ := 
  pennies * 1/100 + nickels * 5/100 + dimes * 10/100 + quarters * 25/100

theorem ice_cream_problem : 
  ∃ (n : ℕ), n * double_scoop_cost = total_amount - leftover ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_problem_l322_32250


namespace NUMINAMATH_CALUDE_reciprocal_roots_quadratic_equation_l322_32207

theorem reciprocal_roots_quadratic_equation :
  ∀ (α β : ℝ),
  (α^2 - 7*α - 1 = 0) →
  (β^2 - 7*β - 1 = 0) →
  (α + β = 7) →
  (α * β = -1) →
  ((1/α)^2 + 7*(1/α) - 1 = 0) ∧
  ((1/β)^2 + 7*(1/β) - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_roots_quadratic_equation_l322_32207


namespace NUMINAMATH_CALUDE_product_five_fourth_sum_l322_32280

theorem product_five_fourth_sum (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c → 
  a * b * c = 5^4 → 
  (a : ℕ) + (b : ℕ) + (c : ℕ) = 131 := by sorry

end NUMINAMATH_CALUDE_product_five_fourth_sum_l322_32280


namespace NUMINAMATH_CALUDE_fraction_product_equals_27_l322_32210

theorem fraction_product_equals_27 : 
  (1 : ℚ) / 3 * 9 / 1 * 1 / 27 * 81 / 1 * 1 / 243 * 729 / 1 = 27 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_equals_27_l322_32210


namespace NUMINAMATH_CALUDE_adjacent_probability_l322_32256

/-- Represents the number of students in the group photo --/
def total_students : ℕ := 6

/-- Represents the number of rows in the seating arrangement --/
def num_rows : ℕ := 3

/-- Represents the number of seats per row --/
def seats_per_row : ℕ := 2

/-- Calculates the total number of seating arrangements --/
def total_arrangements : ℕ := Nat.factorial total_students

/-- Calculates the number of favorable arrangements where Abby and Bridget are adjacent but not in the middle row --/
def favorable_arrangements : ℕ := 4 * 2 * Nat.factorial (total_students - 2)

/-- Represents the probability of Abby and Bridget being adjacent but not in the middle row --/
def probability : ℚ := favorable_arrangements / total_arrangements

/-- Theorem stating that the probability of Abby and Bridget being adjacent but not in the middle row is 4/15 --/
theorem adjacent_probability :
  probability = 4 / 15 := by sorry

end NUMINAMATH_CALUDE_adjacent_probability_l322_32256


namespace NUMINAMATH_CALUDE_intersection_sum_l322_32220

theorem intersection_sum (a b m : ℝ) : 
  ((-m + a = 8) ∧ (m + b = 8)) → a + b = 16 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l322_32220


namespace NUMINAMATH_CALUDE_snow_leopard_arrangement_l322_32276

theorem snow_leopard_arrangement (n : ℕ) (h : n = 9) : 
  (2 : ℕ) * (Nat.factorial (n - 3)) = 1440 := by
  sorry

end NUMINAMATH_CALUDE_snow_leopard_arrangement_l322_32276


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l322_32232

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ) : 
  (∀ n, S n = (n * (a 1 + a n)) / 2) →  -- Definition of sum for arithmetic sequence
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- Definition of arithmetic sequence
  m ≥ 2 →
  S (m - 1) = 16 →
  S m = 25 →
  S (m + 2) = 49 →
  m = 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l322_32232
