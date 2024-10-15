import Mathlib

namespace NUMINAMATH_CALUDE_emperor_strategy_exists_l3620_362000

/-- Represents the nature of a wizard -/
inductive WizardNature
| Good
| Evil

/-- Represents a wizard -/
structure Wizard where
  nature : WizardNature

/-- Represents the Emperor's knowledge about a wizard -/
inductive WizardKnowledge
| Unknown
| KnownGood
| KnownEvil

/-- Represents the state of the festival -/
structure FestivalState where
  wizards : Finset Wizard
  knowledge : Wizard → WizardKnowledge

/-- Represents a strategy for the Emperor -/
structure EmperorStrategy where
  askQuestion : FestivalState → Wizard → Prop
  expelWizard : FestivalState → Option Wizard

/-- The main theorem -/
theorem emperor_strategy_exists :
  ∃ (strategy : EmperorStrategy),
    ∀ (initial_state : FestivalState),
      initial_state.wizards.card = 2015 →
      ∃ (final_state : FestivalState),
        (∀ w ∈ final_state.wizards, w.nature = WizardNature.Good) ∧
        (∃! w, w ∉ final_state.wizards ∧ w.nature = WizardNature.Good) :=
by sorry

end NUMINAMATH_CALUDE_emperor_strategy_exists_l3620_362000


namespace NUMINAMATH_CALUDE_count_non_degenerate_triangles_l3620_362087

/-- The number of points in the figure -/
def total_points : ℕ := 16

/-- The number of collinear points on the base of the triangle -/
def base_points : ℕ := 5

/-- The number of collinear points on the semicircle -/
def semicircle_points : ℕ := 5

/-- The number of non-collinear points -/
def other_points : ℕ := total_points - base_points - semicircle_points

/-- Calculate the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of non-degenerate triangles -/
def non_degenerate_triangles : ℕ := 
  choose total_points 3 - 2 * choose base_points 3

theorem count_non_degenerate_triangles : 
  non_degenerate_triangles = 540 := by sorry

end NUMINAMATH_CALUDE_count_non_degenerate_triangles_l3620_362087


namespace NUMINAMATH_CALUDE_intersection_complement_equals_set_l3620_362071

def U : Set ℕ := {x | x^2 - 4*x - 5 ≤ 0}
def A : Set ℕ := {0, 2}
def B : Set ℕ := {1, 3, 5}

theorem intersection_complement_equals_set : A ∩ (U \ B) = {0, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_set_l3620_362071


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3620_362032

/-- The repeating decimal 0.5656... is equal to the fraction 56/99 -/
theorem repeating_decimal_to_fraction : 
  (∑' n, (56 : ℚ) / (100 ^ (n + 1))) = 56 / 99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3620_362032


namespace NUMINAMATH_CALUDE_wine_cost_theorem_l3620_362038

/-- The cost of a bottle of wine with a cork -/
def wineWithCorkCost (corkCost : ℝ) (extraCost : ℝ) : ℝ :=
  corkCost + (corkCost + extraCost)

/-- Theorem: The cost of a bottle of wine with a cork is $6.10 -/
theorem wine_cost_theorem (corkCost : ℝ) (extraCost : ℝ)
  (h1 : corkCost = 2.05)
  (h2 : extraCost = 2.00) :
  wineWithCorkCost corkCost extraCost = 6.10 := by
  sorry

#eval wineWithCorkCost 2.05 2.00

end NUMINAMATH_CALUDE_wine_cost_theorem_l3620_362038


namespace NUMINAMATH_CALUDE_unique_solution_is_one_l3620_362013

noncomputable def f (x : ℝ) : ℝ := 2 * x * Real.log x + x - 1

theorem unique_solution_is_one :
  ∃! x : ℝ, x > 0 ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_is_one_l3620_362013


namespace NUMINAMATH_CALUDE_intersection_count_l3620_362014

/-- The maximum number of intersection points formed by line segments connecting 
    points on the x-axis to points on the y-axis -/
def max_intersections (x_points y_points : ℕ) : ℕ :=
  (x_points.choose 2) * (y_points.choose 2)

/-- Theorem stating that for 8 points on the x-axis and 6 points on the y-axis, 
    the maximum number of intersections is 420 -/
theorem intersection_count : max_intersections 8 6 = 420 := by
  sorry

end NUMINAMATH_CALUDE_intersection_count_l3620_362014


namespace NUMINAMATH_CALUDE_percentage_equality_l3620_362090

theorem percentage_equality (x y : ℝ) (h1 : 3 * x = 3/4 * y) (h2 : x = 20) : y + 10 = 90 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equality_l3620_362090


namespace NUMINAMATH_CALUDE_equation_solutions_l3620_362072

theorem equation_solutions : 
  let f (x : ℝ) := 1 / (x^2 + 13*x - 10) + 1 / (x^2 + 4*x - 5) + 1 / (x^2 - 17*x - 10)
  ∀ x : ℝ, f x = 0 ↔ x = -2 + 2*Real.sqrt 14 ∨ x = -2 - 2*Real.sqrt 14 ∨ 
                    x = (7 + Real.sqrt 89) / 2 ∨ x = (7 - Real.sqrt 89) / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3620_362072


namespace NUMINAMATH_CALUDE_stock_percentage_problem_l3620_362044

/-- Calculates the percentage of a stock given income, investment, and stock price. -/
def stock_percentage (income : ℚ) (investment : ℚ) (stock_price : ℚ) : ℚ :=
  (income * stock_price) / investment

/-- Theorem stating that given the specific values in the problem, the stock percentage is 30%. -/
theorem stock_percentage_problem :
  let income : ℚ := 500
  let investment : ℚ := 1500
  let stock_price : ℚ := 90
  stock_percentage income investment stock_price = 30 := by sorry

end NUMINAMATH_CALUDE_stock_percentage_problem_l3620_362044


namespace NUMINAMATH_CALUDE_plane_through_three_points_l3620_362066

/-- A plane passing through three points in 3D space. -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- A point in 3D space. -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Check if a point lies on a plane. -/
def Point3D.liesOn (p : Point3D) (plane : Plane3D) : Prop :=
  plane.a * p.x + plane.b * p.y + plane.c * p.z + plane.d = 0

/-- The three given points. -/
def P₀ : Point3D := ⟨2, -1, 2⟩
def P₁ : Point3D := ⟨4, 3, 0⟩
def P₂ : Point3D := ⟨5, 2, 1⟩

/-- The plane equation we want to prove. -/
def targetPlane : Plane3D := ⟨1, -2, -3, 2⟩

theorem plane_through_three_points :
  P₀.liesOn targetPlane ∧ P₁.liesOn targetPlane ∧ P₂.liesOn targetPlane := by
  sorry


end NUMINAMATH_CALUDE_plane_through_three_points_l3620_362066


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3620_362017

theorem complex_number_quadrant : 
  let z : ℂ := (1/2 + (Real.sqrt 3 / 2) * Complex.I)^2
  z.re < 0 ∧ z.im > 0 := by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3620_362017


namespace NUMINAMATH_CALUDE_weight_measurement_l3620_362020

def weight_set : List Nat := [2, 5, 15]

def heaviest_weight (weights : List Nat) : Nat :=
  weights.sum

def different_weights (weights : List Nat) : Finset Nat :=
  sorry

theorem weight_measurement (weights : List Nat := weight_set) :
  (heaviest_weight weights = 22) ∧
  (different_weights weights).card = 9 := by
  sorry

end NUMINAMATH_CALUDE_weight_measurement_l3620_362020


namespace NUMINAMATH_CALUDE_task_completion_probability_l3620_362059

theorem task_completion_probability 
  (p1 : ℚ) (p2 : ℚ) 
  (h1 : p1 = 3 / 8) 
  (h2 : p2 = 3 / 5) : 
  p1 * (1 - p2) = 3 / 20 := by
sorry

end NUMINAMATH_CALUDE_task_completion_probability_l3620_362059


namespace NUMINAMATH_CALUDE_area_of_B_l3620_362067

-- Define the set A
def A : Set (ℝ × ℝ) := {p | p.1 + p.2 ≤ 1 ∧ p.1 ≥ 0 ∧ p.2 ≥ 0}

-- Define the transformation function
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 - p.2)

-- Define the set B
def B : Set (ℝ × ℝ) := f '' A

-- State the theorem
theorem area_of_B : MeasureTheory.volume B = 1 := by sorry

end NUMINAMATH_CALUDE_area_of_B_l3620_362067


namespace NUMINAMATH_CALUDE_division_reduction_l3620_362074

theorem division_reduction (x : ℕ) (h : x > 0) : 36 / x = 36 - 24 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_division_reduction_l3620_362074


namespace NUMINAMATH_CALUDE_sum_of_number_and_its_square_l3620_362061

theorem sum_of_number_and_its_square (x : ℕ) : x = 14 → x + x^2 = 210 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_number_and_its_square_l3620_362061


namespace NUMINAMATH_CALUDE_intersection_point_y_coordinate_l3620_362028

-- Define the original quadratic function
def original_function (x : ℝ) : ℝ := x^2 + 2*x + 1

-- Define the shifted function
def shifted_function (x : ℝ) : ℝ := (x + 3)^2 + 3

-- Theorem statement
theorem intersection_point_y_coordinate :
  shifted_function 0 = 12 :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_y_coordinate_l3620_362028


namespace NUMINAMATH_CALUDE_pencil_price_l3620_362034

theorem pencil_price (num_pens : ℕ) (num_pencils : ℕ) (total_cost : ℕ) (pen_price : ℕ) :
  num_pens = 30 →
  num_pencils = 75 →
  total_cost = 630 →
  pen_price = 16 →
  (total_cost - num_pens * pen_price) / num_pencils = 2 :=
by sorry

end NUMINAMATH_CALUDE_pencil_price_l3620_362034


namespace NUMINAMATH_CALUDE_superbloom_probability_l3620_362024

def campus : Finset Char := {'C', 'A', 'M', 'P', 'U', 'S'}
def sherbert : Finset Char := {'S', 'H', 'E', 'R', 'B', 'E', 'R', 'T'}
def globe : Finset Char := {'G', 'L', 'O', 'B', 'E'}
def superbloom : Finset Char := {'S', 'U', 'P', 'E', 'R', 'B', 'L', 'O', 'O', 'M'}

def probability_campus : ℚ := 1 / (campus.card.choose 3)
def probability_sherbert : ℚ := 9 / (sherbert.card.choose 5)
def probability_globe : ℚ := 1

theorem superbloom_probability :
  probability_campus * probability_sherbert * probability_globe = 9 / 1120 := by
  sorry

end NUMINAMATH_CALUDE_superbloom_probability_l3620_362024


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3620_362083

theorem inequality_solution_set (x : ℝ) : 
  (x^2 - x - 6) / (x - 1) > 0 ↔ (-2 < x ∧ x < 1) ∨ x > 3 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3620_362083


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3620_362084

theorem inequality_solution_set (x : ℝ) : 
  (x^2 - x - 2) / (x - 4) ≥ 3 ↔ x > 4 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3620_362084


namespace NUMINAMATH_CALUDE_adults_cookie_fraction_l3620_362093

theorem adults_cookie_fraction (total_cookies : ℕ) (num_children : ℕ) (cookies_per_child : ℕ) :
  total_cookies = 120 →
  num_children = 4 →
  cookies_per_child = 20 →
  (total_cookies - num_children * cookies_per_child : ℚ) / total_cookies = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_adults_cookie_fraction_l3620_362093


namespace NUMINAMATH_CALUDE_triangle_side_length_l3620_362022

/-- Given a right-angled triangle XYZ where angle XZY is 30° and XZ = 12, prove XY = 12√3 -/
theorem triangle_side_length (X Y Z : ℝ × ℝ) :
  (X.1 - Y.1)^2 + (X.2 - Y.2)^2 = ((X.1 - Z.1)^2 + (X.2 - Z.2)^2) + ((Y.1 - Z.1)^2 + (Y.2 - Z.2)^2) →  -- right-angled triangle
  Real.cos (Real.arccos ((Y.1 - Z.1) * (X.1 - Z.1) + (Y.2 - Z.2) * (X.2 - Z.2)) / 
    (Real.sqrt ((Y.1 - Z.1)^2 + (Y.2 - Z.2)^2) * Real.sqrt ((X.1 - Z.1)^2 + (X.2 - Z.2)^2))) = 1/2 →  -- angle XZY is 30°
  (X.1 - Z.1)^2 + (X.2 - Z.2)^2 = 144 →  -- XZ = 12
  (X.1 - Y.1)^2 + (X.2 - Y.2)^2 = 432  -- XY = 12√3
  := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3620_362022


namespace NUMINAMATH_CALUDE_at_least_one_passes_l3620_362096

/-- Represents the number of questions in the exam pool -/
def total_questions : ℕ := 10

/-- Represents the number of questions A can answer correctly -/
def a_correct : ℕ := 6

/-- Represents the number of questions B can answer correctly -/
def b_correct : ℕ := 8

/-- Represents the number of questions in each exam -/
def exam_questions : ℕ := 3

/-- Represents the minimum number of correct answers needed to pass -/
def pass_threshold : ℕ := 2

/-- Calculates the probability of an event -/
def prob (favorable : ℕ) (total : ℕ) : ℚ := (favorable : ℚ) / (total : ℚ)

/-- Calculates the probability of person A passing the exam -/
def prob_a_pass : ℚ := 
  prob (Nat.choose a_correct 2 * Nat.choose (total_questions - a_correct) 1 + 
        Nat.choose a_correct 3) 
       (Nat.choose total_questions exam_questions)

/-- Calculates the probability of person B passing the exam -/
def prob_b_pass : ℚ := 
  prob (Nat.choose b_correct 2 * Nat.choose (total_questions - b_correct) 1 + 
        Nat.choose b_correct 3) 
       (Nat.choose total_questions exam_questions)

/-- Theorem: The probability that at least one person passes the exam is 44/45 -/
theorem at_least_one_passes : 
  1 - (1 - prob_a_pass) * (1 - prob_b_pass) = 44 / 45 := by sorry

end NUMINAMATH_CALUDE_at_least_one_passes_l3620_362096


namespace NUMINAMATH_CALUDE_flower_pot_cost_difference_flower_pot_cost_difference_proof_l3620_362023

theorem flower_pot_cost_difference 
  (num_pots : ℕ) 
  (total_cost : ℚ) 
  (largest_pot_cost : ℚ) 
  (cost_difference : ℚ) : Prop :=
  num_pots = 6 ∧ 
  total_cost = 33/4 ∧ 
  largest_pot_cost = 13/8 ∧
  (∀ i : ℕ, i < num_pots - 1 → 
    (largest_pot_cost - i * cost_difference) > 
    (largest_pot_cost - (i + 1) * cost_difference)) ∧
  total_cost = (num_pots : ℚ) / 2 * 
    (2 * largest_pot_cost - (num_pots - 1 : ℚ) * cost_difference) →
  cost_difference = 1/10

theorem flower_pot_cost_difference_proof : 
  flower_pot_cost_difference 6 (33/4) (13/8) (1/10) :=
sorry

end NUMINAMATH_CALUDE_flower_pot_cost_difference_flower_pot_cost_difference_proof_l3620_362023


namespace NUMINAMATH_CALUDE_egg_container_problem_l3620_362009

theorem egg_container_problem (num_containers : ℕ) 
  (front_pos back_pos left_pos right_pos : ℕ) :
  num_containers = 28 →
  front_pos + back_pos = 34 →
  left_pos + right_pos = 5 →
  (num_containers * ((front_pos + back_pos - 1) * (left_pos + right_pos - 1))) = 3696 :=
by sorry

end NUMINAMATH_CALUDE_egg_container_problem_l3620_362009


namespace NUMINAMATH_CALUDE_at_least_three_equal_l3620_362031

theorem at_least_three_equal (a b c d : ℕ) 
  (h1 : (a + b)^2 % (c * d) = 0)
  (h2 : (a + c)^2 % (b * d) = 0)
  (h3 : (a + d)^2 % (b * c) = 0)
  (h4 : (b + c)^2 % (a * d) = 0)
  (h5 : (b + d)^2 % (a * c) = 0)
  (h6 : (c + d)^2 % (a * b) = 0) :
  (a = b ∧ b = c) ∨ (a = b ∧ b = d) ∨ (a = c ∧ c = d) ∨ (b = c ∧ c = d) := by
sorry

end NUMINAMATH_CALUDE_at_least_three_equal_l3620_362031


namespace NUMINAMATH_CALUDE_problem_solution_l3620_362047

/-- The function f as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2 else x^2 + a*x

/-- Theorem stating the solution to the problem -/
theorem problem_solution (a : ℝ) : f a (f a 0) = 4 * a → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3620_362047


namespace NUMINAMATH_CALUDE_cos_72_minus_cos_144_eq_zero_l3620_362019

theorem cos_72_minus_cos_144_eq_zero : 
  Real.cos (72 * π / 180) - Real.cos (144 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_72_minus_cos_144_eq_zero_l3620_362019


namespace NUMINAMATH_CALUDE_product_mod_seven_l3620_362064

theorem product_mod_seven : (2023 * 2024 * 2025 * 2026) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seven_l3620_362064


namespace NUMINAMATH_CALUDE_mikes_shopping_cost_l3620_362055

/-- The total amount Mike spent on shopping --/
def total_spent (food_cost wallet_cost shirt_cost : ℝ) : ℝ :=
  food_cost + wallet_cost + shirt_cost

/-- Theorem stating the total amount Mike spent on shopping --/
theorem mikes_shopping_cost :
  ∀ (food_cost wallet_cost shirt_cost : ℝ),
    food_cost = 30 →
    wallet_cost = food_cost + 60 →
    shirt_cost = wallet_cost / 3 →
    total_spent food_cost wallet_cost shirt_cost = 150 := by
  sorry

end NUMINAMATH_CALUDE_mikes_shopping_cost_l3620_362055


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3620_362035

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b : ℝ, b < a ∧ a < 0 → 1/b > 1/a) ∧
  ¬(∀ a b : ℝ, 1/b > 1/a → b < a ∧ a < 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3620_362035


namespace NUMINAMATH_CALUDE_total_customers_is_43_l3620_362018

/-- Represents a table with a number of women and men -/
structure Table where
  women : ℕ
  men : ℕ

/-- Calculates the total number of customers at a table -/
def Table.total (t : Table) : ℕ := t.women + t.men

/-- Represents the waiter's situation -/
structure WaiterSituation where
  table1 : Table
  table2 : Table
  table3 : Table
  table4 : Table
  table5 : Table
  table6 : Table
  walkIn : Table
  table3Left : ℕ
  table4Joined : Table

/-- The initial situation of the waiter -/
def initialSituation : WaiterSituation where
  table1 := { women := 2, men := 4 }
  table2 := { women := 4, men := 3 }
  table3 := { women := 3, men := 5 }
  table4 := { women := 5, men := 2 }
  table5 := { women := 2, men := 1 }
  table6 := { women := 1, men := 2 }
  walkIn := { women := 4, men := 4 }
  table3Left := 2
  table4Joined := { women := 1, men := 2 }

/-- Calculates the total number of customers served by the waiter -/
def totalCustomersServed (s : WaiterSituation) : ℕ :=
  s.table1.total +
  s.table2.total +
  (s.table3.total - s.table3Left) +
  (s.table4.total + s.table4Joined.total) +
  s.table5.total +
  s.table6.total +
  s.walkIn.total

/-- Theorem stating that the total number of customers served is 43 -/
theorem total_customers_is_43 : totalCustomersServed initialSituation = 43 := by
  sorry

end NUMINAMATH_CALUDE_total_customers_is_43_l3620_362018


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3620_362057

theorem trigonometric_identities :
  (Real.sin (75 * π / 180))^2 - (Real.cos (75 * π / 180))^2 = Real.sqrt 3 / 2 ∧
  Real.sin (75 * π / 180) * Real.cos (75 * π / 180) = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3620_362057


namespace NUMINAMATH_CALUDE_min_cubes_satisfy_conditions_num_cubes_is_minimum_l3620_362027

/-- Represents the number of cubes in each identical box -/
def num_cubes : ℕ := 1344

/-- Represents the side length of the outer square in the first girl's frame -/
def frame_outer : ℕ := 50

/-- Represents the side length of the inner square in the first girl's frame -/
def frame_inner : ℕ := 34

/-- Represents the side length of the second girl's square -/
def square_second : ℕ := 62

/-- Represents the side length of the third girl's square -/
def square_third : ℕ := 72

/-- Theorem stating that the given number of cubes satisfies all conditions -/
theorem min_cubes_satisfy_conditions :
  (frame_outer^2 - frame_inner^2 = num_cubes) ∧
  (square_second^2 = num_cubes) ∧
  (square_third^2 + 4 = num_cubes) := by
  sorry

/-- Theorem stating that the given number of cubes is the minimum possible -/
theorem num_cubes_is_minimum (n : ℕ) :
  (n < num_cubes) →
  ¬((frame_outer^2 - frame_inner^2 = n) ∧
    (∃ m : ℕ, m^2 = n) ∧
    (∃ k : ℕ, k^2 + 4 = n)) := by
  sorry

end NUMINAMATH_CALUDE_min_cubes_satisfy_conditions_num_cubes_is_minimum_l3620_362027


namespace NUMINAMATH_CALUDE_not_divisible_sum_of_not_divisible_product_plus_one_l3620_362026

theorem not_divisible_sum_of_not_divisible_product_plus_one (n : ℕ) 
  (h : ∀ (a b : ℕ), ¬(n ∣ 2^a * 3^b + 1)) :
  ∀ (c d : ℕ), ¬(n ∣ 2^c + 3^d) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_sum_of_not_divisible_product_plus_one_l3620_362026


namespace NUMINAMATH_CALUDE_problem_solution_l3620_362095

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 4*x - 5 ≤ 0
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the theorem
theorem problem_solution :
  (∀ m : ℝ, m > 0 → (∀ x : ℝ, p x → q x m) → m ≥ 4) ∧
  (∀ x : ℝ, (p x ∨ q x 5) ∧ ¬(p x ∧ q x 5) → (x ∈ Set.Icc (-4 : ℝ) (-1) ∪ Set.Ioc 5 6)) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l3620_362095


namespace NUMINAMATH_CALUDE_sum_of_xyz_equals_sqrt_13_l3620_362070

theorem sum_of_xyz_equals_sqrt_13 (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + y^2 + x*y = 3)
  (eq2 : y^2 + z^2 + y*z = 4)
  (eq3 : z^2 + x^2 + z*x = 7) :
  x + y + z = Real.sqrt 13 := by
sorry

end NUMINAMATH_CALUDE_sum_of_xyz_equals_sqrt_13_l3620_362070


namespace NUMINAMATH_CALUDE_rectangle_problem_l3620_362073

theorem rectangle_problem (a b k l : ℕ) (h1 : k * l = 47 * (a + b)) 
  (h2 : a * k = b * l) : k = 2256 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_problem_l3620_362073


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l3620_362077

theorem smallest_prime_divisor_of_sum (p : ℕ → ℕ → Prop) :
  (∀ n : ℕ, p n 2 → (∃ m : ℕ, n = 2 * m)) →
  (∀ n : ℕ, p 2 n → n = 2) →
  p 2 (3^19 + 11^13) ∧ 
  ∀ q : ℕ, p q (3^19 + 11^13) → q ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l3620_362077


namespace NUMINAMATH_CALUDE_no_grasshopper_overlap_l3620_362094

/-- Represents the position of a grasshopper -/
structure Position where
  x : ℤ
  y : ℤ

/-- Represents the state of all four grasshoppers -/
structure GrasshopperState where
  g1 : Position
  g2 : Position
  g3 : Position
  g4 : Position

/-- Calculates the center of mass of three positions -/
def centerOfMass (p1 p2 p3 : Position) : Position :=
  { x := (p1.x + p2.x + p3.x) / 3,
    y := (p1.y + p2.y + p3.y) / 3 }

/-- Calculates the symmetric position of a point with respect to another point -/
def symmetricPosition (p center : Position) : Position :=
  { x := 2 * center.x - p.x,
    y := 2 * center.y - p.y }

/-- Performs a single jump for one grasshopper -/
def jump (state : GrasshopperState) (jumper : Fin 4) : GrasshopperState :=
  match jumper with
  | 0 => { state with g1 := symmetricPosition state.g1 (centerOfMass state.g2 state.g3 state.g4) }
  | 1 => { state with g2 := symmetricPosition state.g2 (centerOfMass state.g1 state.g3 state.g4) }
  | 2 => { state with g3 := symmetricPosition state.g3 (centerOfMass state.g1 state.g2 state.g4) }
  | 3 => { state with g4 := symmetricPosition state.g4 (centerOfMass state.g1 state.g2 state.g3) }

/-- Checks if any two grasshoppers are at the same position -/
def hasOverlap (state : GrasshopperState) : Prop :=
  state.g1 = state.g2 ∨ state.g1 = state.g3 ∨ state.g1 = state.g4 ∨
  state.g2 = state.g3 ∨ state.g2 = state.g4 ∨
  state.g3 = state.g4

/-- Initial state of the grasshoppers on a square -/
def initialState (n : ℕ) : GrasshopperState :=
  { g1 := { x := 0,     y := 0 },
    g2 := { x := 3^n,   y := 0 },
    g3 := { x := 3^n,   y := 3^n },
    g4 := { x := 0,     y := 3^n } }

/-- The main theorem stating that no overlap occurs after any number of jumps -/
theorem no_grasshopper_overlap (n : ℕ) :
  ∀ (jumps : List (Fin 4)), ¬(hasOverlap (jumps.foldl jump (initialState n))) :=
sorry

end NUMINAMATH_CALUDE_no_grasshopper_overlap_l3620_362094


namespace NUMINAMATH_CALUDE_power_nine_mod_hundred_l3620_362004

theorem power_nine_mod_hundred : 9^2050 % 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_nine_mod_hundred_l3620_362004


namespace NUMINAMATH_CALUDE_ratio_of_numbers_l3620_362046

theorem ratio_of_numbers (x y : ℝ) (h : x > y) (h' : (x + y) / (x - y) = 4 / 3) :
  x / y = 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_l3620_362046


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l3620_362068

theorem quadratic_roots_property (a b : ℝ) : 
  (3 * a^2 + 4 * a - 7 = 0) → 
  (3 * b^2 + 4 * b - 7 = 0) → 
  (a - 2) * (b - 2) = 13/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l3620_362068


namespace NUMINAMATH_CALUDE_orvin_balloon_purchase_l3620_362063

/-- Represents the cost of balloons in cents -/
def regular_price : ℕ := 200

/-- Represents the total amount of money Orvin has in cents -/
def total_money : ℕ := 40 * regular_price

/-- Represents the cost of a pair of balloons (one at regular price, one at half price) in cents -/
def pair_cost : ℕ := regular_price + regular_price / 2

/-- The maximum number of balloons Orvin can buy -/
def max_balloons : ℕ := 2 * (total_money / pair_cost)

theorem orvin_balloon_purchase :
  max_balloons = 52 := by sorry

end NUMINAMATH_CALUDE_orvin_balloon_purchase_l3620_362063


namespace NUMINAMATH_CALUDE_range_of_a_l3620_362040

theorem range_of_a (a : ℝ) : 
  (∃ x ∈ Set.Icc 1 2, x^2 + 2*x - a ≥ 0) → a ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3620_362040


namespace NUMINAMATH_CALUDE_similar_quadratic_radicals_l3620_362052

def are_similar_quadratic_radicals (a b : ℝ) : Prop :=
  ∃ (k : ℚ), a = k * b

theorem similar_quadratic_radicals :
  are_similar_quadratic_radicals (Real.sqrt 18) (Real.sqrt 72) ∧
  ¬ are_similar_quadratic_radicals (Real.sqrt 12) (Real.sqrt 18) ∧
  ¬ are_similar_quadratic_radicals (Real.sqrt 20) (Real.sqrt 50) ∧
  ¬ are_similar_quadratic_radicals (Real.sqrt 24) (Real.sqrt 32) :=
by sorry

end NUMINAMATH_CALUDE_similar_quadratic_radicals_l3620_362052


namespace NUMINAMATH_CALUDE_odd_heads_probability_not_simple_closed_form_l3620_362006

/-- Represents the probability of getting heads on the i-th flip -/
def p (i : ℕ) : ℚ := 3/4 - i/200

/-- Represents the probability of having an odd number of heads after n flips -/
noncomputable def P : ℕ → ℚ
  | 0 => 0
  | n + 1 => (1 - 2 * p n) * P n + p n

/-- The statement that the probability of odd number of heads after 100 flips
    cannot be expressed in a simple closed form -/
theorem odd_heads_probability_not_simple_closed_form :
  ∃ (f : ℚ → Prop), f (P 100) ∧ ∀ (x : ℚ), f x → x = P 100 :=
sorry

end NUMINAMATH_CALUDE_odd_heads_probability_not_simple_closed_form_l3620_362006


namespace NUMINAMATH_CALUDE_business_partnership_timing_l3620_362082

/-- Represents the number of months after A started that B joined the business. -/
def months_until_b_joined : ℕ → Prop :=
  fun x =>
    let a_investment := 3500 * 12
    let b_investment := 21000 * (12 - x)
    a_investment * 3 = b_investment * 2

theorem business_partnership_timing :
  ∃ x : ℕ, months_until_b_joined x ∧ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_business_partnership_timing_l3620_362082


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3620_362002

theorem max_value_of_expression (x y a : ℝ) (hx : x > 0) (hy : y > 0) (ha : a > 0) :
  (x + y + a)^2 / (x^2 + y^2 + a^2) ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3620_362002


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l3620_362098

theorem smallest_number_with_remainders : ∃ n : ℕ,
  (n % 10 = 9) ∧
  (n % 9 = 8) ∧
  (n % 8 = 7) ∧
  (n % 7 = 6) ∧
  (n % 6 = 5) ∧
  (n % 5 = 4) ∧
  (n % 4 = 3) ∧
  (n % 3 = 2) ∧
  (n % 2 = 1) ∧
  (∀ m : ℕ, m < n →
    ¬((m % 10 = 9) ∧
      (m % 9 = 8) ∧
      (m % 8 = 7) ∧
      (m % 7 = 6) ∧
      (m % 6 = 5) ∧
      (m % 5 = 4) ∧
      (m % 4 = 3) ∧
      (m % 3 = 2) ∧
      (m % 2 = 1))) ∧
  n = 2519 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l3620_362098


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l3620_362048

theorem rectangle_dimensions : ∀ x y : ℝ,
  y = 2 * x →
  2 * (x + y) = 2 * (x * y) →
  x = (3 : ℝ) / 2 ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l3620_362048


namespace NUMINAMATH_CALUDE_trent_travel_distance_l3620_362043

/-- The distance Trent walked from his house to the bus stop -/
def distance_to_bus_stop : ℕ := 4

/-- The distance Trent rode the bus to the library -/
def distance_on_bus : ℕ := 7

/-- The total distance Trent traveled in blocks -/
def total_distance : ℕ := 2 * (distance_to_bus_stop + distance_on_bus)

theorem trent_travel_distance : total_distance = 22 := by
  sorry

end NUMINAMATH_CALUDE_trent_travel_distance_l3620_362043


namespace NUMINAMATH_CALUDE_jeff_fish_problem_l3620_362008

/-- The problem of finding the maximum mass of a single fish caught by Jeff. -/
theorem jeff_fish_problem (n : ℕ) (min_mass : ℝ) (first_three_mass : ℝ) :
  n = 21 ∧
  min_mass = 0.2 ∧
  first_three_mass = 1.5 ∧
  (∀ fish : ℕ, fish ≤ n → ∃ (mass : ℝ), mass ≥ min_mass) ∧
  (first_three_mass / 3 = (first_three_mass + (n - 3) * min_mass) / n) →
  ∃ (max_mass : ℝ), max_mass = 5.6 ∧ 
    ∀ (fish_mass : ℝ), (∃ (fish : ℕ), fish ≤ n ∧ fish_mass ≥ min_mass) → fish_mass ≤ max_mass :=
by sorry

end NUMINAMATH_CALUDE_jeff_fish_problem_l3620_362008


namespace NUMINAMATH_CALUDE_inequality_relations_l3620_362069

theorem inequality_relations (a d e : ℝ) 
  (h1 : a < 0) (h2 : a < d) (h3 : d < e) : 
  (a * d < d * e) ∧ 
  (a * e < d * e) ∧ 
  (a + d < d + e) ∧ 
  (e / a < 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_relations_l3620_362069


namespace NUMINAMATH_CALUDE_triangle_properties_l3620_362011

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a + t.b = 5 ∧
  t.c = Real.sqrt 7 ∧
  4 * (Real.sin ((t.A + t.B) / 2))^2 - Real.cos (2 * t.C) = 7/2 ∧
  t.A + t.B + t.C = Real.pi

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h : triangle_conditions t) : 
  t.C = Real.pi / 3 ∧ 
  (1/2) * t.a * t.b * Real.sin t.C = (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3620_362011


namespace NUMINAMATH_CALUDE_pell_equation_solution_form_l3620_362033

/-- Pell's equation solution type -/
structure PellSolution (d : ℕ) :=
  (x : ℕ)
  (y : ℕ)
  (eq : x^2 - d * y^2 = 1)

/-- Fundamental solution to Pell's equation -/
def fundamental_solution (d : ℕ) : PellSolution d := sorry

/-- Any solution to Pell's equation -/
def any_solution (d : ℕ) : PellSolution d := sorry

/-- Square-free natural number -/
def is_square_free (d : ℕ) : Prop := sorry

theorem pell_equation_solution_form 
  (d : ℕ) 
  (h_square_free : is_square_free d) 
  (x₁ y₁ : ℕ) 
  (h_fund : fundamental_solution d = ⟨x₁, y₁, sorry⟩) 
  (xₙ yₙ : ℕ) 
  (h_any : any_solution d = ⟨xₙ, yₙ, sorry⟩) :
  ∃ (n : ℕ), (xₙ : ℝ) + yₙ * Real.sqrt d = ((x₁ : ℝ) + y₁ * Real.sqrt d) ^ n :=
sorry

end NUMINAMATH_CALUDE_pell_equation_solution_form_l3620_362033


namespace NUMINAMATH_CALUDE_x_minus_y_value_l3620_362042

theorem x_minus_y_value (x y : ℝ) (h : x^2 + 6*x + 9 + Real.sqrt (y - 3) = 0) : 
  x - y = -6 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_value_l3620_362042


namespace NUMINAMATH_CALUDE_median_in_interval_75_79_l3620_362089

structure ScoreInterval :=
  (lower upper : ℕ)
  (count : ℕ)

def total_students : ℕ := 100

def score_distribution : List ScoreInterval :=
  [⟨85, 89, 18⟩, ⟨80, 84, 15⟩, ⟨75, 79, 20⟩, ⟨70, 74, 25⟩, ⟨65, 69, 12⟩, ⟨60, 64, 10⟩]

def cumulative_count (n : ℕ) : ℕ :=
  (score_distribution.take n).foldl (λ acc interval => acc + interval.count) 0

theorem median_in_interval_75_79 :
  ∃ k, k ∈ [75, 76, 77, 78, 79] ∧
    cumulative_count 2 < total_students / 2 ∧
    total_students / 2 ≤ cumulative_count 3 :=
  sorry

end NUMINAMATH_CALUDE_median_in_interval_75_79_l3620_362089


namespace NUMINAMATH_CALUDE_polar_to_cartesian_circle_l3620_362086

/-- The polar equation ρ = cos(π/4 - θ) represents a circle in Cartesian coordinates -/
theorem polar_to_cartesian_circle :
  ∃ (h k r : ℝ), ∀ (x y : ℝ),
    (∃ (ρ θ : ℝ), ρ = Real.cos (π/4 - θ) ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
    (x - h)^2 + (y - k)^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_circle_l3620_362086


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3620_362015

/-- Two vectors in ℝ² are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (x, 4)
  let b : ℝ × ℝ := (4, x)
  parallel a b → x = 4 ∨ x = -4 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3620_362015


namespace NUMINAMATH_CALUDE_jerry_reading_pages_l3620_362005

theorem jerry_reading_pages : ∀ (total_pages pages_read_saturday pages_remaining : ℕ),
  total_pages = 93 →
  pages_read_saturday = 30 →
  pages_remaining = 43 →
  total_pages - pages_read_saturday - pages_remaining = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_jerry_reading_pages_l3620_362005


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3620_362054

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h1 : a 1 = 8)
  (h2 : ∀ n : ℕ, a (n + 1) = a n * (a 2 / a 1))
  (h3 : a 4 = a 3 * a 5) :
  a 2 / a 1 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3620_362054


namespace NUMINAMATH_CALUDE_incorrect_propositions_l3620_362053

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- A plane in 3D space -/
structure Plane3D where
  point : Point3D
  normal : Point3D

/-- Two lines intersect -/
def intersect (l1 l2 : Line3D) : Prop := sorry

/-- A point lies on a line -/
def point_on_line (p : Point3D) (l : Line3D) : Prop := sorry

/-- A point lies on a plane -/
def point_on_plane (p : Point3D) (plane : Plane3D) : Prop := sorry

/-- Two lines are perpendicular -/
def perpendicular (l1 l2 : Line3D) : Prop := sorry

/-- Two lines are skew (not parallel and not intersecting) -/
def skew (l1 l2 : Line3D) : Prop := sorry

/-- A line lies on a plane -/
def line_on_plane (l : Line3D) (p : Plane3D) : Prop := sorry

/-- Three points determine a plane -/
def points_determine_plane (p1 p2 p3 : Point3D) : Prop := sorry

theorem incorrect_propositions :
  -- Proposition ③: Three points on two intersecting lines determine a plane
  ¬ (∀ (l1 l2 : Line3D) (p1 p2 p3 : Point3D),
    intersect l1 l2 →
    point_on_line p1 l1 →
    point_on_line p2 l1 →
    point_on_line p3 l2 →
    points_determine_plane p1 p2 p3) ∧
  -- Proposition ④: Two perpendicular lines are coplanar
  ¬ (∀ (l1 l2 : Line3D),
    perpendicular l1 l2 →
    ∃ (p : Plane3D), line_on_plane l1 p ∧ line_on_plane l2 p) :=
by sorry

end NUMINAMATH_CALUDE_incorrect_propositions_l3620_362053


namespace NUMINAMATH_CALUDE_min_students_same_score_l3620_362045

theorem min_students_same_score (total_students : ℕ) (min_score max_score : ℕ) :
  total_students = 8000 →
  min_score = 30 →
  max_score = 83 →
  ∃ (score : ℕ), min_score ≤ score ∧ score ≤ max_score ∧
    (∃ (students_with_score : ℕ), students_with_score ≥ 149 ∧
      (∀ (s : ℕ), min_score ≤ s ∧ s ≤ max_score →
        (∃ (students : ℕ), students ≤ students_with_score))) :=
by sorry

end NUMINAMATH_CALUDE_min_students_same_score_l3620_362045


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l3620_362076

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 - 2*x = 2 ↔ (x - 1)^2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l3620_362076


namespace NUMINAMATH_CALUDE_lcm_hcf_relation_l3620_362010

theorem lcm_hcf_relation (a b : ℕ) (ha : a = 210) (hlcm : Nat.lcm a b = 2310) :
  Nat.gcd a b = a * b / Nat.lcm a b :=
by sorry

end NUMINAMATH_CALUDE_lcm_hcf_relation_l3620_362010


namespace NUMINAMATH_CALUDE_no_valid_x_l3620_362092

/-- A circle in the xy-plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if a point is on a circle --/
def is_on_circle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

theorem no_valid_x : 
  ∀ x : ℝ, ¬∃ (c : Circle), 
    c.center = (15, 0) ∧ 
    c.radius = 15 ∧
    is_on_circle c (x, 18) ∧ 
    is_on_circle c (x, -18) := by
  sorry

end NUMINAMATH_CALUDE_no_valid_x_l3620_362092


namespace NUMINAMATH_CALUDE_arc_length_central_angle_l3620_362080

theorem arc_length_central_angle (r : ℝ) (θ : ℝ) (h : θ = π / 2) :
  let circum := 2 * π * r
  let arc_length := (θ / (2 * π)) * circum
  r = 15 → arc_length = 7.5 * π := by
  sorry

end NUMINAMATH_CALUDE_arc_length_central_angle_l3620_362080


namespace NUMINAMATH_CALUDE_roots_equation_sum_l3620_362075

theorem roots_equation_sum (α β : ℝ) : 
  α^2 - 3*α - 4 = 0 → β^2 - 3*β - 4 = 0 → 3*α^3 + 7*β^4 = 1591 :=
by
  sorry

end NUMINAMATH_CALUDE_roots_equation_sum_l3620_362075


namespace NUMINAMATH_CALUDE_percentage_calculation_l3620_362099

theorem percentage_calculation (N : ℝ) (P : ℝ) 
  (h1 : N = 125) 
  (h2 : N = (P / 100) * N + 105) : 
  P = 16 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l3620_362099


namespace NUMINAMATH_CALUDE_twelfth_number_is_seven_l3620_362078

/-- A circular arrangement of 20 numbers -/
def CircularArrangement := Fin 20 → ℕ

/-- The property that the sum of any six consecutive numbers is 24 -/
def SumProperty (arr : CircularArrangement) : Prop :=
  ∀ i : Fin 20, (arr i + arr (i + 1) + arr (i + 2) + arr (i + 3) + arr (i + 4) + arr (i + 5)) = 24

theorem twelfth_number_is_seven
  (arr : CircularArrangement)
  (h_sum : SumProperty arr)
  (h_first : arr 0 = 1) :
  arr 11 = 7 := by
  sorry

end NUMINAMATH_CALUDE_twelfth_number_is_seven_l3620_362078


namespace NUMINAMATH_CALUDE_f_min_value_inequality_proof_l3620_362060

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| + |x - 1|

-- Theorem for the minimum value of f(x)
theorem f_min_value : ∃ (a : ℝ), ∀ (x : ℝ), f x ≥ a ∧ ∃ (y : ℝ), f y = a :=
sorry

-- Theorem for the inequality
theorem inequality_proof (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : m^3 + n^3 = 2) : m + n ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_f_min_value_inequality_proof_l3620_362060


namespace NUMINAMATH_CALUDE_area_ratio_is_three_thirtyseconds_l3620_362051

/-- Triangle PQR with points X, Y, Z on its sides -/
structure TriangleWithPoints where
  -- Side lengths of triangle PQR
  pq : ℝ
  qr : ℝ
  rp : ℝ
  -- Ratios for points X, Y, Z
  x : ℝ
  y : ℝ
  z : ℝ
  -- Conditions
  pq_eq : pq = 7
  qr_eq : qr = 24
  rp_eq : rp = 25
  x_pos : x > 0
  y_pos : y > 0
  z_pos : z > 0
  sum_eq : x + y + z = 3/4
  sum_sq_eq : x^2 + y^2 + z^2 = 3/8

/-- The ratio of areas of triangle XYZ to triangle PQR -/
def areaRatio (t : TriangleWithPoints) : ℚ :=
  3/32

/-- Theorem stating that the area ratio is 3/32 -/
theorem area_ratio_is_three_thirtyseconds (t : TriangleWithPoints) :
  areaRatio t = 3/32 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_is_three_thirtyseconds_l3620_362051


namespace NUMINAMATH_CALUDE_third_day_temperature_l3620_362091

/-- Given the average temperature of three days and the temperatures of two of those days,
    calculate the temperature of the third day. -/
theorem third_day_temperature
  (avg_temp : ℚ)
  (day1_temp : ℚ)
  (day3_temp : ℚ)
  (h1 : avg_temp = -7)
  (h2 : day1_temp = -14)
  (h3 : day3_temp = 1)
  : (3 * avg_temp - day1_temp - day3_temp : ℚ) = -8 := by
  sorry

end NUMINAMATH_CALUDE_third_day_temperature_l3620_362091


namespace NUMINAMATH_CALUDE_range_of_f_l3620_362021

noncomputable def f (x : ℝ) : ℝ := Real.arctan (2 * x) + Real.arctan ((2 - x) / (2 + x))

theorem range_of_f :
  Set.range f = {-3 * Real.pi / 4, Real.pi / 4} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l3620_362021


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3620_362003

def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum :
  let a : ℚ := 1/4
  let r : ℚ := 1/4
  let n : ℕ := 6
  geometricSum a r n = 455/1365 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3620_362003


namespace NUMINAMATH_CALUDE_simple_interest_ratio_l3620_362041

/-- Simple interest calculation --/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- The ratio of final amount to initial amount after simple interest --/
def final_to_initial_ratio (rate : ℝ) (time : ℝ) : ℝ :=
  1 + rate * time

theorem simple_interest_ratio :
  let rate : ℝ := 0.1
  let time : ℝ := 10
  final_to_initial_ratio rate time = 2 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_ratio_l3620_362041


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3620_362029

/-- The partial fraction decomposition equation holds for the given values of A, B, and C -/
theorem partial_fraction_decomposition :
  ∃! (A B C : ℝ), ∀ (x : ℝ), x ≠ 0 → x ≠ 1 → x ≠ -1 →
    (-2 * x^2 + 5 * x - 7) / (x^3 - x) = A / x + (B * x + C) / (x^2 - 1) ∧
    A = 7 ∧ B = -9 ∧ C = 5 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3620_362029


namespace NUMINAMATH_CALUDE_jerry_action_figures_count_l3620_362050

/-- Calculates the total number of action figures on Jerry's shelf --/
def total_action_figures (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem stating that the total number of action figures is the sum of initial and added figures --/
theorem jerry_action_figures_count (initial : ℕ) (added : ℕ) :
  total_action_figures initial added = initial + added :=
by sorry

end NUMINAMATH_CALUDE_jerry_action_figures_count_l3620_362050


namespace NUMINAMATH_CALUDE_spinner_probability_l3620_362097

theorem spinner_probability (p_A p_B p_C p_D p_E : ℝ) : 
  p_A = 3/8 →
  p_B = 1/4 →
  p_C = p_D →
  p_C = p_E →
  p_A + p_B + p_C + p_D + p_E = 1 →
  p_C = 1/8 := by
sorry

end NUMINAMATH_CALUDE_spinner_probability_l3620_362097


namespace NUMINAMATH_CALUDE_farm_tax_percentage_l3620_362039

/-- Given a village's farm tax collection and information about Mr. Willam's tax and land,
    prove that the percentage of cultivated land taxed is 12.5%. -/
theorem farm_tax_percentage (total_tax village_tax willam_tax : ℚ) (willam_land_percentage : ℚ) :
  total_tax = 4000 →
  willam_tax = 500 →
  willam_land_percentage = 20833333333333332 / 100000000000000000 →
  (willam_tax / total_tax) * 100 = 125 / 10 :=
by sorry

end NUMINAMATH_CALUDE_farm_tax_percentage_l3620_362039


namespace NUMINAMATH_CALUDE_exam_candidates_l3620_362012

theorem exam_candidates (average_marks : ℝ) (total_marks : ℝ) (h1 : average_marks = 35) (h2 : total_marks = 4200) :
  total_marks / average_marks = 120 := by
  sorry

end NUMINAMATH_CALUDE_exam_candidates_l3620_362012


namespace NUMINAMATH_CALUDE_omega_range_l3620_362030

theorem omega_range (ω : ℝ) (h1 : ω > 1/4) :
  let f : ℝ → ℝ := λ x ↦ Real.sin (ω * x) - Real.cos (ω * x)
  let symmetry_axis (k : ℤ) : ℝ := (3*π/4 + k*π) / ω
  (∀ k : ℤ, symmetry_axis k ∉ Set.Ioo (2*π) (3*π)) →
  ω ∈ Set.Icc (3/8) (7/12) ∪ Set.Icc (7/8) (11/12) :=
by sorry

end NUMINAMATH_CALUDE_omega_range_l3620_362030


namespace NUMINAMATH_CALUDE_school_community_count_l3620_362025

theorem school_community_count (total_boys : ℕ) (muslim_percent hindu_percent sikh_percent : ℚ) :
  total_boys = 850 →
  muslim_percent = 40 / 100 →
  hindu_percent = 28 / 100 →
  sikh_percent = 10 / 100 →
  (total_boys : ℚ) * (1 - (muslim_percent + hindu_percent + sikh_percent)) = 187 := by
  sorry

end NUMINAMATH_CALUDE_school_community_count_l3620_362025


namespace NUMINAMATH_CALUDE_halfway_fraction_l3620_362085

theorem halfway_fraction (a b : ℚ) (ha : a = 1/4) (hb : b = 1/6) :
  (a + b) / 2 = 5/24 := by
  sorry

end NUMINAMATH_CALUDE_halfway_fraction_l3620_362085


namespace NUMINAMATH_CALUDE_infinitely_many_a_making_n4_plus_a_composite_l3620_362058

theorem infinitely_many_a_making_n4_plus_a_composite :
  ∀ k : ℕ, k > 1 → ∃ a : ℕ, a = 4 * k^4 ∧ ∀ n : ℕ, ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ n^4 + a = x * y :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_a_making_n4_plus_a_composite_l3620_362058


namespace NUMINAMATH_CALUDE_ratio_problem_l3620_362056

theorem ratio_problem (a b : ℚ) (h1 : b / a = 5) (h2 : b = 18 - 3 * a) : a = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3620_362056


namespace NUMINAMATH_CALUDE_brainiac_survey_l3620_362036

theorem brainiac_survey (R M : ℕ) : 
  R = 2 * M →                   -- Twice as many like rebus as math
  18 ≤ M ∧ 18 ≤ R →             -- 18 like both rebus and math
  20 ≤ M →                      -- 20 like math but not rebus
  R + M - 18 + 4 = 100          -- Total surveyed is 100
  := by sorry

end NUMINAMATH_CALUDE_brainiac_survey_l3620_362036


namespace NUMINAMATH_CALUDE_total_distance_traveled_l3620_362001

theorem total_distance_traveled (XY XZ : ℝ) (h1 : XY = 4500) (h2 : XZ = 4000) : 
  XY + Real.sqrt (XY^2 - XZ^2) + XZ = 10562 := by
sorry

end NUMINAMATH_CALUDE_total_distance_traveled_l3620_362001


namespace NUMINAMATH_CALUDE_prime_divisibility_l3620_362007

theorem prime_divisibility (p m n : ℕ) : 
  Prime p → 
  p > 2 → 
  m > 1 → 
  n > 0 → 
  Prime ((m^(p*n) - 1) / (m^n - 1)) → 
  (p * n) ∣ ((p - 1)^n + 1) :=
by sorry

end NUMINAMATH_CALUDE_prime_divisibility_l3620_362007


namespace NUMINAMATH_CALUDE_f_derivative_at_one_l3620_362062

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem f_derivative_at_one :
  (deriv f) 1 = 2 * Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_f_derivative_at_one_l3620_362062


namespace NUMINAMATH_CALUDE_fourth_month_sale_is_7230_l3620_362016

/-- Represents the sales data for a grocer over 6 months -/
structure GrocerSales where
  month1 : ℕ
  month2 : ℕ
  month3 : ℕ
  month5 : ℕ
  month6 : ℕ
  average : ℕ

/-- Calculates the sale in the fourth month given the sales data -/
def fourthMonthSale (sales : GrocerSales) : ℕ :=
  sales.average * 6 - (sales.month1 + sales.month2 + sales.month3 + sales.month5 + sales.month6)

/-- Theorem stating that the fourth month sale is 7230 given the provided sales data -/
theorem fourth_month_sale_is_7230 (sales : GrocerSales) 
  (h1 : sales.month1 = 6435)
  (h2 : sales.month2 = 6927)
  (h3 : sales.month3 = 6855)
  (h5 : sales.month5 = 6562)
  (h6 : sales.month6 = 6791)
  (ha : sales.average = 6800) :
  fourthMonthSale sales = 7230 := by
  sorry

#eval fourthMonthSale {
  month1 := 6435,
  month2 := 6927,
  month3 := 6855,
  month5 := 6562,
  month6 := 6791,
  average := 6800
}

end NUMINAMATH_CALUDE_fourth_month_sale_is_7230_l3620_362016


namespace NUMINAMATH_CALUDE_average_weight_b_c_l3620_362081

theorem average_weight_b_c (a b c : ℝ) : 
  (a + b + c) / 3 = 30 → 
  (a + b) / 2 = 25 → 
  b = 16 → 
  (b + c) / 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_b_c_l3620_362081


namespace NUMINAMATH_CALUDE_train_crossing_time_l3620_362065

theorem train_crossing_time (length : ℝ) (time_second : ℝ) (crossing_time : ℝ) : 
  length = 120 →
  time_second = 12 →
  crossing_time = 10.909090909090908 →
  (length / (length / time_second + (2 * length) / crossing_time - length / time_second)) = 10 :=
by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l3620_362065


namespace NUMINAMATH_CALUDE_linear_function_translation_l3620_362037

/-- Given a linear function y = 2x, when translated 3 units to the right along the x-axis,
    the resulting function is y = 2x - 6. -/
theorem linear_function_translation (x y : ℝ) :
  (y = 2 * x) →
  (y = 2 * (x - 3)) →
  (y = 2 * x - 6) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_translation_l3620_362037


namespace NUMINAMATH_CALUDE_evaluate_expression_1_evaluate_expression_2_l3620_362049

-- Question 1
theorem evaluate_expression_1 :
  (3 * Real.sqrt 27 - 2 * Real.sqrt 12) * (2 * Real.sqrt (5 + 1/3) + 3 * Real.sqrt (8 + 1/3)) = 115 := by
  sorry

-- Question 2
theorem evaluate_expression_2 :
  (5 * Real.sqrt 21 - 3 * Real.sqrt 15) / (5 * Real.sqrt (2 + 2/3) - 3 * Real.sqrt (1 + 2/3)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_1_evaluate_expression_2_l3620_362049


namespace NUMINAMATH_CALUDE_polynomial_product_l3620_362088

-- Define the polynomials
def P (x : ℝ) : ℝ := 2 * x^3 - 4 * x^2 + 3 * x
def Q (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3

-- State the theorem
theorem polynomial_product :
  ∀ x : ℝ, P x * Q x = 4 * x^7 - 2 * x^6 - 6 * x^5 + 9 * x^4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_l3620_362088


namespace NUMINAMATH_CALUDE_correct_division_result_l3620_362079

theorem correct_division_result (incorrect_divisor incorrect_quotient correct_divisor : ℕ) 
  (h1 : incorrect_divisor = 72)
  (h2 : incorrect_quotient = 24)
  (h3 : correct_divisor = 36) :
  (incorrect_divisor * incorrect_quotient) / correct_divisor = 48 :=
by sorry

end NUMINAMATH_CALUDE_correct_division_result_l3620_362079
