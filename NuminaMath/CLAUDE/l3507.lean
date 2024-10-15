import Mathlib

namespace NUMINAMATH_CALUDE_figure_to_square_l3507_350704

/-- Represents a figure on a grid --/
structure GridFigure where
  width : ℕ
  height : ℕ
  area : ℕ

/-- Represents a part of the figure after cutting --/
structure FigurePart where
  area : ℕ

/-- Represents a square --/
structure Square where
  side_length : ℕ

/-- Function to cut the figure into parts --/
def cut_figure (f : GridFigure) (n : ℕ) : List FigurePart :=
  sorry

/-- Function to check if parts can form a square --/
def can_form_square (parts : List FigurePart) : Bool :=
  sorry

/-- Main theorem statement --/
theorem figure_to_square (f : GridFigure) 
  (h1 : f.width = 6) 
  (h2 : f.height = 6) 
  (h3 : f.area = 36) : 
  ∃ (parts : List FigurePart), 
    (parts.length = 4) ∧ 
    (∀ p ∈ parts, p.area = 9) ∧
    (can_form_square parts = true) :=
  sorry

end NUMINAMATH_CALUDE_figure_to_square_l3507_350704


namespace NUMINAMATH_CALUDE_circle_and_tangent_line_l3507_350715

-- Define the vertices of the triangle and point P
def A : ℝ × ℝ := (8, 0)
def B : ℝ × ℝ := (0, 6)
def O : ℝ × ℝ := (0, 0)
def P : ℝ × ℝ := (-1, 5)

-- Define the circumcircle equation
def is_circumcircle_equation (eq : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, eq x y ↔ (x - 4)^2 + (y - 3)^2 = 25

-- Define the tangent line equation
def is_tangent_line_equation (eq : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, eq x y ↔ (x = -1 ∨ 21*x - 20*y + 121 = 0)

-- Theorem statement
theorem circle_and_tangent_line (eq_circle eq_line : ℝ → ℝ → Prop) : 
  is_circumcircle_equation eq_circle ∧ is_tangent_line_equation eq_line :=
sorry

end NUMINAMATH_CALUDE_circle_and_tangent_line_l3507_350715


namespace NUMINAMATH_CALUDE_inverse_of_P_l3507_350748

def P (a : ℕ) : Prop := Odd a → Prime a

theorem inverse_of_P : 
  (∀ a : ℕ, P a) ↔ (∀ a : ℕ, Prime a → Odd a) :=
sorry

end NUMINAMATH_CALUDE_inverse_of_P_l3507_350748


namespace NUMINAMATH_CALUDE_work_completion_time_l3507_350780

/-- Given:
  * A can finish a work in x days
  * B can finish the same work in x/2 days
  * A and B together can finish half the work in 1 day
Prove that x = 6 -/
theorem work_completion_time (x : ℝ) 
  (hx : x > 0) 
  (hA : (1 : ℝ) / x = 1 / x) 
  (hB : (1 : ℝ) / (x/2) = 2 / x) 
  (hAB : (1 : ℝ) / x + 2 / x = 1 / 2) : 
  x = 6 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l3507_350780


namespace NUMINAMATH_CALUDE_parabola_intercept_problem_l3507_350785

/-- Given two parabolas with specific properties, prove that h = 36 -/
theorem parabola_intercept_problem :
  ∀ (h j k : ℤ),
  (∀ x, ∃ y, y = 3 * (x - h)^2 + j) →
  (∀ x, ∃ y, y = 2 * (x - h)^2 + k) →
  (3 * h^2 + j = 2013) →
  (2 * h^2 + k = 2014) →
  (∃ x1 x2 : ℤ, x1 > 0 ∧ x2 > 0 ∧ 3 * (x1 - h)^2 + j = 0 ∧ 3 * (x2 - h)^2 + j = 0) →
  (∃ x3 x4 : ℤ, x3 > 0 ∧ x4 > 0 ∧ 2 * (x3 - h)^2 + k = 0 ∧ 2 * (x4 - h)^2 + k = 0) →
  h = 36 :=
by sorry

end NUMINAMATH_CALUDE_parabola_intercept_problem_l3507_350785


namespace NUMINAMATH_CALUDE_sin_product_equality_l3507_350746

theorem sin_product_equality : 
  Real.sin (3 * π / 180) * Real.sin (39 * π / 180) * Real.sin (63 * π / 180) * Real.sin (75 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_equality_l3507_350746


namespace NUMINAMATH_CALUDE_deposit_calculation_l3507_350764

theorem deposit_calculation (total_price : ℝ) (deposit_percentage : ℝ) (remaining_amount : ℝ) :
  deposit_percentage = 0.1 →
  remaining_amount = 1170 →
  (1 - deposit_percentage) * total_price = remaining_amount →
  deposit_percentage * total_price = 130 := by
  sorry

end NUMINAMATH_CALUDE_deposit_calculation_l3507_350764


namespace NUMINAMATH_CALUDE_find_m_l3507_350792

noncomputable def f (x m c : ℝ) : ℝ :=
  if x < m then c / Real.sqrt x else c / Real.sqrt m

theorem find_m : ∃ m : ℝ, 
  (∃ c : ℝ, f 4 m c = 30 ∧ f m m c = 15) → m = 16 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l3507_350792


namespace NUMINAMATH_CALUDE_y_coordinate_of_first_point_l3507_350722

/-- Given a line with equation x = 2y + 5 passing through points (m, n) and (m + 5, n + 2.5),
    prove that the y-coordinate of the first point (n) is equal to (m - 5)/2. -/
theorem y_coordinate_of_first_point 
  (m n : ℝ) 
  (h1 : m = 2 * n + 5) 
  (h2 : m + 5 = 2 * (n + 2.5) + 5) : 
  n = (m - 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_y_coordinate_of_first_point_l3507_350722


namespace NUMINAMATH_CALUDE_museum_groups_l3507_350789

/-- Given a class trip to a museum with the following conditions:
  * The class has 18 students in total
  * Each student takes 4 minutes to go through the museum
  * It takes each group 24 minutes to go through the museum
  Prove that the number of groups Howard split the class into is 3 -/
theorem museum_groups (total_students : ℕ) (student_time : ℕ) (group_time : ℕ)
  (h1 : total_students = 18)
  (h2 : student_time = 4)
  (h3 : group_time = 24) :
  total_students / (group_time / student_time) = 3 :=
sorry

end NUMINAMATH_CALUDE_museum_groups_l3507_350789


namespace NUMINAMATH_CALUDE_trajectory_is_ellipse_trajectory_is_ellipse_proof_l3507_350762

/-- The set of points P satisfying the condition that |F₁F₂| is the arithmetic mean of |PF₁| and |PF₂|, 
    where F₁(-1,0) and F₂(1,0) are fixed points, forms an ellipse. -/
theorem trajectory_is_ellipse (P : ℝ × ℝ) : Prop :=
  let F₁ : ℝ × ℝ := (-1, 0)
  let F₂ : ℝ × ℝ := (1, 0)
  let d₁ := dist P F₁
  let d₂ := dist P F₂
  (dist F₁ F₂ = (d₁ + d₂) / 2) → is_in_ellipse P F₁ F₂
  where
    dist : ℝ × ℝ → ℝ × ℝ → ℝ := λ (x₁, y₁) (x₂, y₂) => Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)
    is_in_ellipse : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ → Prop := sorry

theorem trajectory_is_ellipse_proof : ∀ P, trajectory_is_ellipse P := by sorry

end NUMINAMATH_CALUDE_trajectory_is_ellipse_trajectory_is_ellipse_proof_l3507_350762


namespace NUMINAMATH_CALUDE_fraction_equivalence_l3507_350724

theorem fraction_equivalence : 
  ∀ (n : ℚ), (4 + n) / (7 + n) = 7 / 9 ↔ n = 13 / 2 := by sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l3507_350724


namespace NUMINAMATH_CALUDE_factorization_problem_1_l3507_350747

theorem factorization_problem_1 (a : ℝ) : -2*a^2 + 4*a = -2*a*(a - 2) := by sorry

end NUMINAMATH_CALUDE_factorization_problem_1_l3507_350747


namespace NUMINAMATH_CALUDE_no_triple_prime_l3507_350718

theorem no_triple_prime (p : ℕ) : ¬(Nat.Prime p ∧ Nat.Prime (p^2 + 4) ∧ Nat.Prime (p^2 + 6)) := by
  sorry

end NUMINAMATH_CALUDE_no_triple_prime_l3507_350718


namespace NUMINAMATH_CALUDE_exam_average_l3507_350791

theorem exam_average (group1_count : ℕ) (group1_avg : ℚ) 
                      (group2_count : ℕ) (group2_avg : ℚ) : 
  group1_count = 15 →
  group1_avg = 70 / 100 →
  group2_count = 10 →
  group2_avg = 90 / 100 →
  (group1_count * group1_avg + group2_count * group2_avg) / (group1_count + group2_count) = 78 / 100 := by
  sorry

end NUMINAMATH_CALUDE_exam_average_l3507_350791


namespace NUMINAMATH_CALUDE_outdoor_scouts_hike_l3507_350730

theorem outdoor_scouts_hike (cars taxis vans buses : ℕ) 
  (people_per_car people_per_taxi people_per_van people_per_bus : ℕ) :
  cars = 5 →
  taxis = 8 →
  vans = 3 →
  buses = 2 →
  people_per_car = 4 →
  people_per_taxi = 6 →
  people_per_van = 5 →
  people_per_bus = 20 →
  cars * people_per_car + taxis * people_per_taxi + vans * people_per_van + buses * people_per_bus = 123 :=
by
  sorry

#check outdoor_scouts_hike

end NUMINAMATH_CALUDE_outdoor_scouts_hike_l3507_350730


namespace NUMINAMATH_CALUDE_negation_empty_subset_any_set_l3507_350774

theorem negation_empty_subset_any_set :
  (¬ ∀ A : Set α, ∅ ⊆ A) ↔ (∃ A : Set α, ¬(∅ ⊆ A)) :=
by sorry

end NUMINAMATH_CALUDE_negation_empty_subset_any_set_l3507_350774


namespace NUMINAMATH_CALUDE_circular_field_diameter_l3507_350709

/-- The diameter of a circular field given the fencing cost per meter and total fencing cost -/
theorem circular_field_diameter 
  (cost_per_meter : ℝ) 
  (total_cost : ℝ) 
  (h_cost : cost_per_meter = 3) 
  (h_total : total_cost = 395.84067435231395) : 
  ∃ (diameter : ℝ), abs (diameter - 42) < 0.00001 := by
  sorry

end NUMINAMATH_CALUDE_circular_field_diameter_l3507_350709


namespace NUMINAMATH_CALUDE_base_prime_repr_360_l3507_350742

/-- Base prime representation of a natural number -/
def base_prime_repr (n : ℕ) : ℕ := sorry

/-- Prime factorization of a natural number -/
def prime_factorization (n : ℕ) : List (ℕ × ℕ) := sorry

theorem base_prime_repr_360 :
  base_prime_repr 360 = 321 := by sorry

end NUMINAMATH_CALUDE_base_prime_repr_360_l3507_350742


namespace NUMINAMATH_CALUDE_item_sale_ratio_l3507_350767

theorem item_sale_ratio (c x y : ℝ) (hx : x = 0.8 * c) (hy : y = 1.25 * c) : y / x = 25 / 16 := by
  sorry

end NUMINAMATH_CALUDE_item_sale_ratio_l3507_350767


namespace NUMINAMATH_CALUDE_dodge_to_hyundai_ratio_l3507_350799

/-- Given a car dealership with the following conditions:
  - Total number of vehicles is 400
  - Number of Kia vehicles is 100
  - Number of Hyundai vehicles is half the number of Dodge vehicles
Prove that the ratio of Dodge to Hyundai vehicles is 2:1 -/
theorem dodge_to_hyundai_ratio 
  (total : ℕ) 
  (kia : ℕ) 
  (dodge : ℕ) 
  (hyundai : ℕ) 
  (h1 : total = 400)
  (h2 : kia = 100)
  (h3 : hyundai = dodge / 2)
  (h4 : total = dodge + hyundai + kia) :
  dodge / hyundai = 2 := by
  sorry

end NUMINAMATH_CALUDE_dodge_to_hyundai_ratio_l3507_350799


namespace NUMINAMATH_CALUDE_solution_of_system_l3507_350749

theorem solution_of_system (x y : ℝ) : 
  (1 / Real.sqrt (1 + 2 * x^2) + 1 / Real.sqrt (1 + 2 * y^2) = 2 / Real.sqrt (1 + 2 * x * y)) ∧
  (Real.sqrt (x * (1 - 2 * x)) + Real.sqrt (y * (1 - 2 * y)) = 2 / 9) →
  (x = y) ∧ 
  ((x = 1 / 4 + Real.sqrt 73 / 36) ∨ (x = 1 / 4 - Real.sqrt 73 / 36)) :=
by sorry

end NUMINAMATH_CALUDE_solution_of_system_l3507_350749


namespace NUMINAMATH_CALUDE_min_value_sum_l3507_350716

theorem min_value_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + 1/y = 1) :
  x + 2*y ≥ 8 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_l3507_350716


namespace NUMINAMATH_CALUDE_log_expression_eval_l3507_350717

-- Define lg as base 10 logarithm
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define the fourth root
noncomputable def fourthRoot (x : ℝ) := Real.rpow x (1/4)

theorem log_expression_eval :
  Real.log (fourthRoot 27 / 3) / Real.log 3 + lg 25 + lg 4 + 7^(Real.log 2 / Real.log 7) = 15/4 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_eval_l3507_350717


namespace NUMINAMATH_CALUDE_population_growth_l3507_350738

theorem population_growth (initial_population : ℝ) :
  let growth_factor1 := 1 + 0.05
  let growth_factor2 := 1 + 0.10
  let growth_factor3 := 1 + 0.15
  let final_population := initial_population * growth_factor1 * growth_factor2 * growth_factor3
  (final_population - initial_population) / initial_population * 100 = 33.075 := by
sorry

end NUMINAMATH_CALUDE_population_growth_l3507_350738


namespace NUMINAMATH_CALUDE_loan_principal_calculation_l3507_350745

/-- Calculates the total interest paid on a loan with varying interest rates over different periods. -/
def total_interest (principal : ℝ) : ℝ :=
  principal * (0.08 * 4 + 0.10 * 6 + 0.12 * 5)

/-- Proves that for the given interest rates and periods, a principal of 8000 results in a total interest of 12160. -/
theorem loan_principal_calculation :
  ∃ (principal : ℝ), principal > 0 ∧ total_interest principal = 12160 ∧ principal = 8000 :=
by
  sorry

#eval total_interest 8000

end NUMINAMATH_CALUDE_loan_principal_calculation_l3507_350745


namespace NUMINAMATH_CALUDE_triangle_problem_l3507_350759

noncomputable section

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem -/
theorem triangle_problem (t : Triangle) :
  t.b * sin (2 * t.A) - t.a * sin (t.A + t.C) = 0 →
  t.a = 3 →
  (1 / 2) * t.b * t.c * sin t.A = (3 * sqrt 3) / 2 →
  t.A = π / 3 ∧ 1 / t.b + 1 / t.c = sqrt 3 / 2 := by
  sorry

end

end NUMINAMATH_CALUDE_triangle_problem_l3507_350759


namespace NUMINAMATH_CALUDE_probability_triangle_or_hexagon_l3507_350769

theorem probability_triangle_or_hexagon :
  let total_figures : ℕ := 10
  let triangles : ℕ := 3
  let squares : ℕ := 4
  let circles : ℕ := 2
  let hexagons : ℕ := 1
  let target_figures := triangles + hexagons
  (target_figures : ℚ) / total_figures = 2 / 5 :=
by sorry

end NUMINAMATH_CALUDE_probability_triangle_or_hexagon_l3507_350769


namespace NUMINAMATH_CALUDE_square_sum_value_l3507_350794

theorem square_sum_value (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : x * y + x + y = 71) (h2 : x^2 * y + x * y^2 = 880) :
  x^2 + y^2 = 146 := by
sorry

end NUMINAMATH_CALUDE_square_sum_value_l3507_350794


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3507_350711

theorem quadratic_factorization (x : ℝ) : 16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3507_350711


namespace NUMINAMATH_CALUDE_line_intercept_ratio_l3507_350775

/-- Given two lines with y-intercept 5, where the first line has slope 2 and x-intercept (s, 0),
    and the second line has slope 7 and x-intercept (t, 0), prove that s/t = 7/2 -/
theorem line_intercept_ratio (s t : ℝ) : 
  (5 : ℝ) = 2 * s + 5 ∧ (5 : ℝ) = 7 * t + 5 → s / t = 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_line_intercept_ratio_l3507_350775


namespace NUMINAMATH_CALUDE_fifth_basket_price_l3507_350725

/-- Given 4 baskets with an average cost of $4 and a fifth basket that makes
    the average cost of all 5 baskets $4.8, the price of the fifth basket is $8. -/
theorem fifth_basket_price (num_initial_baskets : Nat) (initial_avg_cost : ℝ)
    (total_baskets : Nat) (new_avg_cost : ℝ) :
    num_initial_baskets = 4 →
    initial_avg_cost = 4 →
    total_baskets = 5 →
    new_avg_cost = 4.8 →
    (total_baskets * new_avg_cost - num_initial_baskets * initial_avg_cost) = 8 := by
  sorry

#check fifth_basket_price

end NUMINAMATH_CALUDE_fifth_basket_price_l3507_350725


namespace NUMINAMATH_CALUDE_brownies_left_l3507_350741

/-- Calculates the number of brownies left after consumption by Tina, her husband, and dinner guests. -/
theorem brownies_left (total : ℝ) (tina_lunch : ℝ) (tina_dinner : ℝ) (husband : ℝ) (guests : ℝ) 
  (days : ℕ) (guest_days : ℕ) : 
  total = 24 ∧ 
  tina_lunch = 1.5 ∧ 
  tina_dinner = 0.5 ∧ 
  husband = 0.75 ∧ 
  guests = 2.5 ∧ 
  days = 5 ∧ 
  guest_days = 2 → 
  total - ((tina_lunch + tina_dinner) * days + husband * days + guests * guest_days) = 5.25 := by
  sorry

end NUMINAMATH_CALUDE_brownies_left_l3507_350741


namespace NUMINAMATH_CALUDE_professor_newtons_students_l3507_350782

theorem professor_newtons_students (N M : ℕ) : 
  N % 4 = 2 →
  N % 5 = 1 →
  N = M + 15 →
  M < 15 →
  N = 26 ∧ M = 11 := by
sorry

end NUMINAMATH_CALUDE_professor_newtons_students_l3507_350782


namespace NUMINAMATH_CALUDE_conditional_prob_B_given_A_l3507_350731

/-- A fair coin is a coin with probability 1/2 for both heads and tails -/
structure FairCoin where
  prob_heads : ℝ
  prob_tails : ℝ
  fair_heads : prob_heads = 1/2
  fair_tails : prob_tails = 1/2

/-- Event A: "the first appearance of heads" when a coin is tossed twice -/
def event_A (c : FairCoin) : ℝ := c.prob_heads

/-- Event B: "the second appearance of tails" -/
def event_B (c : FairCoin) : ℝ := c.prob_tails

/-- The probability of both events A and B occurring -/
def prob_AB (c : FairCoin) : ℝ := c.prob_heads * c.prob_tails

/-- Theorem: The conditional probability P(B|A) is 1/2 -/
theorem conditional_prob_B_given_A (c : FairCoin) : 
  prob_AB c / event_A c = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_conditional_prob_B_given_A_l3507_350731


namespace NUMINAMATH_CALUDE_quadratic_is_perfect_square_l3507_350712

theorem quadratic_is_perfect_square (a : ℝ) : 
  (∃ b c : ℝ, ∀ x : ℝ, 9*x^2 + 24*x + a = (b*x + c)^2) → a = 16 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_is_perfect_square_l3507_350712


namespace NUMINAMATH_CALUDE_max_value_implies_tan_2alpha_l3507_350757

/-- Given a function f(x) = 3sin(x) + cos(x) that attains its maximum value when x = α,
    prove that tan(2α) = -3/4 -/
theorem max_value_implies_tan_2alpha (α : ℝ) 
  (h : ∀ x, 3 * Real.sin x + Real.cos x ≤ 3 * Real.sin α + Real.cos α) : 
  Real.tan (2 * α) = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_implies_tan_2alpha_l3507_350757


namespace NUMINAMATH_CALUDE_sqrt_square_eq_abs_l3507_350708

theorem sqrt_square_eq_abs (x : ℝ) : Real.sqrt (x^2) = |x| := by sorry

end NUMINAMATH_CALUDE_sqrt_square_eq_abs_l3507_350708


namespace NUMINAMATH_CALUDE_unique_positive_cyclic_shift_l3507_350781

def CyclicShift (a : List ℤ) : List (List ℤ) :=
  List.range a.length |>.map (λ i => a.rotate i)

def PositivePartialSums (a : List ℤ) : Prop :=
  List.scanl (· + ·) 0 a |>.tail |>.all (λ x => x > 0)

theorem unique_positive_cyclic_shift
  (a : List ℤ)
  (h_sum : a.sum = 1) :
  ∃! shift, shift ∈ CyclicShift a ∧ PositivePartialSums shift :=
sorry

end NUMINAMATH_CALUDE_unique_positive_cyclic_shift_l3507_350781


namespace NUMINAMATH_CALUDE_mikes_lawn_mowing_earnings_l3507_350729

/-- Proves that Mike's total earnings from mowing lawns is $101 given the conditions --/
theorem mikes_lawn_mowing_earnings : 
  ∀ (total_earnings : ℕ) 
    (mower_blades_cost : ℕ) 
    (num_games : ℕ) 
    (game_cost : ℕ),
  mower_blades_cost = 47 →
  num_games = 9 →
  game_cost = 6 →
  total_earnings = mower_blades_cost + num_games * game_cost →
  total_earnings = 101 := by
sorry

end NUMINAMATH_CALUDE_mikes_lawn_mowing_earnings_l3507_350729


namespace NUMINAMATH_CALUDE_fourth_root_of_390625_l3507_350751

theorem fourth_root_of_390625 (x : ℝ) (h1 : x > 0) (h2 : x^4 = 390625) : x = 25 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_of_390625_l3507_350751


namespace NUMINAMATH_CALUDE_total_heads_calculation_l3507_350779

theorem total_heads_calculation (num_hens : ℕ) (total_feet : ℕ) : num_hens = 20 → total_feet = 200 → ∃ (num_cows : ℕ), num_hens + num_cows = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_heads_calculation_l3507_350779


namespace NUMINAMATH_CALUDE_multiply_98_squared_l3507_350743

theorem multiply_98_squared : 98 * 98 = 9604 := by
  sorry

end NUMINAMATH_CALUDE_multiply_98_squared_l3507_350743


namespace NUMINAMATH_CALUDE_decimal_multiplication_l3507_350753

theorem decimal_multiplication (a b : ℚ) (ha : a = 0.4) (hb : b = 0.75) : a * b = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_decimal_multiplication_l3507_350753


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_neg_two_a_range_when_f_leq_g_on_interval_l3507_350766

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |2*x - 1| + |2*x + a|
def g (x : ℝ) : ℝ := x + 3

-- Part 1
theorem solution_set_when_a_is_neg_two :
  {x : ℝ | f (-2) x < g x} = Set.Ioo 0 2 := by sorry

-- Part 2
theorem a_range_when_f_leq_g_on_interval :
  ∀ a : ℝ, a > -1 →
  (∀ x ∈ Set.Icc (-a/2) (1/2), f a x ≤ g x) →
  a ∈ Set.Ioo (-1) (4/3) := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_neg_two_a_range_when_f_leq_g_on_interval_l3507_350766


namespace NUMINAMATH_CALUDE_equation_has_real_roots_l3507_350784

theorem equation_has_real_roots (K : ℝ) : ∃ x : ℝ, x = K^3 * (x + 1) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_has_real_roots_l3507_350784


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3507_350763

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence satisfying the condition
    2(a_1 + a_3 + a_5) + 3(a_8 + a_10) = 36, prove that a_6 = 3. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_condition : 2 * (a 1 + a 3 + a 5) + 3 * (a 8 + a 10) = 36) :
  a 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3507_350763


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l3507_350761

theorem algebraic_expression_equality : 
  let a : ℝ := Real.sqrt 3 + 2
  (a - Real.sqrt 2) * (a + Real.sqrt 2) - a * (a - 3) = 3 * Real.sqrt 3 + 4 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l3507_350761


namespace NUMINAMATH_CALUDE_exists_large_number_with_invariant_prime_factors_l3507_350778

/-- A function that represents swapping two non-zero digits in a number's decimal representation -/
def swap_digits (n : ℕ) (i j : ℕ) : ℕ := sorry

/-- A function that returns the set of prime factors of a number -/
def prime_factors (n : ℕ) : Set ℕ := sorry

/-- Theorem stating the existence of a number with the required properties -/
theorem exists_large_number_with_invariant_prime_factors :
  ∃ n : ℕ, n > 10^1000 ∧ 
           n % 10 ≠ 0 ∧ 
           ∃ i j : ℕ, i ≠ j ∧ 
                     (swap_digits n i j) ≠ n ∧ 
                     prime_factors (swap_digits n i j) = prime_factors n :=
sorry

end NUMINAMATH_CALUDE_exists_large_number_with_invariant_prime_factors_l3507_350778


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3507_350783

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => x^2 - 4*x + 3
  (f 1 = 0) ∧ (f 3 = 0) ∧ (∀ x : ℝ, f x = 0 → x = 1 ∨ x = 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3507_350783


namespace NUMINAMATH_CALUDE_dice_roll_probability_l3507_350797

def probability_first_die : ℚ := 3 / 8
def probability_second_die : ℚ := 3 / 8

theorem dice_roll_probability : 
  probability_first_die * probability_second_die = 9 / 64 := by
  sorry

end NUMINAMATH_CALUDE_dice_roll_probability_l3507_350797


namespace NUMINAMATH_CALUDE_crayon_box_total_l3507_350755

/-- Represents the number of crayons of each color in a crayon box. -/
structure CrayonBox where
  red : ℕ
  blue : ℕ
  green : ℕ
  pink : ℕ

/-- The total number of crayons in the box. -/
def total_crayons (box : CrayonBox) : ℕ :=
  box.red + box.blue + box.green + box.pink

/-- Theorem stating the total number of crayons in the specific box configuration. -/
theorem crayon_box_total :
  ∃ (box : CrayonBox),
    box.red = 8 ∧
    box.blue = 6 ∧
    box.green = 2 * box.blue / 3 ∧
    box.pink = 6 ∧
    total_crayons box = 24 := by
  sorry

end NUMINAMATH_CALUDE_crayon_box_total_l3507_350755


namespace NUMINAMATH_CALUDE_calculate_expression_l3507_350732

theorem calculate_expression : -(-1) + 3^2 / (1 - 4) * 2 = -5 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3507_350732


namespace NUMINAMATH_CALUDE_geometric_progression_equality_l3507_350770

theorem geometric_progression_equality (x y z : ℝ) :
  (∃ r : ℝ, y = x * r ∧ z = y * r) ↔ (x^2 + y^2) * (y^2 + z^2) = (x*y + y*z)^2 :=
sorry

end NUMINAMATH_CALUDE_geometric_progression_equality_l3507_350770


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3507_350756

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (x - 1)}

-- Define set B
def B : Set ℝ := Set.range (λ x => 2 * x + 1)

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = Set.Ici 1 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3507_350756


namespace NUMINAMATH_CALUDE_alexis_isabella_shopping_ratio_l3507_350786

theorem alexis_isabella_shopping_ratio : 
  let alexis_pants : ℕ := 21
  let alexis_dresses : ℕ := 18
  let isabella_total : ℕ := 13
  (alexis_pants + alexis_dresses) / isabella_total = 3 :=
by sorry

end NUMINAMATH_CALUDE_alexis_isabella_shopping_ratio_l3507_350786


namespace NUMINAMATH_CALUDE_sin_sum_to_product_l3507_350705

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (3 * x) + Real.sin (5 * x) = 2 * Real.sin (4 * x) * Real.cos x := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_to_product_l3507_350705


namespace NUMINAMATH_CALUDE_planar_graph_inequality_l3507_350798

/-- A planar graph is a graph that can be embedded in the plane without edge crossings. -/
structure PlanarGraph where
  /-- The number of edges in the planar graph -/
  E : ℕ
  /-- The number of faces in the planar graph -/
  F : ℕ

/-- For any planar graph, twice the number of edges is greater than or equal to
    three times the number of faces. -/
theorem planar_graph_inequality (G : PlanarGraph) : 2 * G.E ≥ 3 * G.F := by
  sorry

end NUMINAMATH_CALUDE_planar_graph_inequality_l3507_350798


namespace NUMINAMATH_CALUDE_cubic_root_b_value_l3507_350714

theorem cubic_root_b_value (a b : ℚ) : 
  (∃ x : ℝ, x^3 + a*x^2 + b*x + 15 = 0 ∧ x = 3 + Real.sqrt 5) →
  b = -37/2 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_b_value_l3507_350714


namespace NUMINAMATH_CALUDE_grandfather_gift_is_100_l3507_350771

/-- The amount of money Amy's grandfather gave her --/
def grandfather_gift : ℕ := sorry

/-- The number of dolls Amy bought --/
def dolls_bought : ℕ := 3

/-- The cost of each doll in dollars --/
def doll_cost : ℕ := 1

/-- The amount of money Amy has left after buying the dolls --/
def money_left : ℕ := 97

/-- Theorem stating that the grandfather's gift is $100 --/
theorem grandfather_gift_is_100 :
  grandfather_gift = dolls_bought * doll_cost + money_left :=
by sorry

end NUMINAMATH_CALUDE_grandfather_gift_is_100_l3507_350771


namespace NUMINAMATH_CALUDE_sin_negative_690_degrees_l3507_350760

theorem sin_negative_690_degrees : Real.sin ((-690 : ℝ) * π / 180) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_negative_690_degrees_l3507_350760


namespace NUMINAMATH_CALUDE_blue_tiles_count_l3507_350772

/-- Given a pool that needs tiles, this theorem proves the number of blue tiles. -/
theorem blue_tiles_count 
  (total_needed : ℕ) 
  (additional_needed : ℕ) 
  (red_tiles : ℕ) 
  (h1 : total_needed = 100) 
  (h2 : additional_needed = 20) 
  (h3 : red_tiles = 32) : 
  total_needed - additional_needed - red_tiles = 48 := by
  sorry

end NUMINAMATH_CALUDE_blue_tiles_count_l3507_350772


namespace NUMINAMATH_CALUDE_production_average_l3507_350750

/-- Proves that given the conditions in the problem, n = 1 --/
theorem production_average (n : ℕ) : 
  (n * 50 + 60) / (n + 1) = 55 → n = 1 := by
  sorry


end NUMINAMATH_CALUDE_production_average_l3507_350750


namespace NUMINAMATH_CALUDE_z_range_difference_l3507_350719

theorem z_range_difference (x y z : ℝ) 
  (sum_eq : x + y + z = 2) 
  (prod_eq : x * y + y * z + x * z = 0) : 
  ∃ (a b : ℝ), (∀ z', (∃ x' y', x' + y' + z' = 2 ∧ x' * y' + y' * z' + x' * z' = 0) → a ≤ z' ∧ z' ≤ b) ∧ b - a = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_z_range_difference_l3507_350719


namespace NUMINAMATH_CALUDE_min_value_geometric_sequence_l3507_350723

/-- Given a geometric sequence with first term b₁ = 2, the minimum value of 3b₂ + 6b₃ is -3/4,
    where b₂ and b₃ are the second and third terms of the sequence respectively. -/
theorem min_value_geometric_sequence (b₁ b₂ b₃ : ℝ) : 
  b₁ = 2 → (∃ r : ℝ, b₂ = b₁ * r ∧ b₃ = b₂ * r) → 
  (∀ r : ℝ, b₂ = 2 * r → b₃ = 2 * r^2 → 3 * b₂ + 6 * b₃ ≥ -3/4) ∧ 
  (∃ r : ℝ, b₂ = 2 * r ∧ b₃ = 2 * r^2 ∧ 3 * b₂ + 6 * b₃ = -3/4) :=
by sorry

end NUMINAMATH_CALUDE_min_value_geometric_sequence_l3507_350723


namespace NUMINAMATH_CALUDE_right_triangle_condition_l3507_350790

theorem right_triangle_condition (α β γ : Real) : 
  α + β + γ = Real.pi →
  0 ≤ α ∧ α ≤ Real.pi →
  0 ≤ β ∧ β ≤ Real.pi →
  0 ≤ γ ∧ γ ≤ Real.pi →
  Real.sin γ - Real.cos α = Real.cos β →
  α = Real.pi / 2 ∨ β = Real.pi / 2 ∨ γ = Real.pi / 2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_condition_l3507_350790


namespace NUMINAMATH_CALUDE_simple_interest_time_l3507_350754

/-- Given simple interest, principal, and rate, calculate the time in years -/
theorem simple_interest_time (SI P R : ℚ) (h1 : SI = 4016.25) (h2 : P = 16065) (h3 : R = 5) :
  SI = P * R * 5 / 100 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_time_l3507_350754


namespace NUMINAMATH_CALUDE_prop1_prop4_l3507_350776

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (parallelLine : Line → Plane → Prop)
variable (skew : Line → Line → Prop)

-- Proposition 1
theorem prop1 (m : Line) (α β : Plane) :
  perpendicular m α → perpendicular m β → parallel α β := by sorry

-- Proposition 4
theorem prop4 (m n : Line) (α β : Plane) :
  skew m n → contains α m → parallelLine m β → contains β n → parallelLine n α → parallel α β := by sorry

end NUMINAMATH_CALUDE_prop1_prop4_l3507_350776


namespace NUMINAMATH_CALUDE_smallest_n_perfect_square_and_cube_l3507_350720

/-- A number is a perfect square if it's equal to some integer multiplied by itself. -/
def IsPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

/-- A number is a perfect cube if it's equal to some integer multiplied by itself twice. -/
def IsPerfectCube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m * m

/-- 45 is the smallest positive integer n such that 5n is a perfect square and 3n is a perfect cube. -/
theorem smallest_n_perfect_square_and_cube :
  (∀ n : ℕ, 0 < n ∧ n < 45 → ¬(IsPerfectSquare (5 * n) ∧ IsPerfectCube (3 * n))) ∧
  (IsPerfectSquare (5 * 45) ∧ IsPerfectCube (3 * 45)) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_perfect_square_and_cube_l3507_350720


namespace NUMINAMATH_CALUDE_equation_solution_l3507_350739

theorem equation_solution : 
  ∃ (x : ℚ), (x + 4) / (x - 3) = (x - 2) / (x + 2) ↔ x = -2/11 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3507_350739


namespace NUMINAMATH_CALUDE_expression_equals_one_l3507_350795

theorem expression_equals_one :
  (4 * 6) / (12 * 14) * (8 * 12 * 14) / (4 * 6 * 8) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l3507_350795


namespace NUMINAMATH_CALUDE_jogger_speed_l3507_350773

/-- The speed of a jogger on a path with specific conditions -/
theorem jogger_speed (inner_perimeter outer_perimeter : ℝ) 
  (h1 : outer_perimeter - inner_perimeter = 16 * Real.pi)
  (time_diff : ℝ) (h2 : time_diff = 60) :
  ∃ (speed : ℝ), speed = (4 * Real.pi) / 15 ∧ 
    outer_perimeter / speed = inner_perimeter / speed + time_diff :=
sorry

end NUMINAMATH_CALUDE_jogger_speed_l3507_350773


namespace NUMINAMATH_CALUDE_product_even_implies_one_even_one_odd_l3507_350726

/-- Represents a polynomial with integer coefficients -/
def IntPolynomial (n : ℕ) := Fin n → ℤ

/-- Checks if all coefficients of a polynomial are even -/
def allEven (p : IntPolynomial n) : Prop :=
  ∀ i, 2 ∣ p i

/-- Checks if at least one coefficient of a polynomial is odd -/
def hasOddCoeff (p : IntPolynomial n) : Prop :=
  ∃ i, ¬(2 ∣ p i)

/-- Represents the product of two polynomials -/
def polyProduct (a : IntPolynomial n) (b : IntPolynomial m) : IntPolynomial (n + m - 1) :=
  sorry

/-- Main theorem -/
theorem product_even_implies_one_even_one_odd
  (n m : ℕ)
  (a : IntPolynomial n)
  (b : IntPolynomial m)
  (h1 : allEven (polyProduct a b))
  (h2 : ∃ i, ¬(4 ∣ (polyProduct a b) i)) :
  (allEven a ∧ hasOddCoeff b) ∨ (allEven b ∧ hasOddCoeff a) :=
sorry

end NUMINAMATH_CALUDE_product_even_implies_one_even_one_odd_l3507_350726


namespace NUMINAMATH_CALUDE_negative_two_minus_six_l3507_350793

theorem negative_two_minus_six : -2 - 6 = -8 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_minus_six_l3507_350793


namespace NUMINAMATH_CALUDE_always_solution_never_solution_l3507_350700

-- Define the polynomial function
def f (a x : ℝ) : ℝ := (1 + a) * x^4 + x^3 - (3*a + 2) * x^2 - 4*a

-- Theorem 1: For all real a, x = -2 is a solution
theorem always_solution : ∀ a : ℝ, f a (-2) = 0 := by sorry

-- Theorem 2: For all real a, x = 2 is not a solution
theorem never_solution : ∀ a : ℝ, f a 2 ≠ 0 := by sorry

end NUMINAMATH_CALUDE_always_solution_never_solution_l3507_350700


namespace NUMINAMATH_CALUDE_kolya_is_collection_agency_l3507_350702

-- Define the actors in the scenario
structure Person :=
  (name : String)

-- Define the book lending scenario
structure BookLendingScenario :=
  (lender : Person)
  (borrower : Person)
  (collector : Person)
  (books_lent : ℕ)
  (return_promised : Bool)
  (books_returned : Bool)
  (collector_fee : ℕ)

-- Define the characteristics of a collection agency
structure CollectionAgency :=
  (collects_items : Bool)
  (acts_on_behalf : Bool)
  (receives_fee : Bool)

-- Define Kolya's role in the scenario
def kolya_role (scenario : BookLendingScenario) : CollectionAgency :=
  { collects_items := true
  , acts_on_behalf := true
  , receives_fee := scenario.collector_fee > 0 }

-- Theorem statement
theorem kolya_is_collection_agency (scenario : BookLendingScenario) : 
  kolya_role scenario = CollectionAgency.mk true true true :=
sorry

end NUMINAMATH_CALUDE_kolya_is_collection_agency_l3507_350702


namespace NUMINAMATH_CALUDE_cubic_expansion_property_l3507_350707

theorem cubic_expansion_property (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, (5*x + 4)^3 = a₀ + a₁*x + a₂*x^2 + a₃*x^3) →
  (a₀ + a₂) - (a₁ + a₃) = -1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expansion_property_l3507_350707


namespace NUMINAMATH_CALUDE_minimum_width_for_garden_l3507_350736

-- Define the garden width as a real number
variable (w : ℝ)

-- Define the conditions of the problem
def garden_length (w : ℝ) : ℝ := w + 10
def garden_area (w : ℝ) : ℝ := w * garden_length w
def area_constraint (w : ℝ) : Prop := garden_area w ≥ 150

-- Theorem statement
theorem minimum_width_for_garden :
  (∀ x : ℝ, x > 0 → area_constraint x → x ≥ 10) ∧ area_constraint 10 :=
sorry

end NUMINAMATH_CALUDE_minimum_width_for_garden_l3507_350736


namespace NUMINAMATH_CALUDE_fraction_equality_l3507_350744

theorem fraction_equality (P Q : ℤ) :
  (∀ x : ℝ, x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 3 →
    (P / (x + 3) + Q / (x^2 - 3*x) = (x^2 - x + 8) / (x^3 + x^2 - 9*x))) →
  Q / P = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3507_350744


namespace NUMINAMATH_CALUDE_stamps_total_proof_l3507_350740

/-- The number of stamps Lizette has -/
def lizette_stamps : ℕ := 813

/-- The number of stamps Lizette has more than Minerva -/
def lizette_minerva_diff : ℕ := 125

/-- The number of stamps Jermaine has more than Lizette -/
def jermaine_lizette_diff : ℕ := 217

/-- The total number of stamps Minerva, Lizette, and Jermaine have -/
def total_stamps : ℕ := lizette_stamps + (lizette_stamps - lizette_minerva_diff) + (lizette_stamps + jermaine_lizette_diff)

theorem stamps_total_proof : total_stamps = 2531 := by
  sorry

end NUMINAMATH_CALUDE_stamps_total_proof_l3507_350740


namespace NUMINAMATH_CALUDE_sqrt_27_div_sqrt_3_eq_3_l3507_350752

theorem sqrt_27_div_sqrt_3_eq_3 : Real.sqrt 27 / Real.sqrt 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_27_div_sqrt_3_eq_3_l3507_350752


namespace NUMINAMATH_CALUDE_prob_even_modified_die_l3507_350735

/-- Represents a standard 6-sided die -/
def StandardDie : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- Total number of dots on a standard die -/
def TotalDots : ℕ := (StandardDie.sum id)

/-- Probability of a specific dot not being removed in one attempt -/
def ProbNotRemoved : ℚ := (TotalDots - 1) / TotalDots

/-- Probability of a specific dot not being removed in two attempts -/
def ProbNotRemovedTwice : ℚ := ProbNotRemoved ^ 2

/-- Probability of a specific dot being removed in two attempts -/
def ProbRemovedTwice : ℚ := 1 - ProbNotRemovedTwice

/-- Probability of losing exactly one dot from a face with n dots -/
def ProbLoseOneDot (n : ℕ) : ℚ := 2 * (n / TotalDots) * ProbNotRemoved

/-- Probability of a face with n dots remaining even after dot removal -/
def ProbRemainsEven (n : ℕ) : ℚ :=
  if n % 2 = 0
  then ProbNotRemovedTwice + (if n ≥ 2 then ProbRemovedTwice else 0)
  else ProbLoseOneDot n

/-- Theorem: The probability of rolling an even number on the modified die -/
theorem prob_even_modified_die :
  (StandardDie.sum (λ n => (1 : ℚ) / 6 * ProbRemainsEven n)) =
  (StandardDie.sum (λ n => if n % 2 = 0 then (1 : ℚ) / 6 * ProbNotRemovedTwice else 0)) +
  (StandardDie.sum (λ n => if n % 2 = 0 ∧ n ≥ 2 then (1 : ℚ) / 6 * ProbRemovedTwice else 0)) +
  (StandardDie.sum (λ n => if n % 2 = 1 then (1 : ℚ) / 6 * ProbLoseOneDot n else 0)) :=
by sorry


end NUMINAMATH_CALUDE_prob_even_modified_die_l3507_350735


namespace NUMINAMATH_CALUDE_projection_theorem_l3507_350727

def v : Fin 2 → ℝ := ![6, -3]
def u : Fin 2 → ℝ := ![3, 0]

theorem projection_theorem :
  (((v • u) / (u • u)) • u) = ![6, 0] := by
  sorry

end NUMINAMATH_CALUDE_projection_theorem_l3507_350727


namespace NUMINAMATH_CALUDE_ceiling_sum_sqrt_l3507_350765

theorem ceiling_sum_sqrt : ⌈Real.sqrt 19⌉ + ⌈Real.sqrt 57⌉ + ⌈Real.sqrt 119⌉ = 24 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sum_sqrt_l3507_350765


namespace NUMINAMATH_CALUDE_gcf_36_54_l3507_350706

theorem gcf_36_54 : Nat.gcd 36 54 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcf_36_54_l3507_350706


namespace NUMINAMATH_CALUDE_cow_spots_l3507_350787

/-- Calculates the total number of spots on a cow given the number of spots on its left side. -/
def totalSpots (leftSpots : ℕ) : ℕ :=
  let rightSpots := 3 * leftSpots + 7
  leftSpots + rightSpots

/-- Theorem stating that a cow with 16 spots on its left side has 71 spots in total. -/
theorem cow_spots : totalSpots 16 = 71 := by
  sorry

end NUMINAMATH_CALUDE_cow_spots_l3507_350787


namespace NUMINAMATH_CALUDE_power_division_rule_l3507_350703

theorem power_division_rule (x : ℝ) : x^4 / x = x^3 := by sorry

end NUMINAMATH_CALUDE_power_division_rule_l3507_350703


namespace NUMINAMATH_CALUDE_circle_line_distance_range_l3507_350713

theorem circle_line_distance_range (b : ℝ) :
  (∃! (p q : ℝ × ℝ), 
    ((p.1 - 1)^2 + (p.2 - 1)^2 = 4) ∧
    ((q.1 - 1)^2 + (q.2 - 1)^2 = 4) ∧
    (p ≠ q) ∧
    (|p.2 - (p.1 + b)| / Real.sqrt 2 = 1) ∧
    (|q.2 - (q.1 + b)| / Real.sqrt 2 = 1)) →
  b ∈ Set.Ioo (-3 * Real.sqrt 2) (-Real.sqrt 2) ∪ Set.Ioo (Real.sqrt 2) (3 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_line_distance_range_l3507_350713


namespace NUMINAMATH_CALUDE_monotone_decreasing_iff_b_positive_l3507_350721

/-- The function f(x) = (ax + b) / x is monotonically decreasing on (0, +∞) if and only if b > 0 -/
theorem monotone_decreasing_iff_b_positive (a b : ℝ) :
  (∀ x y : ℝ, 0 < x → 0 < y → x < y → (a * x + b) / x > (a * y + b) / y) ↔ b > 0 := by
  sorry

end NUMINAMATH_CALUDE_monotone_decreasing_iff_b_positive_l3507_350721


namespace NUMINAMATH_CALUDE_max_last_place_wins_theorem_l3507_350788

/-- Represents a baseball league -/
structure BaseballLeague where
  teams : ℕ
  gamesPerPair : ℕ
  noTies : Bool
  constantDifference : Bool

/-- Calculates the maximum number of games the last-place team could have won -/
def maxLastPlaceWins (league : BaseballLeague) : ℕ :=
  if league.teams = 14 ∧ league.gamesPerPair = 10 ∧ league.noTies ∧ league.constantDifference then
    52
  else
    0  -- Default value for other cases

/-- Theorem stating the maximum number of games the last-place team could have won -/
theorem max_last_place_wins_theorem (league : BaseballLeague) :
  league.teams = 14 ∧ 
  league.gamesPerPair = 10 ∧ 
  league.noTies ∧ 
  league.constantDifference →
  maxLastPlaceWins league = 52 := by
  sorry

#eval maxLastPlaceWins { teams := 14, gamesPerPair := 10, noTies := true, constantDifference := true }

end NUMINAMATH_CALUDE_max_last_place_wins_theorem_l3507_350788


namespace NUMINAMATH_CALUDE_parabola_focus_l3507_350796

/-- Represents a parabola with equation y^2 = ax and directrix x = -1 -/
structure Parabola where
  a : ℝ
  directrix : ℝ
  eq : ∀ x y : ℝ, y^2 = a * x

/-- The focus of a parabola -/
def focus (p : Parabola) : ℝ × ℝ := sorry

/-- Theorem stating that the focus of the given parabola is at (1, 0) -/
theorem parabola_focus (p : Parabola) (h1 : p.directrix = -1) : focus p = (1, 0) := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_l3507_350796


namespace NUMINAMATH_CALUDE_sum_of_distance_and_reciprocal_l3507_350737

theorem sum_of_distance_and_reciprocal (a b : ℝ) : 
  (|a| = 5 ∧ b = -(1 / (-1/3))) → (a + b = 2 ∨ a + b = -8) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_distance_and_reciprocal_l3507_350737


namespace NUMINAMATH_CALUDE_larger_number_with_hcf_and_lcm_factors_l3507_350758

/-- Given two positive integers with HCF 23 and LCM factors 13 and 14, the larger is 322 -/
theorem larger_number_with_hcf_and_lcm_factors (a b : ℕ+) : 
  (Nat.gcd a b = 23) → 
  (∃ k : ℕ+, Nat.lcm a b = 23 * 13 * 14 * k) → 
  (max a b = 322) := by
sorry

end NUMINAMATH_CALUDE_larger_number_with_hcf_and_lcm_factors_l3507_350758


namespace NUMINAMATH_CALUDE_complement_intersection_equal_set_l3507_350728

def U : Set Int := {0, -1, -2, -3, -4}
def M : Set Int := {0, -1, -2}
def N : Set Int := {0, -3, -4}

theorem complement_intersection_equal_set : (U \ M) ∩ N = {-3, -4} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_equal_set_l3507_350728


namespace NUMINAMATH_CALUDE_min_removals_correct_l3507_350777

/-- Represents a triangular grid constructed with toothpicks -/
structure ToothpickGrid where
  total_toothpicks : ℕ
  upward_triangles : ℕ
  downward_triangles : ℕ

/-- The number of horizontal toothpicks in the grid -/
def horizontal_toothpicks (grid : ToothpickGrid) : ℕ :=
  grid.upward_triangles

/-- The minimum number of toothpicks to remove to eliminate all triangles -/
def min_removals (grid : ToothpickGrid) : ℕ :=
  horizontal_toothpicks grid

theorem min_removals_correct (grid : ToothpickGrid) 
  (h1 : grid.total_toothpicks = 50)
  (h2 : grid.upward_triangles = 15)
  (h3 : grid.downward_triangles = 10) :
  min_removals grid = 15 := by
  sorry

#eval min_removals { total_toothpicks := 50, upward_triangles := 15, downward_triangles := 10 }

end NUMINAMATH_CALUDE_min_removals_correct_l3507_350777


namespace NUMINAMATH_CALUDE_train_length_l3507_350710

/-- The length of a train given specific conditions. -/
theorem train_length (jogger_speed : ℝ) (train_speed : ℝ) (initial_distance : ℝ) (passing_time : ℝ) : 
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  initial_distance = 200 →
  passing_time = 40 →
  (train_speed - jogger_speed) * passing_time - initial_distance = 200 := by
sorry

end NUMINAMATH_CALUDE_train_length_l3507_350710


namespace NUMINAMATH_CALUDE_shark_sightings_relationship_l3507_350701

/-- The number of shark sightings in Daytona Beach per year. -/
def daytona_sightings : ℕ := 26

/-- The number of shark sightings in Cape May per year. -/
def cape_may_sightings : ℕ := 7

/-- Theorem stating the relationship between shark sightings in Daytona Beach and Cape May. -/
theorem shark_sightings_relationship :
  daytona_sightings = 3 * cape_may_sightings + 5 ∧ cape_may_sightings = 7 := by
  sorry

end NUMINAMATH_CALUDE_shark_sightings_relationship_l3507_350701


namespace NUMINAMATH_CALUDE_production_days_calculation_l3507_350734

theorem production_days_calculation (n : ℕ) : 
  (n * 50 + 90) / (n + 1) = 52 → n = 19 := by sorry

end NUMINAMATH_CALUDE_production_days_calculation_l3507_350734


namespace NUMINAMATH_CALUDE_cube_root_simplification_l3507_350733

theorem cube_root_simplification :
  (2^9 * 3^3 * 5^3 * 11^3 : ℝ)^(1/3) = 1320 := by sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l3507_350733


namespace NUMINAMATH_CALUDE_decimal_conversion_and_addition_l3507_350768

def decimal_to_binary (n : ℕ) : List Bool :=
  sorry

def binary_to_decimal (b : List Bool) : ℕ :=
  sorry

def binary_add (a b : List Bool) : List Bool :=
  sorry

theorem decimal_conversion_and_addition :
  let binary_45 := decimal_to_binary 45
  let binary_3 := decimal_to_binary 3
  let sum := binary_add binary_45 binary_3
  binary_to_decimal sum = 48 := by
  sorry

end NUMINAMATH_CALUDE_decimal_conversion_and_addition_l3507_350768
