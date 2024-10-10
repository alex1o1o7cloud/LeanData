import Mathlib

namespace modulus_of_complex_fraction_l3563_356303

theorem modulus_of_complex_fraction (z : ℂ) :
  z = (5 : ℂ) / (1 - 2 * Complex.I) → Complex.abs z = Real.sqrt 5 := by
  sorry

end modulus_of_complex_fraction_l3563_356303


namespace imaginary_part_of_complex_fraction_l3563_356321

theorem imaginary_part_of_complex_fraction (i : ℂ) :
  i^2 = -1 →
  let z := (3 - 2 * i^2) / (1 + i)
  Complex.im z = -5/2 := by sorry

end imaginary_part_of_complex_fraction_l3563_356321


namespace statement_C_is_false_l3563_356310

-- Define the concept of lines in space
variable (Line : Type)

-- Define the perpendicular relationship between lines
variable (perpendicular : Line → Line → Prop)

-- Define the parallel relationship between lines
variable (parallel : Line → Line → Prop)

-- State the theorem to be proven false
theorem statement_C_is_false :
  ¬(∀ (a b c : Line), perpendicular a c → perpendicular b c → parallel a b) :=
sorry

end statement_C_is_false_l3563_356310


namespace museum_groups_l3563_356374

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

end museum_groups_l3563_356374


namespace a_is_guilty_l3563_356352

-- Define the set of suspects
inductive Suspect : Type
| A : Suspect
| B : Suspect
| C : Suspect

-- Define the properties of the crime and suspects
class CrimeScene where
  involved : Suspect → Prop
  canDrive : Suspect → Prop
  usedCar : Prop

-- Define the specific conditions of this crime
axiom crime_conditions (cs : CrimeScene) :
  -- The crime was committed using a car
  cs.usedCar ∧
  -- At least one suspect was involved
  (cs.involved Suspect.A ∨ cs.involved Suspect.B ∨ cs.involved Suspect.C) ∧
  -- C never commits a crime without A
  (cs.involved Suspect.C → cs.involved Suspect.A) ∧
  -- B knows how to drive
  cs.canDrive Suspect.B

-- Theorem: A is guilty
theorem a_is_guilty (cs : CrimeScene) : cs.involved Suspect.A :=
sorry

end a_is_guilty_l3563_356352


namespace equation_solution_difference_l3563_356323

theorem equation_solution_difference : ∃ (s₁ s₂ : ℝ),
  s₁ ≠ s₂ ∧
  s₁ ≠ -6 ∧
  s₂ ≠ -6 ∧
  (s₁^2 - 5*s₁ - 24) / (s₁ + 6) = 3*s₁ + 10 ∧
  (s₂^2 - 5*s₂ - 24) / (s₂ + 6) = 3*s₂ + 10 ∧
  |s₁ - s₂| = 6.5 := by
sorry

end equation_solution_difference_l3563_356323


namespace reflected_light_ray_equation_l3563_356398

/-- Given an incident light ray along y = 2x + 1 reflected by the line y = x,
    the equation of the reflected light ray is x - 2y - 1 = 0 -/
theorem reflected_light_ray_equation (x y : ℝ) : 
  (y = 2*x + 1) → -- incident light ray equation
  (y = x) →       -- reflecting line equation
  (x - 2*y - 1 = 0) -- reflected light ray equation
  := by sorry

end reflected_light_ray_equation_l3563_356398


namespace unit_circle_point_movement_l3563_356311

theorem unit_circle_point_movement (α : Real) (h1 : 0 < α) (h2 : α < Real.pi / 2) :
  let P₀ : ℝ × ℝ := (1, 0)
  let P₁ : ℝ × ℝ := (Real.cos α, Real.sin α)
  let P₂ : ℝ × ℝ := (Real.cos (α + Real.pi / 4), Real.sin (α + Real.pi / 4))
  P₂.1 = -3/5 → P₁.2 = 7 * Real.sqrt 2 / 10 := by
  sorry

end unit_circle_point_movement_l3563_356311


namespace total_weight_loss_l3563_356316

theorem total_weight_loss (first_person_loss second_person_loss third_person_loss fourth_person_loss : ℕ) :
  first_person_loss = 27 →
  second_person_loss = first_person_loss - 7 →
  third_person_loss = 28 →
  fourth_person_loss = 28 →
  first_person_loss + second_person_loss + third_person_loss + fourth_person_loss = 103 :=
by sorry

end total_weight_loss_l3563_356316


namespace total_heads_calculation_l3563_356325

theorem total_heads_calculation (num_hens : ℕ) (total_feet : ℕ) : num_hens = 20 → total_feet = 200 → ∃ (num_cows : ℕ), num_hens + num_cows = 60 := by
  sorry

end total_heads_calculation_l3563_356325


namespace alexis_isabella_shopping_ratio_l3563_356397

theorem alexis_isabella_shopping_ratio : 
  let alexis_pants : ℕ := 21
  let alexis_dresses : ℕ := 18
  let isabella_total : ℕ := 13
  (alexis_pants + alexis_dresses) / isabella_total = 3 :=
by sorry

end alexis_isabella_shopping_ratio_l3563_356397


namespace trajectory_is_ellipse_trajectory_is_ellipse_proof_l3563_356364

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

end trajectory_is_ellipse_trajectory_is_ellipse_proof_l3563_356364


namespace smallest_four_digit_divisible_by_43_l3563_356302

theorem smallest_four_digit_divisible_by_43 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 43 = 0 → n ≥ 1032 :=
by
  sorry

end smallest_four_digit_divisible_by_43_l3563_356302


namespace sequence_formula_l3563_356378

def S (n : ℕ+) : ℤ := -n^2 + 7*n

def a (n : ℕ+) : ℤ := -2*n + 8

theorem sequence_formula (n : ℕ+) : 
  (∀ k : ℕ+, S k = -k^2 + 7*k) → 
  a n = -2*n + 8 := by
  sorry

end sequence_formula_l3563_356378


namespace function_properties_l3563_356328

noncomputable def f (a b c x : ℝ) : ℝ := a * Real.sin x + b * Real.cos x + c

theorem function_properties (a b c : ℝ) 
  (h1 : f a b c 0 = 0)
  (h2 : ∀ x : ℝ, f a b c x ≤ f a b c (Real.pi / 3))
  (h3 : ∃ x : ℝ, f a b c x = 1) :
  (∃ x : ℝ, f a b c x = 1) ∧ 
  (∀ x : ℝ, f a b c x ≤ f (Real.sqrt 3) 1 (-1) x) ∧
  (f a b c (b / a) > f a b c (c / a)) := by
  sorry

#check function_properties

end function_properties_l3563_356328


namespace larger_denomination_proof_l3563_356394

theorem larger_denomination_proof (total_bills : ℕ) (total_value : ℕ) 
  (ten_bills : ℕ) (larger_bills : ℕ) :
  total_bills = 30 →
  total_value = 330 →
  ten_bills = 27 →
  larger_bills = 3 →
  ten_bills + larger_bills = total_bills →
  10 * ten_bills + larger_bills * (total_value - 10 * ten_bills) / larger_bills = total_value →
  (total_value - 10 * ten_bills) / larger_bills = 20 := by
  sorry

end larger_denomination_proof_l3563_356394


namespace decimal_multiplication_l3563_356372

theorem decimal_multiplication (a b : ℚ) (ha : a = 0.4) (hb : b = 0.75) : a * b = 0.3 := by
  sorry

end decimal_multiplication_l3563_356372


namespace oven_usage_calculation_l3563_356367

/-- Represents the problem of calculating oven usage time given electricity price, consumption rate, and total cost. -/
def OvenUsage (price : ℝ) (consumption : ℝ) (total_cost : ℝ) : Prop :=
  let hours := total_cost / (price * consumption)
  hours = 25

/-- Theorem stating that given the specific values in the problem, the oven usage time is 25 hours. -/
theorem oven_usage_calculation :
  OvenUsage 0.10 2.4 6 := by
  sorry

end oven_usage_calculation_l3563_356367


namespace negative_sixty_four_to_seven_sixths_l3563_356300

theorem negative_sixty_four_to_seven_sixths (x : ℝ) : x = (-64)^(7/6) → x = -16384 := by
  sorry

end negative_sixty_four_to_seven_sixths_l3563_356300


namespace quadratic_inequality_condition_l3563_356387

theorem quadratic_inequality_condition (a : ℝ) :
  (∀ x : ℝ, a * x^2 - 2 * a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 1) :=
by sorry

end quadratic_inequality_condition_l3563_356387


namespace small_pizza_slices_l3563_356391

/-- The number of large pizzas --/
def num_large_pizzas : ℕ := 2

/-- The number of small pizzas --/
def num_small_pizzas : ℕ := 2

/-- The number of slices in a large pizza --/
def slices_per_large_pizza : ℕ := 16

/-- The total number of slices eaten --/
def total_slices_eaten : ℕ := 48

/-- Theorem: The number of slices in a small pizza is 8 --/
theorem small_pizza_slices : 
  ∃ (slices_per_small_pizza : ℕ), 
    slices_per_small_pizza * num_small_pizzas + 
    slices_per_large_pizza * num_large_pizzas = total_slices_eaten ∧
    slices_per_small_pizza = 8 := by
  sorry

end small_pizza_slices_l3563_356391


namespace quadratic_function_range_l3563_356320

-- Define the quadratic function
def y (x m : ℝ) : ℝ := (x - 1) * (x - m + 1)

-- State the theorem
theorem quadratic_function_range (m : ℝ) : 
  (∀ x ∈ Set.Icc (-2 : ℝ) 0, y x m > 0) → m > 1 := by
  sorry

end quadratic_function_range_l3563_356320


namespace polynomial_equality_l3563_356326

theorem polynomial_equality (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (1 - 2*x)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a₀ + a₁ + a₃ = -39 := by
  sorry

end polynomial_equality_l3563_356326


namespace not_sufficient_nor_necessary_condition_l3563_356362

theorem not_sufficient_nor_necessary_condition (x y : ℝ) : 
  ¬(∀ x y : ℝ, x > y → x^2 > y^2) ∧ ¬(∀ x y : ℝ, x^2 > y^2 → x > y) := by
  sorry

end not_sufficient_nor_necessary_condition_l3563_356362


namespace number_line_percentage_l3563_356360

theorem number_line_percentage : 
  let start : ℝ := -55
  let end_point : ℝ := 55
  let target : ℝ := 5.5
  let total_distance := end_point - start
  let target_distance := target - start
  (target_distance / total_distance) * 100 = 55 := by
sorry

end number_line_percentage_l3563_356360


namespace prop1_prop4_l3563_356346

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

end prop1_prop4_l3563_356346


namespace simple_interest_time_l3563_356363

/-- Given simple interest, principal, and rate, calculate the time in years -/
theorem simple_interest_time (SI P R : ℚ) (h1 : SI = 4016.25) (h2 : P = 16065) (h3 : R = 5) :
  SI = P * R * 5 / 100 := by
  sorry

end simple_interest_time_l3563_356363


namespace circle_center_coordinates_l3563_356388

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y - 5 = 0

/-- The center of a circle -/
def CircleCenter (h k : ℝ) : Prop :=
  ∀ x y : ℝ, CircleEquation x y ↔ (x - h)^2 + (y - k)^2 = 10

theorem circle_center_coordinates :
  CircleCenter 2 1 := by sorry

end circle_center_coordinates_l3563_356388


namespace triangle_problem_l3563_356349

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

end triangle_problem_l3563_356349


namespace geometric_progression_equality_l3563_356382

theorem geometric_progression_equality (x y z : ℝ) :
  (∃ r : ℝ, y = x * r ∧ z = y * r) ↔ (x^2 + y^2) * (y^2 + z^2) = (x*y + y*z)^2 :=
sorry

end geometric_progression_equality_l3563_356382


namespace max_digit_sum_l3563_356308

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def decimal_to_fraction (a b c : ℕ) : ℚ :=
  (a * 100000 + a * 10000 + b * 1000 + b * 100 + c * 10 + c) / 1000000

theorem max_digit_sum (a b c y : ℕ) :
  is_digit a → is_digit b → is_digit c →
  decimal_to_fraction a b c = 1 / y →
  y > 0 → y ≤ 16 →
  a + b + c ≤ 13 :=
sorry

end max_digit_sum_l3563_356308


namespace cow_spots_l3563_356350

/-- Calculates the total number of spots on a cow given the number of spots on its left side. -/
def totalSpots (leftSpots : ℕ) : ℕ :=
  let rightSpots := 3 * leftSpots + 7
  leftSpots + rightSpots

/-- Theorem stating that a cow with 16 spots on its left side has 71 spots in total. -/
theorem cow_spots : totalSpots 16 = 71 := by
  sorry

end cow_spots_l3563_356350


namespace dice_roll_probability_l3563_356340

def probability_first_die : ℚ := 3 / 8
def probability_second_die : ℚ := 3 / 8

theorem dice_roll_probability : 
  probability_first_die * probability_second_die = 9 / 64 := by
  sorry

end dice_roll_probability_l3563_356340


namespace maple_grove_elementary_difference_l3563_356319

theorem maple_grove_elementary_difference : 
  let classrooms : ℕ := 5
  let students_per_classroom : ℕ := 22
  let hamsters_per_classroom : ℕ := 3
  let total_students : ℕ := classrooms * students_per_classroom
  let total_hamsters : ℕ := classrooms * hamsters_per_classroom
  total_students - total_hamsters = 95 := by
  sorry

end maple_grove_elementary_difference_l3563_356319


namespace parabola_focus_l3563_356359

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

end parabola_focus_l3563_356359


namespace max_last_place_wins_theorem_l3563_356373

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

end max_last_place_wins_theorem_l3563_356373


namespace sufficient_not_necessary_condition_l3563_356351

/-- An odd function from ℝ to ℝ -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem sufficient_not_necessary_condition
  (f : ℝ → ℝ) (hf : OddFunction f) :
  (∀ x₁ x₂ : ℝ, x₁ + x₂ = 0 → f x₁ + f x₂ = 0) ∧
  (∃ x₁ x₂ : ℝ, f x₁ + f x₂ = 0 ∧ x₁ + x₂ ≠ 0) :=
by sorry

end sufficient_not_necessary_condition_l3563_356351


namespace initial_crayons_l3563_356357

theorem initial_crayons (crayons_left : ℕ) (crayons_lost : ℕ) 
  (h1 : crayons_left = 134) 
  (h2 : crayons_lost = 345) : 
  crayons_left + crayons_lost = 479 := by
  sorry

end initial_crayons_l3563_356357


namespace tangent_line_circle_sum_constraint_l3563_356327

theorem tangent_line_circle_sum_constraint (m n : ℝ) : 
  m > 0 → n > 0 → 
  (∃ (x y : ℝ), (m + 1) * x + (n + 1) * y - 2 = 0 ∧ 
                (x - 1)^2 + (y - 1)^2 = 1 ∧
                ∀ (x' y' : ℝ), (x' - 1)^2 + (y' - 1)^2 ≤ 1 → 
                               (m + 1) * x' + (n + 1) * y' - 2 ≥ 0) →
  m + n ≥ 2 + 2 * Real.sqrt 2 :=
by sorry

end tangent_line_circle_sum_constraint_l3563_356327


namespace line_intercept_ratio_l3563_356345

/-- Given two lines with y-intercept 5, where the first line has slope 2 and x-intercept (s, 0),
    and the second line has slope 7 and x-intercept (t, 0), prove that s/t = 7/2 -/
theorem line_intercept_ratio (s t : ℝ) : 
  (5 : ℝ) = 2 * s + 5 ∧ (5 : ℝ) = 7 * t + 5 → s / t = 7 / 2 := by
  sorry

end line_intercept_ratio_l3563_356345


namespace total_items_in_jar_l3563_356314

/-- The total number of items in a jar with candy and secret eggs -/
theorem total_items_in_jar (candy : ℝ) (secret_eggs : ℝ) (h1 : candy = 3409.0) (h2 : secret_eggs = 145.0) :
  candy + secret_eggs = 3554.0 := by
  sorry

end total_items_in_jar_l3563_356314


namespace solution_set_when_a_is_neg_two_a_range_when_f_leq_g_on_interval_l3563_356353

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

end solution_set_when_a_is_neg_two_a_range_when_f_leq_g_on_interval_l3563_356353


namespace grandfather_gift_is_100_l3563_356383

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

end grandfather_gift_is_100_l3563_356383


namespace ellipse_equation_l3563_356318

/-- The equation of an ellipse given its foci and the sum of distances from any point on the ellipse to the foci -/
theorem ellipse_equation (F₁ F₂ M : ℝ × ℝ) (d : ℝ) : 
  F₁ = (-4, 0) →
  F₂ = (4, 0) →
  d = 10 →
  (dist M F₁ + dist M F₂ = d) →
  (M.1^2 / 25 + M.2^2 / 9 = 1) :=
by sorry

end ellipse_equation_l3563_356318


namespace function_inequality_l3563_356386

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h1 : ∀ x y, 0 < x ∧ x < y ∧ y < 2 → f x < f y)
variable (h2 : ∀ x, f (x + 2) = f (2 - x))

-- State the theorem
theorem function_inequality : f 2.5 > f 1 ∧ f 1 > f 3.5 := by sorry

end function_inequality_l3563_356386


namespace larger_number_with_hcf_and_lcm_factors_l3563_356348

/-- Given two positive integers with HCF 23 and LCM factors 13 and 14, the larger is 322 -/
theorem larger_number_with_hcf_and_lcm_factors (a b : ℕ+) : 
  (Nat.gcd a b = 23) → 
  (∃ k : ℕ+, Nat.lcm a b = 23 * 13 * 14 * k) → 
  (max a b = 322) := by
sorry

end larger_number_with_hcf_and_lcm_factors_l3563_356348


namespace geometric_sequence_11th_term_l3563_356305

/-- Given a geometric sequence where the 5th term is 8 and the 8th term is 64,
    prove that the 11th term is 512. -/
theorem geometric_sequence_11th_term (a : ℝ) (r : ℝ) :
  a * r^4 = 8 → a * r^7 = 64 → a * r^10 = 512 := by
  sorry

end geometric_sequence_11th_term_l3563_356305


namespace decimal_conversion_and_addition_l3563_356343

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

end decimal_conversion_and_addition_l3563_356343


namespace dodge_to_hyundai_ratio_l3563_356342

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

end dodge_to_hyundai_ratio_l3563_356342


namespace rectangle_dimensions_l3563_356381

theorem rectangle_dimensions (x : ℝ) : 
  (x - 3) * (3*x + 4) = 9*x - 19 → x = (7 + 2*Real.sqrt 7) / 3 :=
by
  sorry

end rectangle_dimensions_l3563_356381


namespace mutually_exclusive_events_l3563_356368

-- Define the sample space
def Ω : Type := Bool × Bool

-- Define the event "at most one shot is successful"
def at_most_one_successful (ω : Ω) : Prop :=
  ¬(ω.1 ∧ ω.2)

-- Define the event "both shots are successful"
def both_successful (ω : Ω) : Prop :=
  ω.1 ∧ ω.2

-- Theorem: "both shots are successful" is mutually exclusive to "at most one shot is successful"
theorem mutually_exclusive_events :
  ∀ ω : Ω, ¬(at_most_one_successful ω ∧ both_successful ω) :=
by
  sorry


end mutually_exclusive_events_l3563_356368


namespace smallest_n_exceeding_500000_l3563_356369

theorem smallest_n_exceeding_500000 : 
  (∀ k : ℕ, k < 10 → (3 : ℝ) ^ ((k * (k + 1) : ℝ) / 16) ≤ 500000) ∧ 
  (3 : ℝ) ^ ((10 * 11 : ℝ) / 16) > 500000 := by
  sorry

end smallest_n_exceeding_500000_l3563_356369


namespace sin_2theta_value_l3563_356366

theorem sin_2theta_value (θ : Real) (h : Real.sin θ + Real.cos θ = 1/5) :
  Real.sin (2 * θ) = -24/25 := by
  sorry

end sin_2theta_value_l3563_356366


namespace max_weighing_ways_exists_89_ways_l3563_356389

/-- Represents the set of weights as powers of 2 up to 2^9 (512) -/
def weights : Finset ℕ := Finset.range 10

/-- The number of ways a weight P can be measured using weights up to 2^n -/
def K (n : ℕ) (P : ℕ) : ℕ := sorry

/-- The maximum number of ways any weight can be measured using weights up to 2^n -/
def K_max (n : ℕ) : ℕ := sorry

/-- Theorem stating that no load can be weighed in more than 89 different ways -/
theorem max_weighing_ways : K_max 9 ≤ 89 := sorry

/-- Theorem stating that there exists a load that can be weighed in exactly 89 different ways -/
theorem exists_89_ways : ∃ P : ℕ, K 9 P = 89 := sorry

end max_weighing_ways_exists_89_ways_l3563_356389


namespace symmetric_points_sum_power_l3563_356365

/-- Two points are symmetric about the y-axis if their x-coordinates are opposite and their y-coordinates are equal -/
def symmetric_about_y_axis (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = y₂

theorem symmetric_points_sum_power (a b : ℝ) :
  symmetric_about_y_axis a 3 4 b → (a + b)^2008 = 1 := by
  sorry

end symmetric_points_sum_power_l3563_356365


namespace professor_newtons_students_l3563_356379

theorem professor_newtons_students (N M : ℕ) : 
  N % 4 = 2 →
  N % 5 = 1 →
  N = M + 15 →
  M < 15 →
  N = 26 ∧ M = 11 := by
sorry

end professor_newtons_students_l3563_356379


namespace police_officer_ratio_l3563_356309

/-- Proves that the ratio of female officers to male officers on duty is 1:1 -/
theorem police_officer_ratio (total_on_duty : ℕ) (female_officers : ℕ) (female_on_duty_percent : ℚ) :
  total_on_duty = 204 →
  female_officers = 600 →
  female_on_duty_percent = 17 / 100 →
  ∃ (female_on_duty male_on_duty : ℕ),
    female_on_duty = female_on_duty_percent * female_officers ∧
    male_on_duty = total_on_duty - female_on_duty ∧
    female_on_duty = male_on_duty :=
by sorry

end police_officer_ratio_l3563_356309


namespace inequality_of_means_l3563_356393

theorem inequality_of_means (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  (a^2 + b^2) / 2 > a * b ∧ a * b > 2 * a^2 * b^2 / (a^2 + b^2) := by
  sorry

end inequality_of_means_l3563_356393


namespace jogger_speed_l3563_356336

/-- The speed of a jogger on a path with specific conditions -/
theorem jogger_speed (inner_perimeter outer_perimeter : ℝ) 
  (h1 : outer_perimeter - inner_perimeter = 16 * Real.pi)
  (time_diff : ℝ) (h2 : time_diff = 60) :
  ∃ (speed : ℝ), speed = (4 * Real.pi) / 15 ∧ 
    outer_perimeter / speed = inner_perimeter / speed + time_diff :=
sorry

end jogger_speed_l3563_356336


namespace max_value_implies_tan_2alpha_l3563_356347

/-- Given a function f(x) = 3sin(x) + cos(x) that attains its maximum value when x = α,
    prove that tan(2α) = -3/4 -/
theorem max_value_implies_tan_2alpha (α : ℝ) 
  (h : ∀ x, 3 * Real.sin x + Real.cos x ≤ 3 * Real.sin α + Real.cos α) : 
  Real.tan (2 * α) = -3/4 := by
  sorry

end max_value_implies_tan_2alpha_l3563_356347


namespace arithmetic_sequence_problem_l3563_356375

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

end arithmetic_sequence_problem_l3563_356375


namespace work_completion_time_l3563_356384

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

end work_completion_time_l3563_356384


namespace flag_design_count_l3563_356329

/-- The number of possible colors for each stripe -/
def num_colors : ℕ := 3

/-- The number of stripes on the flag -/
def num_stripes : ℕ := 3

/-- The total number of possible flag designs -/
def total_flags : ℕ := num_colors ^ num_stripes

theorem flag_design_count :
  total_flags = 27 :=
by sorry

end flag_design_count_l3563_356329


namespace leos_laundry_bill_l3563_356307

/-- The total bill amount for Leo's laundry -/
def total_bill_amount (trousers_count : ℕ) (initial_shirts_count : ℕ) (missing_shirts_count : ℕ) 
  (trouser_price : ℕ) (shirt_price : ℕ) : ℕ :=
  trousers_count * trouser_price + (initial_shirts_count + missing_shirts_count) * shirt_price

/-- Theorem stating that Leo's total bill amount is $140 -/
theorem leos_laundry_bill : 
  total_bill_amount 10 2 8 9 5 = 140 := by
  sorry

#eval total_bill_amount 10 2 8 9 5

end leos_laundry_bill_l3563_356307


namespace family_admission_price_l3563_356380

/-- Calculates the total admission price for a family visiting an amusement park. -/
theorem family_admission_price 
  (adult_price : ℕ) 
  (child_price : ℕ) 
  (num_adults : ℕ) 
  (num_children : ℕ) 
  (h1 : adult_price = 22)
  (h2 : child_price = 7)
  (h3 : num_adults = 2)
  (h4 : num_children = 2) :
  adult_price * num_adults + child_price * num_children = 58 := by
  sorry

#check family_admission_price

end family_admission_price_l3563_356380


namespace tickets_difference_l3563_356361

theorem tickets_difference (initial_tickets : ℕ) (toys_tickets : ℕ) (clothes_tickets : ℕ)
  (h1 : initial_tickets = 13)
  (h2 : toys_tickets = 8)
  (h3 : clothes_tickets = 18) :
  clothes_tickets - toys_tickets = 10 := by
  sorry

end tickets_difference_l3563_356361


namespace snack_distribution_l3563_356304

theorem snack_distribution (candies jellies students : ℕ) 
  (h1 : candies = 72)
  (h2 : jellies = 56)
  (h3 : students = 4) :
  (candies + jellies) / students = 32 :=
by sorry

end snack_distribution_l3563_356304


namespace isaiah_typing_speed_l3563_356355

theorem isaiah_typing_speed 
  (micah_speed : ℕ) 
  (isaiah_hourly_diff : ℕ) 
  (h1 : micah_speed = 20)
  (h2 : isaiah_hourly_diff = 1200) : 
  (micah_speed * 60 + isaiah_hourly_diff) / 60 = 40 := by
  sorry

end isaiah_typing_speed_l3563_356355


namespace algebraic_expression_equality_l3563_356332

theorem algebraic_expression_equality : 
  let a : ℝ := Real.sqrt 3 + 2
  (a - Real.sqrt 2) * (a + Real.sqrt 2) - a * (a - 3) = 3 * Real.sqrt 3 + 4 := by
  sorry

end algebraic_expression_equality_l3563_356332


namespace air_quality_probability_l3563_356344

theorem air_quality_probability (p_one_day p_two_days : ℝ) 
  (h1 : p_one_day = 0.8)
  (h2 : p_two_days = 0.6) :
  p_two_days / p_one_day = 0.75 := by
  sorry

end air_quality_probability_l3563_356344


namespace no_closed_broken_line_315_l3563_356392

/-- A closed broken line with the given properties -/
structure ClosedBrokenLine where
  segments : ℕ
  intersecting : Bool
  perpendicular : Bool
  symmetric : Bool

/-- The number of segments in our specific case -/
def n : ℕ := 315

/-- Theorem stating the impossibility of constructing the specified closed broken line -/
theorem no_closed_broken_line_315 :
  ¬ ∃ (line : ClosedBrokenLine), 
    line.segments = n ∧
    line.intersecting ∧
    line.perpendicular ∧
    line.symmetric :=
sorry


end no_closed_broken_line_315_l3563_356392


namespace planar_graph_inequality_l3563_356341

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

end planar_graph_inequality_l3563_356341


namespace zeros_inequality_l3563_356356

open Real

theorem zeros_inequality (x₁ x₂ m : ℝ) (h₁ : x₁ < x₂) 
  (h₂ : exp (m * x₁) - log x₁ + (m - 1) * x₁ = 0) 
  (h₃ : exp (m * x₂) - log x₂ + (m - 1) * x₂ = 0) : 
  2 * log x₁ + log x₂ > exp 1 := by
  sorry

end zeros_inequality_l3563_356356


namespace expression_simplification_l3563_356313

theorem expression_simplification (a b : ℝ) 
  (h : |2*a - 1| + (b + 4)^2 = 0) : 
  a^3*b - a^2*b^3 - 1/2*(4*a*b - 6*a^2*b^3 - 1) + 2*(a*b - a^2*b^3) = 0 := by
sorry

end expression_simplification_l3563_356313


namespace unique_positive_cyclic_shift_l3563_356385

def CyclicShift (a : List ℤ) : List (List ℤ) :=
  List.range a.length |>.map (λ i => a.rotate i)

def PositivePartialSums (a : List ℤ) : Prop :=
  List.scanl (· + ·) 0 a |>.tail |>.all (λ x => x > 0)

theorem unique_positive_cyclic_shift
  (a : List ℤ)
  (h_sum : a.sum = 1) :
  ∃! shift, shift ∈ CyclicShift a ∧ PositivePartialSums shift :=
sorry

end unique_positive_cyclic_shift_l3563_356385


namespace coordinates_of_P_fixed_points_of_N_min_length_AB_l3563_356322

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + (y - 4)^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop := x - 2*y = 0

-- Define point P on line l
structure Point_P where
  x : ℝ
  y : ℝ
  on_line : line_l x y

-- Define tangent PA
def tangent_PA (p : Point_P) (a : ℝ × ℝ) : Prop :=
  circle_M a.1 a.2 ∧ 
  (p.x - a.1)^2 + (p.y - a.2)^2 = ((0 - p.x)^2 + (4 - p.y)^2)

-- Theorem 1
theorem coordinates_of_P : 
  ∃ (p : Point_P), (∃ (a : ℝ × ℝ), tangent_PA p a ∧ (p.x - a.1)^2 + (p.y - a.2)^2 = 12) →
  (p.x = 0 ∧ p.y = 0) ∨ (p.x = 16/5 ∧ p.y = 8/5) :=
sorry

-- Define circle N (circumcircle of triangle PAM)
def circle_N (p : Point_P) (x y : ℝ) : Prop :=
  (2*x + y - 4) * p.y - (x^2 + y^2 - 4*y) = 0

-- Theorem 2
theorem fixed_points_of_N :
  ∀ (p : Point_P), 
    (circle_N p 0 4 ∧ circle_N p (8/5) (4/5)) ∧
    (∀ (x y : ℝ), circle_N p x y → (x = 0 ∧ y = 4) ∨ (x = 8/5 ∧ y = 4/5)) :=
sorry

-- Define chord AB
def chord_AB (p : Point_P) (x y : ℝ) : Prop :=
  2 * p.y * x + (p.y - 4) * y + 12 - 4 * p.y = 0

-- Theorem 3
theorem min_length_AB :
  ∃ (p : Point_P), 
    (∀ (p' : Point_P), 
      (∀ (x y : ℝ), chord_AB p x y → 
        (x - 0)^2 + (y - 4)^2 ≥ (x - 0)^2 + (y - 4)^2)) ∧
    (∃ (a b : ℝ × ℝ), chord_AB p a.1 a.2 ∧ chord_AB p b.1 b.2 ∧ 
      (a.1 - b.1)^2 + (a.2 - b.2)^2 = 11) :=
sorry

end coordinates_of_P_fixed_points_of_N_min_length_AB_l3563_356322


namespace new_pet_ratio_l3563_356315

/-- Represents the number of pets of each type -/
structure PetCount where
  dogs : ℕ
  cats : ℕ
  birds : ℕ

/-- Calculates the new pet count after changes -/
def newPetCount (initial : PetCount) : PetCount :=
  { dogs := initial.dogs - 15,
    cats := initial.cats + 4 - 12,
    birds := initial.birds + 7 - 5 }

/-- Theorem stating the new ratio of pets after changes -/
theorem new_pet_ratio (initial : PetCount) :
  initial.dogs + initial.cats + initial.birds = 315 →
  initial.dogs * 35 = 315 * 10 →
  initial.cats * 35 = 315 * 17 →
  initial.birds * 35 = 315 * 8 →
  let final := newPetCount initial
  (final.dogs, final.cats, final.birds) = (75, 145, 74) :=
by sorry

end new_pet_ratio_l3563_356315


namespace inequality_contradiction_l3563_356358

theorem inequality_contradiction (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ¬(a + b < c + d ∧ (a + b) * (c + d) < a * b + c * d ∧ (a + b) * c * d < a * b * (c + d)) :=
by sorry

end inequality_contradiction_l3563_356358


namespace sector_max_area_l3563_356312

/-- Given a sector of a circle with perimeter 30 cm, prove that the maximum area is 225/4 cm² and the corresponding central angle is 2 radians. -/
theorem sector_max_area (l R : ℝ) (h1 : l + 2*R = 30) (h2 : l > 0) (h3 : R > 0) :
  ∃ (S : ℝ), S ≤ 225/4 ∧
  (S = 225/4 → l = 15 ∧ R = 15/2 ∧ l / R = 2) :=
sorry

end sector_max_area_l3563_356312


namespace deposit_calculation_l3563_356376

theorem deposit_calculation (total_price : ℝ) (deposit_percentage : ℝ) (remaining_amount : ℝ) :
  deposit_percentage = 0.1 →
  remaining_amount = 1170 →
  (1 - deposit_percentage) * total_price = remaining_amount →
  deposit_percentage * total_price = 130 := by
  sorry

end deposit_calculation_l3563_356376


namespace sqrt_27_div_sqrt_3_eq_3_l3563_356371

theorem sqrt_27_div_sqrt_3_eq_3 : Real.sqrt 27 / Real.sqrt 3 = 3 := by
  sorry

end sqrt_27_div_sqrt_3_eq_3_l3563_356371


namespace item_sale_ratio_l3563_356354

theorem item_sale_ratio (c x y : ℝ) (hx : x = 0.8 * c) (hy : y = 1.25 * c) : y / x = 25 / 16 := by
  sorry

end item_sale_ratio_l3563_356354


namespace min_removals_correct_l3563_356395

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

end min_removals_correct_l3563_356395


namespace parabola_intercept_problem_l3563_356396

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

end parabola_intercept_problem_l3563_356396


namespace fraction_increase_condition_l3563_356399

theorem fraction_increase_condition (m n : ℤ) (h1 : n ≠ 0) (h2 : n ≠ -1) :
  (m : ℚ) / n < (m + 1 : ℚ) / (n + 1) ↔ (n > 0 ∧ m < n) ∨ (n < -1 ∧ m > n) := by
  sorry

end fraction_increase_condition_l3563_356399


namespace min_value_expression_l3563_356338

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 1/y) * (x + 1/y - 2023) + (y + 1/x) * (y + 1/x - 2023) ≥ -2050513 := by
  sorry

end min_value_expression_l3563_356338


namespace fourth_root_of_390625_l3563_356370

theorem fourth_root_of_390625 (x : ℝ) (h1 : x > 0) (h2 : x^4 = 390625) : x = 25 := by
  sorry

end fourth_root_of_390625_l3563_356370


namespace exists_large_number_with_invariant_prime_factors_l3563_356334

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

end exists_large_number_with_invariant_prime_factors_l3563_356334


namespace largest_number_in_L_shape_l3563_356333

/-- Represents the different orientations of the "L" shape -/
inductive LShape
  | First  : LShape  -- (x-8, x-7, x)
  | Second : LShape  -- (x-7, x-6, x)
  | Third  : LShape  -- (x-7, x-1, x)
  | Fourth : LShape  -- (x-8, x-1, x)

/-- Calculates the sum of the three numbers in the "L" shape -/
def sumLShape (shape : LShape) (x : ℕ) : ℕ :=
  match shape with
  | LShape.First  => x - 8 + x - 7 + x
  | LShape.Second => x - 7 + x - 6 + x
  | LShape.Third  => x - 7 + x - 1 + x
  | LShape.Fourth => x - 8 + x - 1 + x

/-- The main theorem to be proved -/
theorem largest_number_in_L_shape : 
  ∃ (shape : LShape) (x : ℕ), sumLShape shape x = 2015 ∧ 
  (∀ (shape' : LShape) (y : ℕ), sumLShape shape' y = 2015 → y ≤ x) ∧ 
  x = 676 := by
  sorry

end largest_number_in_L_shape_l3563_356333


namespace monopoly_prefers_durable_coffee_machine_production_decision_l3563_356377

/-- Represents the type of coffee machine -/
inductive CoffeeMachineType
| Durable
| LowQuality

/-- Represents the market structure -/
inductive MarketStructure
| Monopoly
| PerfectlyCompetitive

/-- Represents a coffee machine -/
structure CoffeeMachine where
  type : CoffeeMachineType
  productionCost : ℝ

/-- Represents the consumer's utility from using a coffee machine -/
def consumerUtility : ℝ := 20

/-- Represents the lifespan of a coffee machine in periods -/
def machineLifespan (t : CoffeeMachineType) : ℕ :=
  match t with
  | CoffeeMachineType.Durable => 2
  | CoffeeMachineType.LowQuality => 1

/-- Calculates the profit for a monopolist selling a coffee machine -/
def monopolyProfit (m : CoffeeMachine) : ℝ :=
  (consumerUtility * machineLifespan m.type) - m.productionCost

/-- Theorem: In a monopoly, durable machines are produced when low-quality machine cost exceeds 6 -/
theorem monopoly_prefers_durable (c : ℝ) :
  let durableMachine : CoffeeMachine := ⟨CoffeeMachineType.Durable, 12⟩
  let lowQualityMachine : CoffeeMachine := ⟨CoffeeMachineType.LowQuality, c⟩
  monopolyProfit durableMachine > 2 * monopolyProfit lowQualityMachine ↔ c > 6 := by
  sorry

/-- Main theorem combining all conditions -/
theorem coffee_machine_production_decision 
  (marketStructure : MarketStructure) 
  (c : ℝ) : 
  (marketStructure = MarketStructure.Monopoly ∧ c > 6) ↔ 
  (∃ (d : CoffeeMachine), d.type = CoffeeMachineType.Durable ∧ 
   ∀ (l : CoffeeMachine), l.type = CoffeeMachineType.LowQuality → 
   monopolyProfit d > monopolyProfit l) := by
  sorry

end monopoly_prefers_durable_coffee_machine_production_decision_l3563_356377


namespace committee_probability_l3563_356324

def science_club_size : ℕ := 24
def num_boys : ℕ := 12
def num_girls : ℕ := 12
def committee_size : ℕ := 5

def probability_at_least_two_of_each : ℚ :=
  4704 / 7084

theorem committee_probability :
  let total_committees := Nat.choose science_club_size committee_size
  let valid_committees := total_committees - (
    Nat.choose num_boys 0 * Nat.choose num_girls 5 +
    Nat.choose num_boys 1 * Nat.choose num_girls 4 +
    Nat.choose num_boys 4 * Nat.choose num_girls 1 +
    Nat.choose num_boys 5 * Nat.choose num_girls 0
  )
  (valid_committees : ℚ) / total_committees = probability_at_least_two_of_each :=
by sorry

end committee_probability_l3563_356324


namespace blue_tiles_count_l3563_356335

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

end blue_tiles_count_l3563_356335


namespace product_purely_imaginary_l3563_356306

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the product function
def product (x : ℝ) : ℂ := (x - i) * ((x + 2) - i) * ((x + 4) - i)

-- Define the property of being purely imaginary
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0

-- State the theorem
theorem product_purely_imaginary (x : ℝ) : 
  isPurelyImaginary (product x) ↔ 
  (x = -3 ∨ x = (-3 + Real.sqrt 13) / 2 ∨ x = (-3 - Real.sqrt 13) / 2) :=
sorry

end product_purely_imaginary_l3563_356306


namespace geometric_series_relation_l3563_356301

/-- Given real numbers c and d satisfying an infinite geometric series equation,
    prove that another related infinite geometric series equals 3/5. -/
theorem geometric_series_relation (c d : ℝ) 
    (h : (c/d) / (1 - 1/d) = 3) :
    (c/(c+2*d)) / (1 - 1/(c+2*d)) = 3/5 := by
  sorry

end geometric_series_relation_l3563_356301


namespace integer_root_of_polynomial_l3563_356317

-- Define the polynomial
def P (x b c : ℚ) : ℚ := x^4 + 7*x^3 + b*x + c

-- State the theorem
theorem integer_root_of_polynomial (b c : ℚ) :
  (∃ (r : ℚ), r^2 = 5 ∧ P (2 + r) b c = 0) →  -- 2 + √5 is a root
  (∃ (n : ℤ), P n b c = 0) →                  -- There exists an integer root
  P 0 b c = 0                                 -- 0 is a root
:= by sorry

end integer_root_of_polynomial_l3563_356317


namespace polygon_sides_l3563_356339

theorem polygon_sides (n : ℕ) : 
  (n ≥ 3) →  -- Ensure it's a valid polygon
  ((n - 2) * 180 = 3 * 360) → -- Interior angles sum is 3 times exterior angles sum
  n = 8 := by
sorry

end polygon_sides_l3563_356339


namespace negation_empty_subset_any_set_l3563_356337

theorem negation_empty_subset_any_set :
  (¬ ∀ A : Set α, ∅ ⊆ A) ↔ (∃ A : Set α, ¬(∅ ⊆ A)) :=
by sorry

end negation_empty_subset_any_set_l3563_356337


namespace sin_negative_690_degrees_l3563_356331

theorem sin_negative_690_degrees : Real.sin ((-690 : ℝ) * π / 180) = 1 / 2 := by sorry

end sin_negative_690_degrees_l3563_356331


namespace ship_passengers_l3563_356330

theorem ship_passengers :
  let total : ℕ := 900
  let north_america : ℚ := 1/4
  let europe : ℚ := 2/15
  let africa : ℚ := 1/5
  let asia : ℚ := 1/6
  let south_america : ℚ := 1/12
  let oceania : ℚ := 1/20
  let other_regions : ℕ := 105
  (north_america + europe + africa + asia + south_america + oceania) * total + other_regions = total :=
by sorry

end ship_passengers_l3563_356330


namespace wire_sharing_l3563_356390

/-- Given a wire of total length 150 cm, where one person's share is 16 cm shorter than the other's,
    prove that the shorter share is 67 cm. -/
theorem wire_sharing (total_length : ℕ) (difference : ℕ) (seokgi_share : ℕ) (yeseul_share : ℕ) :
  total_length = 150 ∧ difference = 16 ∧ seokgi_share + yeseul_share = total_length ∧ 
  yeseul_share = seokgi_share + difference → seokgi_share = 67 :=
by sorry

end wire_sharing_l3563_356390
