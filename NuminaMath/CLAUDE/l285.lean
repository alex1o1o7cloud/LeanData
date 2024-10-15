import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_inequality_l285_28501

/-- A quadratic function f(x) = x^2 + bx + c where f(-1) = f(3) -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

theorem quadratic_inequality (b c : ℝ) :
  f b c (-1) = f b c 3 →
  f b c 1 < c ∧ c < f b c (-1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l285_28501


namespace NUMINAMATH_CALUDE_min_distance_sum_l285_28552

/-- Given points A and B, and a point P on a circle, prove the minimum value of |PA|^2 + |PB|^2 -/
theorem min_distance_sum (A B P : ℝ × ℝ) : 
  A = (-2, 0) →
  B = (2, 0) →
  (P.1 - 3)^2 + (P.2 - 4)^2 = 4 →
  (P.1 + 2)^2 + P.2^2 + (P.1 - 2)^2 + P.2^2 ≥ 26 := by
  sorry

#check min_distance_sum

end NUMINAMATH_CALUDE_min_distance_sum_l285_28552


namespace NUMINAMATH_CALUDE_power_of_seven_l285_28514

theorem power_of_seven (k : ℕ) (h : 7^k = 2) : 7^(4*k + 2) = 784 := by
  sorry

end NUMINAMATH_CALUDE_power_of_seven_l285_28514


namespace NUMINAMATH_CALUDE_yellow_highlighters_count_l285_28540

theorem yellow_highlighters_count (pink : ℕ) (blue : ℕ) (total : ℕ) (yellow : ℕ) : 
  pink = 10 → blue = 8 → total = 33 → yellow = total - (pink + blue) → yellow = 15 := by
  sorry

end NUMINAMATH_CALUDE_yellow_highlighters_count_l285_28540


namespace NUMINAMATH_CALUDE_mat_weavers_problem_l285_28560

/-- The number of mat-weavers in the first group -/
def first_group_weavers : ℕ := 4

/-- The number of mats woven by the first group -/
def first_group_mats : ℕ := 4

/-- The number of days taken by the first group -/
def first_group_days : ℕ := 4

/-- The number of mat-weavers in the second group -/
def second_group_weavers : ℕ := 16

/-- The number of mats woven by the second group -/
def second_group_mats : ℕ := 64

/-- The number of days taken by the second group -/
def second_group_days : ℕ := 16

theorem mat_weavers_problem :
  first_group_weavers * second_group_mats * first_group_days =
  second_group_weavers * first_group_mats * second_group_days :=
by sorry

end NUMINAMATH_CALUDE_mat_weavers_problem_l285_28560


namespace NUMINAMATH_CALUDE_area_of_triangle_BQW_l285_28562

/-- Given a rectangle ABCD with the following properties:
    - AZ = WC = 8 units
    - AB = 16 units
    - Area of trapezoid ZWCD is 160 square units
    - Q divides ZW in the ratio 1:3 starting from Z
    Prove that the area of triangle BQW is 16 square units. -/
theorem area_of_triangle_BQW (AZ WC AB : ℝ) (area_ZWCD : ℝ) (Q : ℝ) :
  AZ = 8 →
  WC = 8 →
  AB = 16 →
  area_ZWCD = 160 →
  Q = 2 →  -- This represents Q dividing ZW in 1:3 ratio
  (1/2 : ℝ) * AB * Q = 16 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_BQW_l285_28562


namespace NUMINAMATH_CALUDE_kickball_players_l285_28500

/-- The number of students who played kickball on Wednesday -/
def wednesday_players : ℕ := sorry

/-- The number of students who played kickball on Thursday -/
def thursday_players : ℕ := sorry

/-- The total number of students who played kickball on both days -/
def total_players : ℕ := 65

theorem kickball_players :
  wednesday_players = 37 ∧
  thursday_players = wednesday_players - 9 ∧
  wednesday_players + thursday_players = total_players :=
by sorry

end NUMINAMATH_CALUDE_kickball_players_l285_28500


namespace NUMINAMATH_CALUDE_amy_total_distance_l285_28502

/-- Calculates the total distance Amy biked over two days given the conditions. -/
def total_distance (yesterday : ℕ) (less_than_twice : ℕ) : ℕ :=
  yesterday + (2 * yesterday - less_than_twice)

/-- Proves that Amy biked 33 miles in total over two days. -/
theorem amy_total_distance :
  total_distance 12 3 = 33 := by
  sorry

end NUMINAMATH_CALUDE_amy_total_distance_l285_28502


namespace NUMINAMATH_CALUDE_total_blue_marbles_l285_28583

/-- The total number of blue marbles owned by Jason, Tom, and Emily is 104. -/
theorem total_blue_marbles (jason_blue : ℕ) (tom_blue : ℕ) (emily_blue : ℕ)
  (h1 : jason_blue = 44)
  (h2 : tom_blue = 24)
  (h3 : emily_blue = 36) :
  jason_blue + tom_blue + emily_blue = 104 :=
by sorry

end NUMINAMATH_CALUDE_total_blue_marbles_l285_28583


namespace NUMINAMATH_CALUDE_expressions_equality_l285_28518

theorem expressions_equality (a b c : ℝ) : a + b * c = (a + b) * (a + c) ↔ a + b + c = 1 := by
  sorry

end NUMINAMATH_CALUDE_expressions_equality_l285_28518


namespace NUMINAMATH_CALUDE_georginas_parrots_l285_28534

/-- Represents a parrot with its phrases and learning rate -/
structure Parrot where
  name : String
  current_phrases : ℕ
  phrases_per_week : ℕ
  initial_phrases : ℕ

/-- Calculates the number of weekdays since a parrot was bought -/
def weekdays_since_bought (p : Parrot) : ℕ :=
  ((p.current_phrases - p.initial_phrases + p.phrases_per_week - 1) / p.phrases_per_week) * 5

/-- The main theorem about Georgina's parrots -/
theorem georginas_parrots :
  let polly : Parrot := { name := "Polly", current_phrases := 17, phrases_per_week := 2, initial_phrases := 3 }
  let pedro : Parrot := { name := "Pedro", current_phrases := 12, phrases_per_week := 3, initial_phrases := 0 }
  let penelope : Parrot := { name := "Penelope", current_phrases := 8, phrases_per_week := 1, initial_phrases := 0 }
  let pascal : Parrot := { name := "Pascal", current_phrases := 20, phrases_per_week := 4, initial_phrases := 1 }
  weekdays_since_bought polly = 35 ∧
  weekdays_since_bought pedro = 20 ∧
  weekdays_since_bought penelope = 40 ∧
  weekdays_since_bought pascal = 25 := by
  sorry

end NUMINAMATH_CALUDE_georginas_parrots_l285_28534


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l285_28524

theorem max_sum_of_factors (A B C : ℕ) : 
  A ≠ B ∧ B ≠ C ∧ A ≠ C →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A * B * C = 2550 →
  A + B + C ≤ 98 := by
sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l285_28524


namespace NUMINAMATH_CALUDE_smallest_n_inequality_l285_28585

theorem smallest_n_inequality (x y z w : ℝ) : 
  ∃ (n : ℕ), n = 4 ∧ 
  (∀ (m : ℕ), m < n → ∃ (a b c d : ℝ), (a^2 + b^2 + c^2 + d^2)^2 > m*(a^4 + b^4 + c^4 + d^4)) ∧
  (x^2 + y^2 + z^2 + w^2)^2 ≤ n*(x^4 + y^4 + z^4 + w^4) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_inequality_l285_28585


namespace NUMINAMATH_CALUDE_khalil_dogs_count_l285_28511

/-- Represents the veterinary clinic problem -/
def veterinary_clinic_problem (dog_cost cat_cost : ℕ) (num_cats total_cost : ℕ) : Prop :=
  ∃ (num_dogs : ℕ), 
    dog_cost * num_dogs + cat_cost * num_cats = total_cost

/-- Proves that the number of dogs Khalil took to the clinic is 20 -/
theorem khalil_dogs_count : veterinary_clinic_problem 60 40 60 3600 → 
  ∃ (num_dogs : ℕ), num_dogs = 20 ∧ 60 * num_dogs + 40 * 60 = 3600 :=
by
  sorry


end NUMINAMATH_CALUDE_khalil_dogs_count_l285_28511


namespace NUMINAMATH_CALUDE_max_m_value_l285_28573

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem max_m_value :
  (∃ (m : ℝ), ∃ (x : ℝ), x ∈ Set.Icc 0 1 ∧ f x = m) →
  (∀ (m : ℝ), (∃ (x : ℝ), x ∈ Set.Icc 0 1 ∧ f x = m) → m ≤ 0) ∧
  (∃ (x : ℝ), x ∈ Set.Icc 0 1 ∧ f x = 0) :=
by sorry


end NUMINAMATH_CALUDE_max_m_value_l285_28573


namespace NUMINAMATH_CALUDE_work_earnings_equation_l285_28547

theorem work_earnings_equation (t : ℝ) : 
  (t + 1) * (3 * t - 3) = (3 * t - 5) * (t + 2) + 2 → t = 5 := by
  sorry

end NUMINAMATH_CALUDE_work_earnings_equation_l285_28547


namespace NUMINAMATH_CALUDE_integer_solutions_inequality_l285_28504

theorem integer_solutions_inequality (a : ℝ) (h_pos : a > 0) 
  (h_three_solutions : ∃ x y z : ℤ, x < y ∧ y < z ∧ 
    (∀ w : ℤ, 1 < w * a ∧ w * a < 2 ↔ w = x ∨ w = y ∨ w = z)) :
  ∃ p q r : ℤ, p < q ∧ q < r ∧ 
    (∀ w : ℤ, 2 < w * a ∧ w * a < 3 ↔ w = p ∨ w = q ∨ w = r) := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_inequality_l285_28504


namespace NUMINAMATH_CALUDE_divisibility_by_seven_l285_28581

theorem divisibility_by_seven (A a b : ℕ) : A = 100 * a + b →
  (7 ∣ A ↔ 7 ∣ (2 * a + b)) ∧ (7 ∣ A ↔ 7 ∣ (5 * a - b)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_seven_l285_28581


namespace NUMINAMATH_CALUDE_prescription_final_cost_l285_28531

/-- Calculates the final cost of a prescription after cashback and rebate --/
theorem prescription_final_cost
  (original_cost : ℝ)
  (cashback_percentage : ℝ)
  (rebate : ℝ)
  (h1 : original_cost = 150)
  (h2 : cashback_percentage = 0.1)
  (h3 : rebate = 25) :
  original_cost - (cashback_percentage * original_cost + rebate) = 110 :=
by sorry

end NUMINAMATH_CALUDE_prescription_final_cost_l285_28531


namespace NUMINAMATH_CALUDE_inequality_solution_l285_28598

-- Define the solution set for x^2 - ax - b < 0
def solution_set (a b : ℝ) : Set ℝ := {x | 2 < x ∧ x < 3}

-- Theorem statement
theorem inequality_solution :
  ∀ a b : ℝ, (solution_set a b = {x | 2 < x ∧ x < 3}) →
  (a = 5 ∧ b = -6) ∧
  ({x : ℝ | b * x^2 - a * x - 1 > 0} = {x : ℝ | -1/2 < x ∧ x < -1/3}) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l285_28598


namespace NUMINAMATH_CALUDE_nine_integer_chords_l285_28529

/-- Represents a circle with a given radius and a point P at a given distance from the center -/
structure CircleWithPoint where
  radius : ℝ
  distanceToP : ℝ

/-- Counts the number of integer-length chords passing through P -/
def countIntegerChords (c : CircleWithPoint) : ℕ :=
  sorry

/-- The main theorem to be proved -/
theorem nine_integer_chords (c : CircleWithPoint) 
  (h1 : c.radius = 20)
  (h2 : c.distanceToP = 12) : 
  countIntegerChords c = 9 :=
sorry

end NUMINAMATH_CALUDE_nine_integer_chords_l285_28529


namespace NUMINAMATH_CALUDE_complement_of_N_in_M_l285_28508

def M : Set Nat := {0, 1, 2, 3, 4, 5}
def N : Set Nat := {0, 2, 3}

theorem complement_of_N_in_M :
  M \ N = {1, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_N_in_M_l285_28508


namespace NUMINAMATH_CALUDE_largest_three_digit_number_l285_28564

def digits : Finset Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def valid_equation (a b c d e f : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits ∧ f ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f ∧
  a + 10 * b + c = 100 * d + 10 * e + f

theorem largest_three_digit_number :
  ∀ a b c d e f : Nat,
    valid_equation a b c d e f →
    100 * d + 10 * e + f ≤ 105 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_number_l285_28564


namespace NUMINAMATH_CALUDE_fraction_sum_simplification_l285_28550

theorem fraction_sum_simplification :
  3 / 840 + 37 / 120 = 131 / 420 := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_simplification_l285_28550


namespace NUMINAMATH_CALUDE_unique_a_value_l285_28563

def A (a : ℝ) : Set ℝ := {1, 3, a^2}
def B (a : ℝ) : Set ℝ := {1, a+2}

theorem unique_a_value : ∀ a : ℝ, A a ∩ B a = B a → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_a_value_l285_28563


namespace NUMINAMATH_CALUDE_doughnuts_per_box_l285_28567

theorem doughnuts_per_box (total_doughnuts : ℕ) (num_boxes : ℕ) 
  (h1 : total_doughnuts = 48)
  (h2 : num_boxes = 4)
  (h3 : total_doughnuts % num_boxes = 0) : 
  total_doughnuts / num_boxes = 12 := by
sorry

end NUMINAMATH_CALUDE_doughnuts_per_box_l285_28567


namespace NUMINAMATH_CALUDE_max_value_a_l285_28576

theorem max_value_a (a b c d : ℕ+) 
  (h1 : a < 3 * b)
  (h2 : b < 4 * c)
  (h3 : c < 5 * d)
  (h4 : d < 80) :
  a ≤ 4724 ∧ ∃ (a' b' c' d' : ℕ+), 
    a' = 4724 ∧ 
    a' < 3 * b' ∧ 
    b' < 4 * c' ∧ 
    c' < 5 * d' ∧ 
    d' < 80 :=
sorry

end NUMINAMATH_CALUDE_max_value_a_l285_28576


namespace NUMINAMATH_CALUDE_range_of_a_for_surjective_f_l285_28586

/-- A piecewise function f dependent on parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (1 - 2*a)*x + 3*a else 2^(x - 1)

/-- The theorem stating the relationship between the range of f and the range of a -/
theorem range_of_a_for_surjective_f :
  (∀ a : ℝ, Set.range (f a) = Set.univ) ↔ (Set.Icc 0 (1/2) : Set ℝ) = {a : ℝ | 0 ≤ a ∧ a < 1/2} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_surjective_f_l285_28586


namespace NUMINAMATH_CALUDE_small_rhombus_area_l285_28556

theorem small_rhombus_area (r : ℝ) (h : r = 10) : 
  let large_rhombus_diagonal := 2 * r
  let small_rhombus_side := large_rhombus_diagonal / 2
  small_rhombus_side ^ 2 = 100 := by sorry

end NUMINAMATH_CALUDE_small_rhombus_area_l285_28556


namespace NUMINAMATH_CALUDE_left_number_20th_row_l285_28559

/- Define the sequence of numbers in the array -/
def array_sequence (n : ℕ) : ℕ := n^2

/- Define the sum of numbers in the first n rows -/
def sum_of_rows (n : ℕ) : ℕ := n^2

/- Define the number on the far left of the nth row -/
def left_number (n : ℕ) : ℕ := sum_of_rows (n - 1) + 1

/- Theorem statement -/
theorem left_number_20th_row : left_number 20 = 362 := by
  sorry

end NUMINAMATH_CALUDE_left_number_20th_row_l285_28559


namespace NUMINAMATH_CALUDE_frustum_volume_l285_28597

/-- The volume of a frustum formed by cutting a square pyramid --/
theorem frustum_volume (base_edge : ℝ) (altitude : ℝ) (small_base_edge : ℝ) (small_altitude : ℝ)
  (h1 : base_edge = 10)
  (h2 : altitude = 10)
  (h3 : small_base_edge = 5)
  (h4 : small_altitude = 5) :
  (base_edge ^ 2 * altitude / 3) - (small_base_edge ^ 2 * small_altitude / 3) = 875 / 3 := by
  sorry

end NUMINAMATH_CALUDE_frustum_volume_l285_28597


namespace NUMINAMATH_CALUDE_PQ_length_l285_28548

-- Define the point R
def R : ℝ × ℝ := (10, 8)

-- Define the lines
def line1 (x y : ℝ) : Prop := 7 * y = 24 * x
def line2 (x y : ℝ) : Prop := 13 * y = 5 * x

-- Define P and Q
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry

-- State that P is on line1
axiom P_on_line1 : line1 P.1 P.2

-- State that Q is on line2
axiom Q_on_line2 : line2 Q.1 Q.2

-- R is the midpoint of PQ
axiom R_midpoint : R = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Theorem to prove
theorem PQ_length : 
  Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) = 4648 / 277 := by sorry

end NUMINAMATH_CALUDE_PQ_length_l285_28548


namespace NUMINAMATH_CALUDE_meal_profit_and_purchase_theorem_l285_28544

/-- Represents the profit for meals A and B -/
structure MealProfit where
  a : ℝ
  b : ℝ

/-- Represents the purchase quantities for meals A and B -/
structure PurchaseQuantity where
  a : ℝ
  b : ℝ

/-- Conditions for the meal profit problem -/
def meal_profit_conditions (p : MealProfit) : Prop :=
  p.a + 2 * p.b = 35 ∧ 2 * p.a + 3 * p.b = 60

/-- Conditions for the meal purchase problem -/
def meal_purchase_conditions (q : PurchaseQuantity) : Prop :=
  q.a + q.b = 1000 ∧ q.a ≤ 3/2 * q.b

/-- The theorem to be proved -/
theorem meal_profit_and_purchase_theorem 
  (p : MealProfit) 
  (q : PurchaseQuantity) 
  (h1 : meal_profit_conditions p) 
  (h2 : meal_purchase_conditions q) :
  p.a = 15 ∧ 
  p.b = 10 ∧ 
  q.a = 600 ∧ 
  q.b = 400 ∧ 
  p.a * q.a + p.b * q.b = 13000 := by
  sorry

end NUMINAMATH_CALUDE_meal_profit_and_purchase_theorem_l285_28544


namespace NUMINAMATH_CALUDE_car_ownership_proof_l285_28579

def total_cars (cathy_cars : ℕ) : ℕ :=
  let carol_cars := 2 * cathy_cars
  let susan_cars := carol_cars - 2
  let lindsey_cars := cathy_cars + 4
  cathy_cars + carol_cars + susan_cars + lindsey_cars

theorem car_ownership_proof (cathy_cars : ℕ) (h : cathy_cars = 5) : total_cars cathy_cars = 32 := by
  sorry

end NUMINAMATH_CALUDE_car_ownership_proof_l285_28579


namespace NUMINAMATH_CALUDE_arccos_cos_eleven_l285_28526

theorem arccos_cos_eleven : 
  Real.arccos (Real.cos 11) = 11 - 4 * Real.pi + 2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_arccos_cos_eleven_l285_28526


namespace NUMINAMATH_CALUDE_tangent_line_equation_l285_28566

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - x^2

-- Define the point of tangency
def P : ℝ × ℝ := (2, 4)

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 2*x

-- Theorem statement
theorem tangent_line_equation :
  let slope := f' P.1
  let tangent_eq (x y : ℝ) := slope * (x - P.1) - (y - P.2)
  tangent_eq = λ x y => 8*x - y - 12 := by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l285_28566


namespace NUMINAMATH_CALUDE_line_equation_correct_l285_28571

/-- A line in the 2D plane represented by its slope and a point it passes through. -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The equation of a line in slope-intercept form (y = mx + b). -/
def lineEquation (l : Line) : ℝ → ℝ := fun x => l.slope * x + (l.point.2 - l.slope * l.point.1)

theorem line_equation_correct (l : Line) : 
  l.slope = 3 ∧ l.point = (-2, 0) → lineEquation l = fun x => 3 * x + 6 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_correct_l285_28571


namespace NUMINAMATH_CALUDE_tan_alpha_minus_pi_fourth_l285_28575

theorem tan_alpha_minus_pi_fourth (α β : Real) 
  (h1 : Real.tan (α + β) = 2)
  (h2 : Real.tan (β + Real.pi / 4) = 3) :
  Real.tan (α - Real.pi / 4) = -1 / 7 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_minus_pi_fourth_l285_28575


namespace NUMINAMATH_CALUDE_cylinder_height_relationship_l285_28519

theorem cylinder_height_relationship (r₁ h₁ r₂ h₂ : ℝ) :
  r₁ > 0 ∧ h₁ > 0 ∧ r₂ > 0 ∧ h₂ > 0 →
  r₂ = 1.2 * r₁ →
  π * r₁^2 * h₁ = π * r₂^2 * h₂ →
  h₁ = 1.44 * h₂ := by
sorry

end NUMINAMATH_CALUDE_cylinder_height_relationship_l285_28519


namespace NUMINAMATH_CALUDE_workshop_workers_count_l285_28570

/-- Given a workshop with workers including technicians and non-technicians,
    prove that the total number of workers is 22 under the following conditions:
    - The average salary of all workers is 850
    - There are 7 technicians with an average salary of 1000
    - The average salary of non-technicians is 780 -/
theorem workshop_workers_count :
  ∀ (W : ℕ) (avg_salary tech_salary nontech_salary : ℚ),
    avg_salary = 850 →
    tech_salary = 1000 →
    nontech_salary = 780 →
    (W : ℚ) * avg_salary = 7 * tech_salary + (W - 7 : ℚ) * nontech_salary →
    W = 22 := by
  sorry


end NUMINAMATH_CALUDE_workshop_workers_count_l285_28570


namespace NUMINAMATH_CALUDE_borrowed_sheets_theorem_l285_28536

/-- Represents a collection of sheets with page numbers -/
structure Sheets :=
  (total_sheets : ℕ)
  (total_pages : ℕ)
  (borrowed_sheets : ℕ)

/-- Calculates the average page number of remaining sheets -/
def average_remaining_pages (s : Sheets) : ℚ :=
  let remaining_pages := s.total_pages - 2 * s.borrowed_sheets
  let sum_remaining := (s.total_pages * (s.total_pages + 1) / 2) -
                       (2 * s.borrowed_sheets * (2 * s.borrowed_sheets + 1) / 2)
  sum_remaining / remaining_pages

/-- The main theorem to prove -/
theorem borrowed_sheets_theorem (s : Sheets) 
  (h1 : s.total_sheets = 30)
  (h2 : s.total_pages = 60)
  (h3 : s.borrowed_sheets = 10) :
  average_remaining_pages s = 25 := by
  sorry

#eval average_remaining_pages ⟨30, 60, 10⟩

end NUMINAMATH_CALUDE_borrowed_sheets_theorem_l285_28536


namespace NUMINAMATH_CALUDE_ellipse_equation_l285_28532

/-- Given an ellipse with standard form equation, prove its specific equation -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∃ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1) → -- standard form of ellipse
  (3^2 = a^2 - b^2) →                    -- condition for right focus at (3,0)
  (9/b^2 = 1) →                          -- condition for point (0,-3) on ellipse
  (∀ (x y : ℝ), x^2/18 + y^2/9 = 1 ↔ x^2/a^2 + y^2/b^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l285_28532


namespace NUMINAMATH_CALUDE_debby_dvd_count_l285_28522

/-- The number of DVDs Debby sold -/
def sold_dvds : ℕ := 6

/-- The number of DVDs Debby had left after selling -/
def remaining_dvds : ℕ := 7

/-- The initial number of DVDs Debby owned -/
def initial_dvds : ℕ := sold_dvds + remaining_dvds

theorem debby_dvd_count : initial_dvds = 13 := by sorry

end NUMINAMATH_CALUDE_debby_dvd_count_l285_28522


namespace NUMINAMATH_CALUDE_correct_num_persons_first_group_l285_28538

/-- The number of persons in the first group that can repair a road -/
def num_persons_first_group : ℕ := 39

/-- The number of days the first group works -/
def days_first_group : ℕ := 12

/-- The number of hours per day the first group works -/
def hours_per_day_first_group : ℕ := 5

/-- The number of persons in the second group -/
def num_persons_second_group : ℕ := 30

/-- The number of days the second group works -/
def days_second_group : ℕ := 13

/-- The number of hours per day the second group works -/
def hours_per_day_second_group : ℕ := 6

/-- Theorem stating that the number of persons in the first group is correct -/
theorem correct_num_persons_first_group :
  num_persons_first_group * days_first_group * hours_per_day_first_group =
  num_persons_second_group * days_second_group * hours_per_day_second_group :=
by sorry

end NUMINAMATH_CALUDE_correct_num_persons_first_group_l285_28538


namespace NUMINAMATH_CALUDE_solution_set_of_equations_l285_28554

theorem solution_set_of_equations (x y : ℝ) :
  (x^2 - 2*x*y = 1 ∧ 5*x^2 - 2*x*y + 2*y^2 = 5) ↔
  ((x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = 0) ∨ (x = 1/3 ∧ y = -4/3) ∨ (x = -1/3 ∧ y = 4/3)) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_equations_l285_28554


namespace NUMINAMATH_CALUDE_no_base6_digit_divisible_by_7_l285_28546

/-- Converts a base-6 number to base-10 --/
def base6ToBase10 (d : ℕ) : ℕ := 3 * 6^3 + d * 6^2 + d * 6 + 6

/-- Represents a base-6 digit --/
def isBase6Digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 5

theorem no_base6_digit_divisible_by_7 : 
  ∀ d : ℕ, isBase6Digit d → ¬(base6ToBase10 d % 7 = 0) := by
  sorry

#check no_base6_digit_divisible_by_7

end NUMINAMATH_CALUDE_no_base6_digit_divisible_by_7_l285_28546


namespace NUMINAMATH_CALUDE_common_number_in_overlapping_sets_l285_28599

theorem common_number_in_overlapping_sets (numbers : List ℝ) : 
  numbers.length = 9 →
  (numbers.take 5).sum / 5 = 7 →
  (numbers.drop 4).sum / 5 = 10 →
  numbers.sum / 9 = 74 / 9 →
  ∃ x ∈ numbers.take 5 ∩ numbers.drop 4, x = 11 :=
by sorry

end NUMINAMATH_CALUDE_common_number_in_overlapping_sets_l285_28599


namespace NUMINAMATH_CALUDE_dogs_count_l285_28535

/-- Represents the number of animals in a pet shop -/
structure PetShop where
  dogs : ℕ
  cats : ℕ
  bunnies : ℕ

/-- The ratio of dogs to cats to bunnies is 4 : 7 : 9 -/
def ratio_condition (shop : PetShop) : Prop :=
  ∃ (x : ℕ), shop.dogs = 4 * x ∧ shop.cats = 7 * x ∧ shop.bunnies = 9 * x

/-- The total number of dogs and bunnies is 364 -/
def total_condition (shop : PetShop) : Prop :=
  shop.dogs + shop.bunnies = 364

/-- Theorem stating that under the given conditions, there are 112 dogs -/
theorem dogs_count (shop : PetShop) 
  (h_ratio : ratio_condition shop) 
  (h_total : total_condition shop) : 
  shop.dogs = 112 := by
  sorry

end NUMINAMATH_CALUDE_dogs_count_l285_28535


namespace NUMINAMATH_CALUDE_smallest_gcd_of_multiples_l285_28530

theorem smallest_gcd_of_multiples (a b : ℕ+) (h : Nat.gcd a b = 18) :
  ∃ (m n : ℕ+), 12 * m = 12 * a ∧ 20 * n = 20 * b ∧ 
    Nat.gcd (12 * m) (20 * n) = 72 ∧
    ∀ (x y : ℕ+), 12 * x = 12 * a → 20 * y = 20 * b → 
      Nat.gcd (12 * x) (20 * y) ≥ 72 :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_of_multiples_l285_28530


namespace NUMINAMATH_CALUDE_walnut_trees_planted_l285_28555

/-- The number of walnut trees planted in a park -/
def trees_planted (initial final : ℕ) : ℕ := final - initial

/-- Theorem: The number of walnut trees planted is the difference between
    the final number of trees and the initial number of trees -/
theorem walnut_trees_planted (initial final planted : ℕ) 
  (h1 : initial = 107)
  (h2 : final = 211)
  (h3 : planted = trees_planted initial final) :
  planted = 104 := by
  sorry

end NUMINAMATH_CALUDE_walnut_trees_planted_l285_28555


namespace NUMINAMATH_CALUDE_shekar_biology_score_l285_28589

/-- Represents a student's scores in various subjects -/
structure StudentScores where
  mathematics : ℕ
  science : ℕ
  socialStudies : ℕ
  english : ℕ
  biology : ℕ

/-- Calculates the average score given a StudentScores instance -/
def calculateAverage (scores : StudentScores) : ℚ :=
  (scores.mathematics + scores.science + scores.socialStudies + scores.english + scores.biology) / 5

/-- Theorem: Given Shekar's scores and average, his biology score must be 95 -/
theorem shekar_biology_score :
  ∀ (scores : StudentScores),
    scores.mathematics = 76 →
    scores.science = 65 →
    scores.socialStudies = 82 →
    scores.english = 67 →
    calculateAverage scores = 77 →
    scores.biology = 95 := by
  sorry


end NUMINAMATH_CALUDE_shekar_biology_score_l285_28589


namespace NUMINAMATH_CALUDE_chord_intercept_theorem_l285_28545

def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 4*y - 20 = 0

def line_equation (x y c : ℝ) : Prop :=
  5*x - 12*y + c = 0

def chord_length (c : ℝ) : ℝ := 8

theorem chord_intercept_theorem (c : ℝ) :
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_equation x₁ y₁ ∧ circle_equation x₂ y₂ ∧
    line_equation x₁ y₁ c ∧ line_equation x₂ y₂ c ∧
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = (chord_length c)^2) ↔
  c = 10 ∨ c = -68 :=
sorry

end NUMINAMATH_CALUDE_chord_intercept_theorem_l285_28545


namespace NUMINAMATH_CALUDE_hexagon_ratio_is_two_l285_28593

/-- Represents a hexagon with specific properties -/
structure Hexagon where
  /-- Total number of unit squares in the hexagon -/
  total_squares : ℕ
  /-- Number of unit squares above the diagonal PQ -/
  squares_above_pq : ℕ
  /-- Base length of the triangle above PQ -/
  triangle_base : ℝ
  /-- Total length of XQ + QY -/
  xq_plus_qy : ℝ
  /-- Condition: The area above PQ is half of the total area -/
  area_condition : squares_above_pq + (triangle_base * triangle_base / 4) = (total_squares + triangle_base * triangle_base / 4) / 2

/-- Theorem: For a hexagon with the given properties, XQ/QY = 2 -/
theorem hexagon_ratio_is_two (h : Hexagon) (h_total : h.total_squares = 8) 
  (h_above : h.squares_above_pq = 3) (h_base : h.triangle_base = 4) (h_xq_qy : h.xq_plus_qy = 4) : 
  ∃ (xq qy : ℝ), xq + qy = h.xq_plus_qy ∧ xq / qy = 2 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_ratio_is_two_l285_28593


namespace NUMINAMATH_CALUDE_willie_stickers_l285_28503

/-- Given Willie starts with 124 stickers and gives away 23, prove he ends up with 101 stickers. -/
theorem willie_stickers : 
  let initial_stickers : ℕ := 124
  let given_away : ℕ := 23
  initial_stickers - given_away = 101 := by
  sorry

end NUMINAMATH_CALUDE_willie_stickers_l285_28503


namespace NUMINAMATH_CALUDE_window_width_calculation_l285_28557

/-- Represents the dimensions of a glass pane -/
structure Pane where
  height : ℝ
  width : ℝ

/-- Represents the dimensions of a window -/
structure Window where
  rows : ℕ
  columns : ℕ
  pane : Pane
  border_width : ℝ

/-- Calculates the width of a window given its specifications -/
def window_width (w : Window) : ℝ :=
  w.columns * w.pane.width + (w.columns + 1) * w.border_width

/-- Theorem stating the width of the window with given specifications -/
theorem window_width_calculation (x : ℝ) :
  let w : Window := {
    rows := 3,
    columns := 4,
    pane := { height := 4 * x, width := 3 * x },
    border_width := 3
  }
  window_width w = 12 * x + 15 := by sorry

end NUMINAMATH_CALUDE_window_width_calculation_l285_28557


namespace NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l285_28591

theorem unique_number_satisfying_conditions : ∃! n : ℕ,
  35 < n ∧ n < 70 ∧ (n - 3) % 6 = 0 ∧ (n - 1) % 8 = 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l285_28591


namespace NUMINAMATH_CALUDE_negative_five_plus_abs_negative_three_equals_negative_two_l285_28587

theorem negative_five_plus_abs_negative_three_equals_negative_two :
  -5 + |(-3)| = -2 := by
  sorry

end NUMINAMATH_CALUDE_negative_five_plus_abs_negative_three_equals_negative_two_l285_28587


namespace NUMINAMATH_CALUDE_max_value_of_s_l285_28517

theorem max_value_of_s (x y : ℝ) (h : 4 * x^2 - 5 * x * y + 4 * y^2 = 5) :
  ∃ (s_max : ℝ), s_max = (10 : ℝ) / 3 ∧ ∀ (s : ℝ), s = x^2 + y^2 → s ≤ s_max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_s_l285_28517


namespace NUMINAMATH_CALUDE_valid_pairs_l285_28549

def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n

def valid_pair (a b : ℕ) : Prop :=
  is_integer ((a^2 + b) / (b^2 - a)) ∧
  is_integer ((b^2 + a) / (a^2 - b))

theorem valid_pairs :
  ∀ a b : ℕ, valid_pair a b ↔
    ((a = 1 ∧ b = 2) ∨
     (a = 2 ∧ b = 1) ∨
     (a = 2 ∧ b = 2) ∨
     (a = 2 ∧ b = 3) ∨
     (a = 3 ∧ b = 2) ∨
     (a = 3 ∧ b = 3)) :=
sorry

end NUMINAMATH_CALUDE_valid_pairs_l285_28549


namespace NUMINAMATH_CALUDE_animal_shelter_dogs_l285_28541

theorem animal_shelter_dogs (initial_dogs : ℕ) (adoption_rate : ℚ) (returned_dogs : ℕ) : 
  initial_dogs = 80 → 
  adoption_rate = 2/5 →
  returned_dogs = 5 →
  initial_dogs - (initial_dogs * adoption_rate).floor + returned_dogs = 53 := by
sorry

end NUMINAMATH_CALUDE_animal_shelter_dogs_l285_28541


namespace NUMINAMATH_CALUDE_flower_pots_theorem_l285_28520

/-- Represents the number of pots of each type of flower seedling --/
structure FlowerPots where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Checks if the given FlowerPots satisfies all conditions --/
def isValid (pots : FlowerPots) : Prop :=
  pots.a > 0 ∧ pots.b > 0 ∧ pots.c > 0 ∧
  pots.a + pots.b + pots.c = 16 ∧
  2 * pots.a + 4 * pots.b + 10 * pots.c = 50

/-- The theorem stating that the only valid numbers of pots for type A are 10 and 13 --/
theorem flower_pots_theorem :
  ∀ pots : FlowerPots, isValid pots → pots.a = 10 ∨ pots.a = 13 := by
  sorry

end NUMINAMATH_CALUDE_flower_pots_theorem_l285_28520


namespace NUMINAMATH_CALUDE_horse_oats_consumption_l285_28595

theorem horse_oats_consumption 
  (num_horses : ℕ) 
  (meals_per_day : ℕ) 
  (days : ℕ) 
  (total_oats : ℕ) 
  (h1 : num_horses = 4) 
  (h2 : meals_per_day = 2) 
  (h3 : days = 3) 
  (h4 : total_oats = 96) : 
  total_oats / (num_horses * meals_per_day * days) = 4 := by
sorry

end NUMINAMATH_CALUDE_horse_oats_consumption_l285_28595


namespace NUMINAMATH_CALUDE_factorization_problem_fraction_simplification_l285_28588

-- Factorization problem
theorem factorization_problem (m : ℝ) : m^3 - 4*m^2 + 4*m = m*(m-2)^2 := by
  sorry

-- Fraction simplification problem
theorem fraction_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  2 / (x^2 - 1) - 1 / (x - 1) = -1 / (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problem_fraction_simplification_l285_28588


namespace NUMINAMATH_CALUDE_ratio_equality_l285_28539

theorem ratio_equality (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_diff : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h_eq : y / (x + z) = (x + 2*y) / (z + 2*y) ∧ (x + 2*y) / (z + 2*y) = x / (2*y)) :
  x / y = 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l285_28539


namespace NUMINAMATH_CALUDE_max_value_problem_l285_28505

theorem max_value_problem (x y : ℝ) 
  (h1 : x - y ≥ 2) 
  (h2 : x + y ≤ 3) 
  (h3 : x ≥ 0) 
  (h4 : y ≥ 0) : 
  ∃ (z : ℝ), z = 6 ∧ ∀ (w : ℝ), w = 2*x - 3*y → w ≤ z :=
by sorry

end NUMINAMATH_CALUDE_max_value_problem_l285_28505


namespace NUMINAMATH_CALUDE_max_area_and_optimal_length_l285_28507

/-- Represents the dimensions and cost of a simple house. -/
structure SimpleHouse where
  x : ℝ  -- Length of front wall
  y : ℝ  -- Length of side wall
  h : ℝ  -- Height of walls
  colorSteelPrice : ℝ  -- Price per meter of color steel
  compositeSteelPrice : ℝ  -- Price per meter of composite steel
  roofPrice : ℝ  -- Price per square meter of roof material
  maxCost : ℝ  -- Maximum allowed cost

/-- Calculates the total material cost of the house. -/
def materialCost (h : SimpleHouse) : ℝ :=
  2 * h.x * h.colorSteelPrice * h.h + 
  2 * h.y * h.compositeSteelPrice * h.h + 
  h.x * h.y * h.roofPrice

/-- Calculates the area of the house. -/
def area (h : SimpleHouse) : ℝ := h.x * h.y

/-- Theorem stating the maximum area and optimal front wall length. -/
theorem max_area_and_optimal_length (h : SimpleHouse) 
    (h_height : h.h = 2.5)
    (h_colorSteel : h.colorSteelPrice = 450)
    (h_compositeSteel : h.compositeSteelPrice = 200)
    (h_roof : h.roofPrice = 200)
    (h_maxCost : h.maxCost = 32000)
    (h_cost_constraint : materialCost h ≤ h.maxCost) :
    ∃ (maxArea : ℝ) (optimalLength : ℝ),
      maxArea = 100 ∧
      optimalLength = 20 / 3 ∧
      ∀ (x y : ℝ), 
        x > 0 → y > 0 → 
        materialCost { h with x := x, y := y } ≤ h.maxCost →
        area { h with x := x, y := y } ≤ maxArea ∧
        (area { h with x := x, y := y } = maxArea → x = optimalLength) :=
  sorry

end NUMINAMATH_CALUDE_max_area_and_optimal_length_l285_28507


namespace NUMINAMATH_CALUDE_cubic_polynomial_roots_l285_28521

/-- A cubic polynomial with rational coefficients -/
structure CubicPolynomial where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Predicate to check if a, b, c are roots of the polynomial -/
def has_roots (p : CubicPolynomial) : Prop :=
  (p.a^3 + p.a * p.a^2 + p.b * p.a + p.c = 0) ∧
  (p.b^3 + p.a * p.b^2 + p.b * p.b + p.c = 0) ∧
  (p.c^3 + p.a * p.c^2 + p.b * p.c + p.c = 0)

/-- The set of valid polynomials -/
def valid_polynomials : Set CubicPolynomial :=
  {⟨0, 0, 0⟩, ⟨1, -2, 0⟩}

theorem cubic_polynomial_roots (p : CubicPolynomial) :
  has_roots p ↔ p ∈ valid_polynomials := by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_roots_l285_28521


namespace NUMINAMATH_CALUDE_baking_scoops_calculation_l285_28542

/-- Calculates the total number of scoops needed for baking a cake --/
def total_scoops (flour_cups : ℚ) (sugar_cups : ℚ) (scoop_size : ℚ) : ℕ :=
  (flour_cups / scoop_size + sugar_cups / scoop_size).ceil.toNat

/-- Proves that given 3 cups of flour, 2 cups of sugar, and a 1/3 cup scoop, 
    the total number of scoops needed is 15 --/
theorem baking_scoops_calculation : 
  total_scoops 3 2 (1/3) = 15 := by sorry

end NUMINAMATH_CALUDE_baking_scoops_calculation_l285_28542


namespace NUMINAMATH_CALUDE_water_amount_in_new_recipe_l285_28551

/-- Represents the ratio of ingredients in a recipe -/
structure RecipeRatio where
  flour : ℚ
  water : ℚ
  sugar : ℚ

/-- The original recipe ratio -/
def original_ratio : RecipeRatio := ⟨7, 2, 1⟩

/-- The new recipe ratio -/
def new_ratio : RecipeRatio :=
  let flour_water_ratio := original_ratio.flour / original_ratio.water
  let flour_sugar_ratio := original_ratio.flour / original_ratio.sugar
  ⟨original_ratio.flour,
   original_ratio.flour / (2 * flour_water_ratio),
   original_ratio.flour / (flour_sugar_ratio / 2)⟩

/-- The amount of sugar in the new recipe -/
def sugar_amount : ℚ := 4

theorem water_amount_in_new_recipe :
  (sugar_amount * new_ratio.water / new_ratio.sugar) = 2 := by
  sorry

end NUMINAMATH_CALUDE_water_amount_in_new_recipe_l285_28551


namespace NUMINAMATH_CALUDE_multiple_p_values_exist_l285_28515

theorem multiple_p_values_exist : ∃ p₁ p₂ : ℝ, 
  0 < p₁ ∧ p₁ < 1 ∧ 
  0 < p₂ ∧ p₂ < 1 ∧ 
  p₁ ≠ p₂ ∧
  (Nat.choose 5 3 : ℝ) * p₁^3 * (1 - p₁)^2 = 144/625 ∧
  (Nat.choose 5 3 : ℝ) * p₂^3 * (1 - p₂)^2 = 144/625 :=
by sorry

end NUMINAMATH_CALUDE_multiple_p_values_exist_l285_28515


namespace NUMINAMATH_CALUDE_fraction_simplification_l285_28568

theorem fraction_simplification :
  (3 : ℝ) / (2 * Real.sqrt 50 + 3 * Real.sqrt 8 - Real.sqrt 18) = (3 * Real.sqrt 2) / 26 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l285_28568


namespace NUMINAMATH_CALUDE_expected_black_pairs_in_circular_arrangement_l285_28582

/-- The number of cards in the modified deck -/
def total_cards : ℕ := 60

/-- The number of black cards in the deck -/
def black_cards : ℕ := 30

/-- The number of red cards in the deck -/
def red_cards : ℕ := 30

/-- The expected number of pairs of adjacent black cards in a circular arrangement -/
def expected_black_pairs : ℚ := 870 / 59

theorem expected_black_pairs_in_circular_arrangement :
  let total := total_cards
  let black := black_cards
  let red := red_cards
  total = black + red →
  expected_black_pairs = (black * (black - 1) : ℚ) / (total - 1) := by
  sorry

end NUMINAMATH_CALUDE_expected_black_pairs_in_circular_arrangement_l285_28582


namespace NUMINAMATH_CALUDE_total_area_calculation_l285_28516

/-- Calculates the total area of rooms given initial dimensions and modifications --/
theorem total_area_calculation (initial_length initial_width increase : ℕ) : 
  let new_length : ℕ := initial_length + increase
  let new_width : ℕ := initial_width + increase
  let single_room_area : ℕ := new_length * new_width
  let total_area : ℕ := 4 * single_room_area + 2 * single_room_area
  (initial_length = 13 ∧ initial_width = 18 ∧ increase = 2) → total_area = 1800 := by
  sorry

#check total_area_calculation

end NUMINAMATH_CALUDE_total_area_calculation_l285_28516


namespace NUMINAMATH_CALUDE_max_product_constraint_l285_28509

theorem max_product_constraint (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a/4 + b/5 = 1) :
  a * b ≤ 5 := by
sorry

end NUMINAMATH_CALUDE_max_product_constraint_l285_28509


namespace NUMINAMATH_CALUDE_a_plus_b_values_l285_28533

theorem a_plus_b_values (a b : ℝ) : 
  (abs a = 1) → (b = -2) → ((a + b = -1) ∨ (a + b = -3)) :=
by sorry

end NUMINAMATH_CALUDE_a_plus_b_values_l285_28533


namespace NUMINAMATH_CALUDE_second_hand_movement_l285_28596

/-- Represents the number of seconds it takes for the second hand to move from one number to another on a clock face -/
def secondsBetweenNumbers (start finish : Nat) : Nat :=
  ((finish - start + 12) % 12) * 5

theorem second_hand_movement : secondsBetweenNumbers 5 9 ≠ 4 := by
  sorry

end NUMINAMATH_CALUDE_second_hand_movement_l285_28596


namespace NUMINAMATH_CALUDE_justice_plants_l285_28510

/-- The number of plants Justice wants in her home -/
def desired_plants (ferns palms succulents additional : ℕ) : ℕ :=
  ferns + palms + succulents + additional

/-- Theorem stating the total number of plants Justice wants -/
theorem justice_plants : 
  desired_plants 3 5 7 9 = 24 := by
  sorry

end NUMINAMATH_CALUDE_justice_plants_l285_28510


namespace NUMINAMATH_CALUDE_janet_complaint_time_l285_28523

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The time Janet spends looking for her keys daily (in minutes) -/
def daily_key_search_time : ℕ := 8

/-- The total time Janet saves weekly by not losing her keys (in minutes) -/
def weekly_time_saved : ℕ := 77

/-- The time Janet spends complaining after finding her keys daily (in minutes) -/
def daily_complaint_time : ℕ := (weekly_time_saved - days_in_week * daily_key_search_time) / days_in_week

theorem janet_complaint_time :
  daily_complaint_time = 3 :=
sorry

end NUMINAMATH_CALUDE_janet_complaint_time_l285_28523


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_condition_l285_28527

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between planes
variable (plane_parallel : Plane → Plane → Prop)

-- Define the parallel relation between a line and a plane
variable (line_parallel_to_plane : Line → Plane → Prop)

-- Define the containment relation of a line in a plane
variable (line_in_plane : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_plane_condition 
  (a : Line) (α : Plane) :
  (∃ β : Plane, line_in_plane a β ∧ plane_parallel α β) →
  line_parallel_to_plane a α :=
by sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_condition_l285_28527


namespace NUMINAMATH_CALUDE_component_reliability_l285_28506

/-- Represents the service life of an electronic component in years -/
def ServiceLife : Type := ℝ

/-- The probability that a single electronic component works normally for more than 9 years -/
def ProbSingleComponentWorksOver9Years : ℝ := 0.2

/-- The number of electronic components in parallel -/
def NumComponents : ℕ := 3

/-- The probability that the component (made up of 3 parallel electronic components) 
    can work normally for more than 9 years -/
def ProbComponentWorksOver9Years : ℝ :=
  1 - (1 - ProbSingleComponentWorksOver9Years) ^ NumComponents

theorem component_reliability :
  ProbComponentWorksOver9Years = 0.488 :=
sorry

end NUMINAMATH_CALUDE_component_reliability_l285_28506


namespace NUMINAMATH_CALUDE_mp3_player_problem_l285_28537

def initial_songs : Nat := 8
def deleted_songs : Nat := 5
def added_songs : Nat := 30
def added_song_durations : List Nat := [3, 4, 2, 6, 5, 3, 4, 5, 6, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 8, 4, 3, 5, 6, 7, 8, 9, 10]

theorem mp3_player_problem :
  (initial_songs - deleted_songs + added_songs = 33) ∧
  (added_song_durations.sum = 145) := by
  sorry

end NUMINAMATH_CALUDE_mp3_player_problem_l285_28537


namespace NUMINAMATH_CALUDE_modified_short_bingo_arrangements_l285_28561

theorem modified_short_bingo_arrangements : Nat.factorial 15 / Nat.factorial 8 = 1816214400 := by
  sorry

end NUMINAMATH_CALUDE_modified_short_bingo_arrangements_l285_28561


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_properties_l285_28558

theorem quadratic_equation_roots_properties : ∃ (r₁ r₂ : ℝ),
  (r₁^2 - 6*r₁ + 8 = 0) ∧
  (r₂^2 - 6*r₂ + 8 = 0) ∧
  (r₁ ≠ r₂) ∧
  (|r₁ - r₂| = 2) ∧
  (|r₁| + |r₂| = 6) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_properties_l285_28558


namespace NUMINAMATH_CALUDE_orthocenter_ratio_l285_28584

-- Define the triangle XYZ
structure Triangle (X Y Z : ℝ × ℝ) : Prop where
  -- Add any necessary conditions for a valid triangle

-- Define the length of a side
def side_length (A B : ℝ × ℝ) : ℝ := sorry

-- Define the measure of an angle
def angle_measure (A B C : ℝ × ℝ) : ℝ := sorry

-- Define an altitude of a triangle
def altitude (A B C P : ℝ × ℝ) : Prop := sorry

-- Define the orthocenter of a triangle
def orthocenter (X Y Z H : ℝ × ℝ) : Prop := sorry

-- Main theorem
theorem orthocenter_ratio {X Y Z P H : ℝ × ℝ} :
  Triangle X Y Z →
  side_length Y Z = 5 →
  side_length X Z = 4 * Real.sqrt 2 →
  angle_measure X Z Y = π / 4 →
  altitude X Y Z P →
  orthocenter X Y Z H →
  (side_length X H) / (side_length H P) = 3 := by
  sorry

end NUMINAMATH_CALUDE_orthocenter_ratio_l285_28584


namespace NUMINAMATH_CALUDE_min_sum_absolute_differences_l285_28543

theorem min_sum_absolute_differences (a : ℚ) : 
  ∃ (min : ℚ), min = 4 ∧ ∀ (x : ℚ), |x-1| + |x-2| + |x-3| + |x-4| ≥ min := by
  sorry

end NUMINAMATH_CALUDE_min_sum_absolute_differences_l285_28543


namespace NUMINAMATH_CALUDE_total_toys_l285_28512

/-- The number of toys each child has -/
structure ToyCount where
  jerry : ℕ
  gabriel : ℕ
  jaxon : ℕ

/-- The conditions of the problem -/
def toy_conditions (t : ToyCount) : Prop :=
  t.jerry = t.gabriel + 8 ∧
  t.gabriel = 2 * t.jaxon ∧
  t.jaxon = 15

/-- The theorem stating the total number of toys -/
theorem total_toys (t : ToyCount) (h : toy_conditions t) : 
  t.jerry + t.gabriel + t.jaxon = 83 := by
  sorry

end NUMINAMATH_CALUDE_total_toys_l285_28512


namespace NUMINAMATH_CALUDE_wall_building_time_l285_28578

/-- Represents the time taken to build a wall given different workforce scenarios -/
theorem wall_building_time
  (original_men : ℕ)
  (original_days : ℕ)
  (new_total_men : ℕ)
  (fast_men : ℕ)
  (h1 : original_men = 20)
  (h2 : original_days = 6)
  (h3 : new_total_men = 30)
  (h4 : fast_men = 10)
  (h5 : fast_men ≤ new_total_men) :
  let effective_workforce := new_total_men - fast_men + 2 * fast_men
  let new_days := (original_men * original_days) / effective_workforce
  new_days = 3 := by sorry

end NUMINAMATH_CALUDE_wall_building_time_l285_28578


namespace NUMINAMATH_CALUDE_triangle_side_length_l285_28528

/-- Given a triangle ABC where the internal angles form an arithmetic sequence,
    and sides a = 4 and c = 3, prove that the length of side b is √13 -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A + B + C = Real.pi →  -- Sum of angles in a triangle
  B = (A + C) / 2 →      -- Angles form an arithmetic sequence
  a = 4 →                -- Length of side a
  c = 3 →                -- Length of side c
  b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) →  -- Cosine rule
  b = Real.sqrt 13 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l285_28528


namespace NUMINAMATH_CALUDE_tan_negative_405_degrees_l285_28594

theorem tan_negative_405_degrees : Real.tan ((-405 : ℝ) * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_405_degrees_l285_28594


namespace NUMINAMATH_CALUDE_infinitely_many_a_composite_sum_l285_28572

theorem infinitely_many_a_composite_sum : ∃ f : ℕ → ℕ, 
  (∀ k : ℕ, f k > f (k - 1)) ∧ 
  (∀ a : ℕ, ∃ m : ℕ, a = f m) ∧
  (∀ a : ℕ, a ∈ Set.range f → ∀ n : ℕ, ∃ x y : ℕ, x > 1 ∧ y > 1 ∧ n^4 + a = x * y) :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_a_composite_sum_l285_28572


namespace NUMINAMATH_CALUDE_intersection_of_perpendicular_lines_l285_28513

/-- Given two lines in a plane, where one is perpendicular to the other and passes through a specific point, this theorem proves that their intersection point is as calculated. -/
theorem intersection_of_perpendicular_lines 
  (line1 : ℝ → ℝ)
  (line2 : ℝ → ℝ)
  (h1 : ∀ x, line1 x = -3 * x + 4)
  (h2 : ∀ x, line2 x = (1/3) * x - 1)
  (h3 : line2 3 = -2)
  (h4 : ∀ x y, line1 x = y → line2 x = y → x = 1.5 ∧ y = -0.5) :
  ∃ x y, line1 x = y ∧ line2 x = y ∧ x = 1.5 ∧ y = -0.5 := by
sorry


end NUMINAMATH_CALUDE_intersection_of_perpendicular_lines_l285_28513


namespace NUMINAMATH_CALUDE_restore_original_example_l285_28580

def original_product : ℕ := 4 * 5 * 4 * 5 * 4

def changed_product : ℕ := 2247

def num_changed_digits : ℕ := 2

theorem restore_original_example :
  (original_product = 2240) ∧
  (∃ (a b : ℕ), a ≠ b ∧ a ≤ 9 ∧ b ≤ 9 ∧
    changed_product = original_product + a * 10 - b) :=
sorry

end NUMINAMATH_CALUDE_restore_original_example_l285_28580


namespace NUMINAMATH_CALUDE_new_room_size_l285_28553

theorem new_room_size (bedroom_size : ℝ) (bathroom_size : ℝ) 
  (h1 : bedroom_size = 309) 
  (h2 : bathroom_size = 150) : 
  2 * (bedroom_size + bathroom_size) = 918 := by
  sorry

end NUMINAMATH_CALUDE_new_room_size_l285_28553


namespace NUMINAMATH_CALUDE_original_number_proof_l285_28590

theorem original_number_proof (h1 : 213 * 16 = 3408) 
  (h2 : 1.6 * x = 34.080000000000005) : x = 21.3 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l285_28590


namespace NUMINAMATH_CALUDE_cake_remaining_l285_28574

theorem cake_remaining (alex_portion jordan_portion remaining_portion : ℚ) : 
  alex_portion = 40 / 100 →
  jordan_portion = (1 - alex_portion) / 2 →
  remaining_portion = 1 - alex_portion - jordan_portion →
  remaining_portion = 30 / 100 := by
  sorry

end NUMINAMATH_CALUDE_cake_remaining_l285_28574


namespace NUMINAMATH_CALUDE_multiply_99_105_l285_28565

theorem multiply_99_105 : 99 * 105 = 10395 := by
  sorry

end NUMINAMATH_CALUDE_multiply_99_105_l285_28565


namespace NUMINAMATH_CALUDE_expression_evaluation_l285_28592

theorem expression_evaluation :
  let x : ℚ := -2
  let y : ℚ := 1/2
  (x + 2*y)^2 - (x + y)*(x - y) = -11/4 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l285_28592


namespace NUMINAMATH_CALUDE_angle_from_terminal_point_l285_28569

theorem angle_from_terminal_point (θ : Real) :
  θ ∈ Set.Icc 0 (2 * Real.pi) →
  (Real.sin θ = Real.sin (3 * Real.pi / 4) ∧ Real.cos θ = Real.cos (3 * Real.pi / 4)) →
  θ = 7 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_from_terminal_point_l285_28569


namespace NUMINAMATH_CALUDE_valentino_farm_birds_l285_28577

/-- The number of birds on Mr. Valentino's farm -/
def total_birds (chickens ducks turkeys : ℕ) : ℕ := chickens + ducks + turkeys

/-- Theorem stating the total number of birds on Mr. Valentino's farm -/
theorem valentino_farm_birds :
  ∃ (chickens ducks turkeys : ℕ),
    chickens = 200 ∧
    ducks = 2 * chickens ∧
    turkeys = 3 * ducks ∧
    total_birds chickens ducks turkeys = 1800 := by
  sorry

end NUMINAMATH_CALUDE_valentino_farm_birds_l285_28577


namespace NUMINAMATH_CALUDE_two_people_available_l285_28525

-- Define the types for people and days
inductive Person : Type
| Anna : Person
| Bill : Person
| Carl : Person
| Dana : Person

inductive Day : Type
| Monday : Day
| Tuesday : Day
| Wednesday : Day
| Thursday : Day
| Friday : Day
| Saturday : Day

-- Define a function to represent availability
def isAvailable : Person → Day → Bool
| Person.Anna, Day.Monday => false
| Person.Anna, Day.Tuesday => true
| Person.Anna, Day.Wednesday => false
| Person.Anna, Day.Thursday => true
| Person.Anna, Day.Friday => true
| Person.Anna, Day.Saturday => false
| Person.Bill, Day.Monday => true
| Person.Bill, Day.Tuesday => false
| Person.Bill, Day.Wednesday => true
| Person.Bill, Day.Thursday => false
| Person.Bill, Day.Friday => false
| Person.Bill, Day.Saturday => true
| Person.Carl, Day.Monday => false
| Person.Carl, Day.Tuesday => false
| Person.Carl, Day.Wednesday => true
| Person.Carl, Day.Thursday => false
| Person.Carl, Day.Friday => false
| Person.Carl, Day.Saturday => true
| Person.Dana, Day.Monday => true
| Person.Dana, Day.Tuesday => true
| Person.Dana, Day.Wednesday => false
| Person.Dana, Day.Thursday => true
| Person.Dana, Day.Friday => true
| Person.Dana, Day.Saturday => false

-- Define a function to count available people for a given day
def countAvailable (d : Day) : Nat :=
  List.foldl (λ count p => if isAvailable p d then count + 1 else count) 0 [Person.Anna, Person.Bill, Person.Carl, Person.Dana]

-- Theorem: For each day, exactly 2 people can attend the meeting
theorem two_people_available (d : Day) : countAvailable d = 2 := by
  sorry

#eval [Day.Monday, Day.Tuesday, Day.Wednesday, Day.Thursday, Day.Friday, Day.Saturday].map countAvailable

end NUMINAMATH_CALUDE_two_people_available_l285_28525
