import Mathlib

namespace NUMINAMATH_CALUDE_max_value_is_72_l3745_374548

/-- Represents a type of stone with its weight and value -/
structure Stone where
  weight : ℕ
  value : ℕ

/-- The problem setup -/
def stones : List Stone := [
  { weight := 3, value := 9 },
  { weight := 6, value := 15 },
  { weight := 1, value := 1 }
]

/-- The maximum weight Tanya can carry -/
def maxWeight : ℕ := 24

/-- The minimum number of each type of stone available -/
def minStoneCount : ℕ := 10

/-- Calculates the maximum value of stones that can be carried given the constraints -/
def maxValue : ℕ :=
  sorry -- Proof goes here

/-- Theorem stating that the maximum value is 72 -/
theorem max_value_is_72 : maxValue = 72 := by
  sorry -- Proof goes here

end NUMINAMATH_CALUDE_max_value_is_72_l3745_374548


namespace NUMINAMATH_CALUDE_parabola_sum_l3745_374577

/-- A parabola with equation x = ay² + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_sum (p : Parabola) : 
  p.x_coord (-6) = 8 → p.x_coord (-4) = 10 → p.a + p.b + p.c = 32.5 := by
  sorry

end NUMINAMATH_CALUDE_parabola_sum_l3745_374577


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3745_374587

theorem complex_equation_solution (z : ℂ) :
  z / (1 - Complex.I) = Complex.I → z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3745_374587


namespace NUMINAMATH_CALUDE_weight_loss_challenge_l3745_374515

theorem weight_loss_challenge (W : ℝ) (W_pos : W > 0) : 
  let weight_after_loss := W * 0.9
  let weight_with_clothes := weight_after_loss * 1.02
  let measured_loss_percentage := (W - weight_with_clothes) / W * 100
  measured_loss_percentage = 8.2 := by
sorry

end NUMINAMATH_CALUDE_weight_loss_challenge_l3745_374515


namespace NUMINAMATH_CALUDE_equal_areas_of_equal_ratios_l3745_374514

noncomputable def curvilinearTrapezoidArea (a b : ℝ) : ℝ := ∫ x in a..b, (1 / x)

theorem equal_areas_of_equal_ratios (a₁ b₁ a₂ b₂ : ℝ) 
  (ha₁ : 0 < a₁) (hb₁ : a₁ < b₁)
  (ha₂ : 0 < a₂) (hb₂ : a₂ < b₂)
  (h_ratio : b₁ / a₁ = b₂ / a₂) :
  curvilinearTrapezoidArea a₁ b₁ = curvilinearTrapezoidArea a₂ b₂ := by
  sorry

end NUMINAMATH_CALUDE_equal_areas_of_equal_ratios_l3745_374514


namespace NUMINAMATH_CALUDE_function_max_min_range_l3745_374553

open Real

theorem function_max_min_range (m : ℝ) (f : ℝ → ℝ) : 
  (∀ x, f x = m * sin (x + π/4) - Real.sqrt 2 * sin x) → 
  (∃ max min : ℝ, ∀ x ∈ Set.Ioo 0 (7*π/6), f x ≤ max ∧ min ≤ f x) →
  2 < m ∧ m < 3 + Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_function_max_min_range_l3745_374553


namespace NUMINAMATH_CALUDE_largest_divisor_of_m_squared_minus_4n_squared_l3745_374593

theorem largest_divisor_of_m_squared_minus_4n_squared (m n : ℤ) 
  (h_m_odd : Odd m) (h_n_odd : Odd n) (h_m_gt_n : m > n) : 
  (∀ k : ℤ, k ∣ (m^2 - 4*n^2) → k = 1 ∨ k = -1) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_m_squared_minus_4n_squared_l3745_374593


namespace NUMINAMATH_CALUDE_different_answers_for_fedya_question_l3745_374505

-- Define the types of people
inductive Person : Type
| truthTeller : Person
| liar : Person

-- Define the possible answers
inductive Answer : Type
| yes : Answer
| no : Answer

-- Define the function that determines how a person answers
def answerQuestion (p : Person) (isNameFedya : Bool) : Answer :=
  match p with
  | Person.truthTeller => if isNameFedya then Answer.yes else Answer.no
  | Person.liar => if isNameFedya then Answer.no else Answer.yes

-- State the theorem
theorem different_answers_for_fedya_question 
  (fedya : Person) 
  (vadim : Person) 
  (h1 : fedya = Person.truthTeller) 
  (h2 : vadim = Person.liar) :
  answerQuestion fedya true ≠ answerQuestion vadim false :=
sorry

end NUMINAMATH_CALUDE_different_answers_for_fedya_question_l3745_374505


namespace NUMINAMATH_CALUDE_f_is_even_l3745_374504

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 1

-- Theorem stating that f is an even function
theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by
  sorry

end NUMINAMATH_CALUDE_f_is_even_l3745_374504


namespace NUMINAMATH_CALUDE_inequality_solution_implies_a_less_than_three_l3745_374538

theorem inequality_solution_implies_a_less_than_three (a : ℝ) : 
  (∃ x : ℝ, |x + 1| - |x - 2| > a) → a < 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_a_less_than_three_l3745_374538


namespace NUMINAMATH_CALUDE_decagon_diagonal_intersections_eq_choose_l3745_374598

/-- The number of interior intersection points of diagonals in a regular decagon -/
def decagon_diagonal_intersections : ℕ :=
  Nat.choose 10 4

/-- Theorem: The number of interior intersection points of diagonals in a regular decagon
    is equal to the number of ways to choose 4 vertices out of 10 -/
theorem decagon_diagonal_intersections_eq_choose :
  decagon_diagonal_intersections = Nat.choose 10 4 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonal_intersections_eq_choose_l3745_374598


namespace NUMINAMATH_CALUDE_model_airplane_competition_l3745_374545

/-- Represents a model airplane -/
structure ModelAirplane where
  speed : ℝ
  flightTime : ℝ

/-- Theorem about model airplane competition -/
theorem model_airplane_competition 
  (m h c : ℝ) 
  (model1 model2 : ModelAirplane) 
  (h_positive : h > 0)
  (m_positive : m > 0)
  (c_positive : c > 0)
  (time_diff : model2.flightTime = model1.flightTime + m)
  (headwind_distance : 
    (model1.speed - c) * model1.flightTime = 
    (model2.speed - c) * model2.flightTime + h) :
  (h > c * m → 
    model1.speed * model1.flightTime > 
    model2.speed * model2.flightTime) ∧
  (h < c * m → 
    model1.speed * model1.flightTime < 
    model2.speed * model2.flightTime) ∧
  (h = c * m → 
    model1.speed * model1.flightTime = 
    model2.speed * model2.flightTime) := by
  sorry

end NUMINAMATH_CALUDE_model_airplane_competition_l3745_374545


namespace NUMINAMATH_CALUDE_marta_average_earnings_l3745_374590

/-- Represents Marta's work and earnings on her grandparent's farm --/
structure FarmWork where
  total_collected : ℕ
  task_a_rate : ℕ
  task_b_rate : ℕ
  task_c_rate : ℕ
  tips : ℕ
  task_a_hours : ℕ
  task_b_hours : ℕ
  task_c_hours : ℕ

/-- Calculates the average hourly earnings including tips --/
def average_hourly_earnings (work : FarmWork) : ℚ :=
  work.total_collected / (work.task_a_hours + work.task_b_hours + work.task_c_hours)

/-- Theorem stating that Marta's average hourly earnings, including tips, is $16 per hour --/
theorem marta_average_earnings :
  let work := FarmWork.mk 240 12 10 8 50 3 5 7
  average_hourly_earnings work = 16 := by
  sorry


end NUMINAMATH_CALUDE_marta_average_earnings_l3745_374590


namespace NUMINAMATH_CALUDE_haley_balls_count_l3745_374564

/-- Given that each bag can contain 4 balls and 9 bags will be used,
    prove that the number of balls Haley has is equal to 36. -/
theorem haley_balls_count (balls_per_bag : ℕ) (num_bags : ℕ) (h1 : balls_per_bag = 4) (h2 : num_bags = 9) :
  balls_per_bag * num_bags = 36 := by
  sorry

end NUMINAMATH_CALUDE_haley_balls_count_l3745_374564


namespace NUMINAMATH_CALUDE_equation_solutions_l3745_374513

theorem equation_solutions (a b c d : ℤ) (hab : a ≠ b) :
  let f : ℤ × ℤ → ℤ := λ (x, y) ↦ (x + a * y + c) * (x + b * y + d)
  (∃ S : Finset (ℤ × ℤ), (∀ p ∈ S, f p = 2) ∧ S.card ≤ 4) ∧
  ((|a - b| = 1 ∨ |a - b| = 2) → (c - d) % 2 ≠ 0 →
    ∃ S : Finset (ℤ × ℤ), (∀ p ∈ S, f p = 2) ∧ S.card = 4) := by
  sorry

#check equation_solutions

end NUMINAMATH_CALUDE_equation_solutions_l3745_374513


namespace NUMINAMATH_CALUDE_set_equality_l3745_374547

def U : Finset ℕ := {1,2,3,4,5,6,7,8}
def A : Finset ℕ := {3,4,5}
def B : Finset ℕ := {1,3,6}
def C : Finset ℕ := {2,7,8}

theorem set_equality : C = (C ∪ A) ∩ (C ∪ B) := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l3745_374547


namespace NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l3745_374508

theorem greatest_value_quadratic_inequality :
  ∃ (a_max : ℝ), a_max = 9 ∧
  (∀ a : ℝ, a^2 - 14*a + 45 ≤ 0 → a ≤ a_max) ∧
  (a_max^2 - 14*a_max + 45 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l3745_374508


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l3745_374568

/-- A rectangle with a perimeter of 72 meters and a length-to-width ratio of 5:2 has a diagonal of 194/7 meters. -/
theorem rectangle_diagonal (length width : ℝ) : 
  2 * (length + width) = 72 →
  length / width = 5 / 2 →
  Real.sqrt (length^2 + width^2) = 194 / 7 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l3745_374568


namespace NUMINAMATH_CALUDE_triangle_midpoint_sum_l3745_374535

theorem triangle_midpoint_sum (a b c : ℝ) (h : a + b + c = 15) :
  (a + b) / 2 + (b + c) / 2 + (c + a) / 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_triangle_midpoint_sum_l3745_374535


namespace NUMINAMATH_CALUDE_negative_integer_sum_with_square_is_six_l3745_374501

theorem negative_integer_sum_with_square_is_six (N : ℤ) : 
  N < 0 → N^2 + N = 6 → N = -3 := by
  sorry

end NUMINAMATH_CALUDE_negative_integer_sum_with_square_is_six_l3745_374501


namespace NUMINAMATH_CALUDE_least_positive_angle_l3745_374527

theorem least_positive_angle (x a b : ℝ) (h1 : Real.tan x = 2 * a / (3 * b)) 
  (h2 : Real.tan (3 * x) = 3 * b / (2 * a + 3 * b)) :
  x = Real.arctan (2 / 3) ∧ x > 0 ∧ ∀ y, y > 0 → y = Real.arctan (2 / 3) → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_least_positive_angle_l3745_374527


namespace NUMINAMATH_CALUDE_correct_tense_for_ongoing_past_to_present_action_l3745_374563

/-- Represents different verb tenses -/
inductive VerbTense
  | simple_past
  | past_continuous
  | present_perfect_continuous
  | future_continuous

/-- Represents the characteristics of an action -/
structure ActionCharacteristics where
  ongoing : Bool
  started_in_past : Bool
  continues_to_present : Bool

/-- Theorem stating that for an action that is ongoing, started in the past, 
    and continues to the present, the correct tense is present perfect continuous -/
theorem correct_tense_for_ongoing_past_to_present_action 
  (action : ActionCharacteristics) 
  (h1 : action.ongoing = true) 
  (h2 : action.started_in_past = true) 
  (h3 : action.continues_to_present = true) : 
  VerbTense.present_perfect_continuous = 
    (match action with
      | ⟨true, true, true⟩ => VerbTense.present_perfect_continuous
      | _ => VerbTense.simple_past) :=
by sorry


end NUMINAMATH_CALUDE_correct_tense_for_ongoing_past_to_present_action_l3745_374563


namespace NUMINAMATH_CALUDE_a_range_l3745_374522

/-- Proposition p: For any x∈R, x²-2x > a -/
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*x > a

/-- Proposition q: The function f(x)=x²+2ax+2-a has a zero point on R -/
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

/-- The range of values for a is (-2,-1) ∪ [1, +∞) -/
def range_of_a (a : ℝ) : Prop := (a > -2 ∧ a < -1) ∨ a ≥ 1

theorem a_range (a : ℝ) (h1 : p a ∨ q a) (h2 : ¬(p a ∧ q a)) : range_of_a a := by
  sorry

end NUMINAMATH_CALUDE_a_range_l3745_374522


namespace NUMINAMATH_CALUDE_ceiling_of_3_7_l3745_374518

theorem ceiling_of_3_7 : ⌈(3.7 : ℝ)⌉ = 4 := by sorry

end NUMINAMATH_CALUDE_ceiling_of_3_7_l3745_374518


namespace NUMINAMATH_CALUDE_smallest_c_inequality_l3745_374582

theorem smallest_c_inequality (c : ℝ) : 
  (∀ x y : ℝ, x ≥ 0 ∧ y ≥ 0 → Real.sqrt (x^2 + y^2) + c * |x - y| ≥ (x + y) / 2) ↔ c ≥ (1/2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_smallest_c_inequality_l3745_374582


namespace NUMINAMATH_CALUDE_chess_tournament_matches_l3745_374546

/-- The number of matches required in a single-elimination tournament -/
def matches_required (num_players : ℕ) : ℕ :=
  num_players - 1

/-- Theorem: A single-elimination tournament with 32 players requires 31 matches -/
theorem chess_tournament_matches :
  matches_required 32 = 31 :=
by
  sorry


end NUMINAMATH_CALUDE_chess_tournament_matches_l3745_374546


namespace NUMINAMATH_CALUDE_no_trapezoid_solution_l3745_374573

theorem no_trapezoid_solution : 
  ¬ ∃ (b₁ b₂ : ℕ), 
    (b₁ % 10 = 0) ∧ 
    (b₂ % 10 = 0) ∧ 
    ((b₁ + b₂) * 30 / 2 = 1080) :=
by sorry

end NUMINAMATH_CALUDE_no_trapezoid_solution_l3745_374573


namespace NUMINAMATH_CALUDE_decimal_to_binary_27_l3745_374555

theorem decimal_to_binary_27 : 
  (27 : ℕ).digits 2 = [1, 1, 0, 1, 1] := by sorry

end NUMINAMATH_CALUDE_decimal_to_binary_27_l3745_374555


namespace NUMINAMATH_CALUDE_square_perimeter_l3745_374559

theorem square_perimeter (total_area overlap_area circle_area : ℝ) : 
  total_area = 2018 →
  overlap_area = 137 →
  circle_area = 1371 →
  ∃ (square_side : ℝ), 
    square_side > 0 ∧ 
    square_side^2 = total_area - (circle_area - overlap_area) ∧
    4 * square_side = 112 :=
by sorry

end NUMINAMATH_CALUDE_square_perimeter_l3745_374559


namespace NUMINAMATH_CALUDE_symmetrical_circles_sum_l3745_374533

/-- Two circles are symmetrical with respect to the line y = x + 1 -/
def symmetrical_circles (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), (x - a)^2 + (y - b)^2 = 1 ∧ 
                (x - 1)^2 + (y - 3)^2 = 1 ∧
                y = x + 1

/-- If two circles are symmetrical with respect to the line y = x + 1,
    then the sum of their center coordinates is 2 -/
theorem symmetrical_circles_sum (a b : ℝ) :
  symmetrical_circles a b → a + b = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_symmetrical_circles_sum_l3745_374533


namespace NUMINAMATH_CALUDE_initial_girls_count_l3745_374558

theorem initial_girls_count (total : ℕ) (initial_girls : ℕ) : 
  (initial_girls = total / 2) →  -- Initially, 50% of the group are girls
  (initial_girls - 3 : ℚ) / total = 2/5 →  -- After changes, 40% are girls
  initial_girls = 15 := by
sorry

end NUMINAMATH_CALUDE_initial_girls_count_l3745_374558


namespace NUMINAMATH_CALUDE_min_value_when_a_is_one_exactly_two_zeros_l3745_374586

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2^x - a else 4*(x-a)*(x-2*a)

-- Theorem 1: Minimum value when a = 1
theorem min_value_when_a_is_one :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f 1 x_min ≤ f 1 x ∧ f 1 x_min = -1 :=
sorry

-- Theorem 2: Condition for exactly two zeros
theorem exactly_two_zeros (a : ℝ) :
  (∃ (x y : ℝ), x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧ ∀ (z : ℝ), f a z = 0 → z = x ∨ z = y) ↔
  (1/2 ≤ a ∧ a < 1) ∨ (a ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_min_value_when_a_is_one_exactly_two_zeros_l3745_374586


namespace NUMINAMATH_CALUDE_cupcake_net_profit_l3745_374571

/-- Calculates the net profit from selling cupcakes given the specified conditions. -/
theorem cupcake_net_profit : 
  let cost_per_cupcake : ℚ := 0.75
  let selling_price : ℚ := 2.00
  let burnt_cupcakes : ℕ := 24
  let eaten_cupcakes : ℕ := 9
  let total_cupcakes : ℕ := 72
  let sellable_cupcakes : ℕ := total_cupcakes - (burnt_cupcakes + eaten_cupcakes)
  let total_cost : ℚ := cost_per_cupcake * total_cupcakes
  let total_revenue : ℚ := selling_price * sellable_cupcakes
  total_revenue - total_cost = 24.00 := by
sorry


end NUMINAMATH_CALUDE_cupcake_net_profit_l3745_374571


namespace NUMINAMATH_CALUDE_johns_purchase_cost_l3745_374502

/-- Calculates the total cost of John's purchase of soap and shampoo. -/
def total_cost (soap_bars : ℕ) (soap_weight : ℝ) (soap_price : ℝ)
                (shampoo_bottles : ℕ) (shampoo_weight : ℝ) (shampoo_price : ℝ) : ℝ :=
  (soap_bars : ℝ) * soap_weight * soap_price +
  (shampoo_bottles : ℝ) * shampoo_weight * shampoo_price

/-- Proves that John's total spending on soap and shampoo is $41.40. -/
theorem johns_purchase_cost : 
  total_cost 20 1.5 0.5 15 2.2 0.8 = 41.40 := by
  sorry

end NUMINAMATH_CALUDE_johns_purchase_cost_l3745_374502


namespace NUMINAMATH_CALUDE_ellipse_and_line_intersection_l3745_374567

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop :=
  y^2 = 4 * Real.sqrt 2 * x

-- Define the line l
def line (k m x y : ℝ) : Prop :=
  y = k * x + m

-- Define the isosceles triangle condition
def isosceles_triangle (A M N : ℝ × ℝ) : Prop :=
  (A.1 - M.1)^2 + (A.2 - M.2)^2 = (A.1 - N.1)^2 + (A.2 - N.2)^2

theorem ellipse_and_line_intersection
  (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : ∃ (x y : ℝ), ellipse a b x y ∧ parabola x y)
  (h4 : ∃ (x1 x2 y1 y2 : ℝ), 
    x1^2 + x2^2 + y1^2 + y2^2 = 2 * a * b * Real.sqrt 3)
  (k m : ℝ) (h5 : k ≠ 0)
  (h6 : ∃ (M N : ℝ × ℝ), 
    ellipse a b M.1 M.2 ∧ 
    ellipse a b N.1 N.2 ∧ 
    line k m M.1 M.2 ∧ 
    line k m N.1 N.2 ∧ 
    M ≠ N)
  (h7 : ∃ (A : ℝ × ℝ), ellipse a b A.1 A.2 ∧ A.2 < 0)
  (h8 : ∀ (A M N : ℝ × ℝ), 
    ellipse a b A.1 A.2 ∧ A.2 < 0 ∧
    ellipse a b M.1 M.2 ∧ 
    ellipse a b N.1 N.2 ∧ 
    line k m M.1 M.2 ∧ 
    line k m N.1 N.2 →
    isosceles_triangle A M N) :
  a = Real.sqrt 3 ∧ b = 1 ∧ 1/2 < m ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_and_line_intersection_l3745_374567


namespace NUMINAMATH_CALUDE_meters_examined_l3745_374511

/-- The percentage of defective meters -/
def defective_percentage : ℝ := 0.08

/-- The number of defective meters found -/
def defective_count : ℕ := 2

/-- The total number of meters examined -/
def total_meters : ℕ := 2500

theorem meters_examined :
  (defective_percentage / 100 * total_meters : ℝ) = defective_count := by
  sorry

end NUMINAMATH_CALUDE_meters_examined_l3745_374511


namespace NUMINAMATH_CALUDE_joe_fruit_probability_l3745_374599

/-- The number of meals Joe has in a day. -/
def num_meals : ℕ := 3

/-- The number of fruit options Joe has for each meal. -/
def num_fruits : ℕ := 4

/-- The probability of choosing a specific fruit for a meal. -/
def prob_single_fruit : ℚ := 1 / num_fruits

/-- The probability of eating the same fruit for all meals. -/
def prob_same_fruit : ℚ := prob_single_fruit ^ num_meals

/-- The probability of eating at least two different kinds of fruit in a day. -/
def prob_different_fruits : ℚ := 1 - (num_fruits * prob_same_fruit)

theorem joe_fruit_probability : prob_different_fruits = 15 / 16 := by
  sorry

end NUMINAMATH_CALUDE_joe_fruit_probability_l3745_374599


namespace NUMINAMATH_CALUDE_remainder_of_division_l3745_374516

theorem remainder_of_division (n : ℕ) : 
  (2^224 + 104) % (2^112 + 2^56 + 1) = 103 := by
sorry

end NUMINAMATH_CALUDE_remainder_of_division_l3745_374516


namespace NUMINAMATH_CALUDE_family_age_sum_seven_years_ago_l3745_374539

/-- A family of 5 members -/
structure Family :=
  (age1 age2 age3 age4 age5 : ℕ)

/-- The sum of ages of the family members -/
def ageSum (f : Family) : ℕ := f.age1 + f.age2 + f.age3 + f.age4 + f.age5

/-- Theorem: Given a family of 5 whose ages sum to 80, with the two youngest being 6 and 8 years old,
    the sum of their ages 7 years ago was 45 -/
theorem family_age_sum_seven_years_ago (f : Family)
  (h1 : ageSum f = 80)
  (h2 : f.age4 = 8)
  (h3 : f.age5 = 6)
  (h4 : f.age1 ≥ 7 ∧ f.age2 ≥ 7 ∧ f.age3 ≥ 7) :
  (f.age1 - 7) + (f.age2 - 7) + (f.age3 - 7) + 1 = 45 :=
by sorry

end NUMINAMATH_CALUDE_family_age_sum_seven_years_ago_l3745_374539


namespace NUMINAMATH_CALUDE_work_completion_time_l3745_374565

/-- The time taken for three workers to complete a work together, given their individual completion times -/
theorem work_completion_time (tx ty tz : ℝ) (htx : tx = 20) (hty : ty = 40) (htz : tz = 30) :
  (1 / tx + 1 / ty + 1 / tz)⁻¹ = 120 / 13 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3745_374565


namespace NUMINAMATH_CALUDE_max_value_fraction_l3745_374580

theorem max_value_fraction (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y + y * z) / (x^2 + y^2 + z^2) ≤ Real.sqrt 2 / 2 ∧
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ (a * b + b * c) / (a^2 + b^2 + c^2) = Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_fraction_l3745_374580


namespace NUMINAMATH_CALUDE_circle_line_relationship_l3745_374572

-- Define the circle C: x^2 + y^2 = 4
def C (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line l: √3x + y - 8 = 0
def l (x y : ℝ) : Prop := Real.sqrt 3 * x + y - 8 = 0

-- Theorem statement
theorem circle_line_relationship :
  -- C and l are separate
  (∀ x y : ℝ, C x y → ¬ l x y) ∧
  -- The shortest distance from any point on C to l is 2
  (∀ x y : ℝ, C x y → ∃ d : ℝ, d = 2 ∧ 
    ∀ x' y' : ℝ, l x' y' → Real.sqrt ((x - x')^2 + (y - y')^2) ≥ d) :=
sorry

end NUMINAMATH_CALUDE_circle_line_relationship_l3745_374572


namespace NUMINAMATH_CALUDE_unique_coin_configuration_l3745_374585

/-- Represents the different types of coins -/
inductive CoinType
  | Penny
  | Nickel
  | Dime

/-- The value of each coin type in cents -/
def coinValue : CoinType → Nat
  | CoinType.Penny => 1
  | CoinType.Nickel => 5
  | CoinType.Dime => 10

/-- A configuration of coins -/
structure CoinConfiguration where
  pennies : Nat
  nickels : Nat
  dimes : Nat

/-- The total number of coins in a configuration -/
def CoinConfiguration.totalCoins (c : CoinConfiguration) : Nat :=
  c.pennies + c.nickels + c.dimes

/-- The total value of coins in a configuration in cents -/
def CoinConfiguration.totalValue (c : CoinConfiguration) : Nat :=
  c.pennies * coinValue CoinType.Penny +
  c.nickels * coinValue CoinType.Nickel +
  c.dimes * coinValue CoinType.Dime

/-- Theorem: There is a unique coin configuration with 8 coins, 53 cents total value,
    and at least one of each coin type, which must have exactly 3 nickels -/
theorem unique_coin_configuration :
  ∃! c : CoinConfiguration,
    c.totalCoins = 8 ∧
    c.totalValue = 53 ∧
    c.pennies ≥ 1 ∧
    c.nickels ≥ 1 ∧
    c.dimes ≥ 1 ∧
    c.nickels = 3 := by
  sorry


end NUMINAMATH_CALUDE_unique_coin_configuration_l3745_374585


namespace NUMINAMATH_CALUDE_square_construction_implies_parallel_l3745_374500

-- Define the triangle ABC
variable (A B C : Plane)

-- Define the squares constructed on the sides of triangle ABC
variable (A₂ B₁ B₂ C₁ : Plane)

-- Define the additional squares
variable (A₃ A₄ B₃ B₄ : Plane)

-- Define the property of being a square
def is_square (P Q R S : Plane) : Prop := sorry

-- Define the property of being external to a triangle
def is_external_to_triangle (S₁ S₂ S₃ S₄ P Q R : Plane) : Prop := sorry

-- Define the property of being parallel
def is_parallel (P₁ P₂ Q₁ Q₂ : Plane) : Prop := sorry

theorem square_construction_implies_parallel :
  is_square A B B₁ A₂ →
  is_square B C C₁ B₂ →
  is_square C A A₁ C₂ →
  is_external_to_triangle A B B₁ A₂ A B C →
  is_external_to_triangle B C C₁ B₂ B C A →
  is_external_to_triangle C A A₁ C₂ C A B →
  is_square A₁ A₂ A₃ A₄ →
  is_square B₁ B₂ B₃ B₄ →
  is_external_to_triangle A₁ A₂ A₃ A₄ A A₁ A₂ →
  is_external_to_triangle B₁ B₂ B₃ B₄ B B₁ B₂ →
  is_parallel A₃ B₄ A B := by sorry

end NUMINAMATH_CALUDE_square_construction_implies_parallel_l3745_374500


namespace NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l3745_374550

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem fifth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_a1 : a 1 = 1)
  (h_a4 : a 4 = 8) :
  a 5 = 16 :=
sorry

end NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l3745_374550


namespace NUMINAMATH_CALUDE_ada_paul_test_scores_l3745_374520

/-- Ada and Paul's test scores problem -/
theorem ada_paul_test_scores 
  (a1 a2 a3 p1 p2 p3 : ℤ) 
  (h1 : a1 = p1 + 10)
  (h2 : a2 = p2 + 4)
  (h3 : (p1 + p2 + p3) / 3 = (a1 + a2 + a3) / 3 + 4) :
  p3 - a3 = 26 := by
sorry

end NUMINAMATH_CALUDE_ada_paul_test_scores_l3745_374520


namespace NUMINAMATH_CALUDE_solve_system_of_equations_l3745_374576

theorem solve_system_of_equations (a x y : ℚ) 
  (eq1 : a * x + y = 8)
  (eq2 : 3 * x - 4 * y = 5)
  (eq3 : 7 * x - 3 * y = 23) :
  a = 25 / 16 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_of_equations_l3745_374576


namespace NUMINAMATH_CALUDE_white_surface_area_fraction_l3745_374510

/-- Represents a cube with side length and number of smaller cubes -/
structure Cube where
  side_length : ℕ
  num_smaller_cubes : ℕ

/-- Represents the composition of a larger cube -/
structure CubeComposition where
  large_cube : Cube
  small_cube : Cube
  num_red : ℕ
  num_white : ℕ

/-- Calculate the surface area of a cube -/
def surface_area (c : Cube) : ℕ := 6 * c.side_length^2

/-- Calculate the minimum number of visible faces for white cubes -/
def min_visible_white_faces (cc : CubeComposition) : ℕ :=
  cc.num_white - 1

/-- The theorem stating the fraction of white surface area -/
theorem white_surface_area_fraction (cc : CubeComposition) 
  (h1 : cc.large_cube.side_length = 4)
  (h2 : cc.small_cube.side_length = 1)
  (h3 : cc.large_cube.num_smaller_cubes = 64)
  (h4 : cc.num_red = 56)
  (h5 : cc.num_white = 8) :
  (min_visible_white_faces cc : ℚ) / (surface_area cc.large_cube : ℚ) = 7 / 96 := by
  sorry

end NUMINAMATH_CALUDE_white_surface_area_fraction_l3745_374510


namespace NUMINAMATH_CALUDE_cone_sphere_ratio_l3745_374581

theorem cone_sphere_ratio (r : ℝ) (h : ℝ) (R : ℝ) : 
  R = 2 * r →
  (1/3) * (4/3) * Real.pi * r^3 = (1/3) * Real.pi * R^2 * h →
  h / R = 1/6 :=
by sorry

end NUMINAMATH_CALUDE_cone_sphere_ratio_l3745_374581


namespace NUMINAMATH_CALUDE_park_visitors_l3745_374542

theorem park_visitors (bike_riders : ℕ) (hikers : ℕ) : 
  bike_riders = 249 →
  hikers = bike_riders + 178 →
  bike_riders + hikers = 676 :=
by sorry

end NUMINAMATH_CALUDE_park_visitors_l3745_374542


namespace NUMINAMATH_CALUDE_intersection_implies_a_values_l3745_374531

theorem intersection_implies_a_values (a : ℝ) : 
  let M : Set ℝ := {5, a^2 - 3*a + 5}
  let N : Set ℝ := {1, 3}
  (M ∩ N).Nonempty → a = 1 ∨ a = 2 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_a_values_l3745_374531


namespace NUMINAMATH_CALUDE_susan_single_digit_in_ten_steps_l3745_374524

/-- Represents a multi-digit number as a list of digits -/
def MultiDigitNumber := List Nat

/-- Represents a position where a plus sign can be inserted -/
def PlusPosition := Nat

/-- Represents a set of positions where plus signs are inserted -/
def PlusPositions := List PlusPosition

/-- Performs one step of Susan's operation -/
def performStep (n : MultiDigitNumber) (positions : PlusPositions) : MultiDigitNumber :=
  sorry

/-- Checks if a number is a single digit -/
def isSingleDigit (n : MultiDigitNumber) : Prop :=
  n.length = 1

/-- Main theorem: Susan can always obtain a single-digit number in at most ten steps -/
theorem susan_single_digit_in_ten_steps (n : MultiDigitNumber) :
  ∃ (steps : List PlusPositions),
    steps.length ≤ 10 ∧
    isSingleDigit (steps.foldl performStep n) :=
  sorry

end NUMINAMATH_CALUDE_susan_single_digit_in_ten_steps_l3745_374524


namespace NUMINAMATH_CALUDE_angle_through_point_l3745_374541

theorem angle_through_point (α : Real) : 
  0 ≤ α → α < 2 * Real.pi → 
  let P : Real × Real := (Real.sin (2 * Real.pi / 3), Real.cos (2 * Real.pi / 3))
  (Real.cos α = P.1 ∧ Real.sin α = P.2) →
  α = 11 * Real.pi / 6 := by
sorry

end NUMINAMATH_CALUDE_angle_through_point_l3745_374541


namespace NUMINAMATH_CALUDE_rectangle_area_l3745_374525

/-- The area of a rectangle bounded by lines y = 2a, y = 3b, x = 4c, and x = 5d,
    where a, b, c, and d are positive numbers. -/
theorem rectangle_area (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (2 * a - 3 * b) * (5 * d - 4 * c) = 10 * a * d - 8 * a * c - 15 * b * d + 12 * b * c := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3745_374525


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l3745_374578

theorem more_girls_than_boys 
  (total_pupils : ℕ) 
  (girls : ℕ) 
  (h1 : total_pupils = 1455)
  (h2 : girls = 868)
  (h3 : girls > total_pupils - girls) : 
  girls - (total_pupils - girls) = 281 :=
by
  sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l3745_374578


namespace NUMINAMATH_CALUDE_systematic_sampling_result_l3745_374549

def population_size : ℕ := 50
def sample_size : ℕ := 5
def starting_point : ℕ := 5
def step_size : ℕ := population_size / sample_size

def systematic_sample (start : ℕ) (step : ℕ) (n : ℕ) : List ℕ :=
  List.range n |> List.map (fun i => start + i * step)

theorem systematic_sampling_result :
  systematic_sample starting_point step_size sample_size = [5, 15, 25, 35, 45] := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_result_l3745_374549


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l3745_374597

theorem min_value_x_plus_y (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (h : (x + 1) * (y + 1) = 9) : 
  x + y ≥ 4 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ (x + 1) * (y + 1) = 9 ∧ x + y = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l3745_374597


namespace NUMINAMATH_CALUDE_bridget_erasers_l3745_374536

theorem bridget_erasers (initial final given : ℕ) : 
  initial = 8 → final = 11 → given = final - initial :=
by
  sorry

end NUMINAMATH_CALUDE_bridget_erasers_l3745_374536


namespace NUMINAMATH_CALUDE_min_value_of_function_l3745_374574

theorem min_value_of_function (x : ℝ) (h : x > 4) : 
  ∃ (y_min : ℝ), y_min = 6 ∧ ∀ y, y = x + 1 / (x - 4) → y ≥ y_min :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l3745_374574


namespace NUMINAMATH_CALUDE_f_is_quadratic_l3745_374503

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation 2x^2 - x - 3 = 0 -/
def f (x : ℝ) : ℝ := 2 * x^2 - x - 3

/-- Theorem: f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l3745_374503


namespace NUMINAMATH_CALUDE_fifth_degree_polynomial_existence_l3745_374551

theorem fifth_degree_polynomial_existence : ∃ (P : ℝ → ℝ),
  (∀ x : ℝ, P x = 0 → x < 0) ∧
  (∀ x : ℝ, (deriv P) x = 0 → x > 0) ∧
  (∃ x : ℝ, P x = 0 ∧ ∀ y : ℝ, y ≠ x → P y ≠ 0) ∧
  (∃ x : ℝ, (deriv P) x = 0 ∧ ∀ y : ℝ, y ≠ x → (deriv P) y ≠ 0) ∧
  (∃ a b c d e f : ℝ, ∀ x : ℝ, P x = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f) :=
by sorry

end NUMINAMATH_CALUDE_fifth_degree_polynomial_existence_l3745_374551


namespace NUMINAMATH_CALUDE_cosine_inequality_l3745_374596

theorem cosine_inequality (y : Real) :
  y ∈ Set.Icc 0 Real.pi →
  (∀ x : Real, Real.cos (x + y) ≤ Real.cos x * Real.cos y) ↔
  (y = 0 ∨ y = Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_cosine_inequality_l3745_374596


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3745_374588

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

/-- The sum of specific terms in the sequence equals 120 -/
def sum_condition (a : ℕ → ℝ) : Prop :=
  a 4 + a 6 + a 8 + a 10 + a 12 = 120

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) (h2 : sum_condition a) : 
  2 * a 10 - a 12 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3745_374588


namespace NUMINAMATH_CALUDE_existence_of_unique_representation_sets_l3745_374584

-- Define the property of being an infinite set of non-negative integers
def IsInfiniteNonNegSet (S : Set ℕ) : Prop :=
  Set.Infinite S ∧ ∀ x ∈ S, x ≥ 0

-- Define the property that every non-negative integer has a unique representation
def HasUniqueRepresentation (A B : Set ℕ) : Prop :=
  ∀ n : ℕ, ∃! (a b : ℕ), a ∈ A ∧ b ∈ B ∧ n = a + b

-- The main theorem
theorem existence_of_unique_representation_sets :
  ∃ A B : Set ℕ, IsInfiniteNonNegSet A ∧ IsInfiniteNonNegSet B ∧ HasUniqueRepresentation A B :=
sorry

end NUMINAMATH_CALUDE_existence_of_unique_representation_sets_l3745_374584


namespace NUMINAMATH_CALUDE_unique_solution_abc_l3745_374521

theorem unique_solution_abc :
  ∀ A B C : ℝ,
  A = 2 * B - 3 * C →
  B = 2 * C - 5 →
  A + B + C = 100 →
  A = 18.75 ∧ B = 52.5 ∧ C = 28.75 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_abc_l3745_374521


namespace NUMINAMATH_CALUDE_optimal_sequence_l3745_374544

theorem optimal_sequence (x₁ x₂ x₃ x₄ : ℝ) 
  (h_order : 0 ≤ x₄ ∧ x₄ ≤ x₃ ∧ x₃ ≤ x₂ ∧ x₂ ≤ x₁ ∧ x₁ ≤ 1) 
  (h_eq : (1 - x₁)^2 + (x₁ - x₂)^2 + (x₂ - x₃)^2 + (x₃ - x₄)^2 + x₄^2 = 1/5) :
  x₁ = 4/5 ∧ x₂ = 3/5 ∧ x₃ = 2/5 ∧ x₄ = 1/5 := by
sorry

end NUMINAMATH_CALUDE_optimal_sequence_l3745_374544


namespace NUMINAMATH_CALUDE_min_value_inequality_l3745_374529

theorem min_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  Real.sqrt ((x^2 + y^2) * (5 * x^2 + y^2)) / (x * y) ≥ Real.sqrt 5 + 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l3745_374529


namespace NUMINAMATH_CALUDE_excess_meat_sales_l3745_374540

def meat_market_sales (thursday_sales : ℕ) (saturday_sales : ℕ) (original_plan : ℕ) : Prop :=
  let friday_sales := 2 * thursday_sales
  let sunday_sales := saturday_sales / 2
  let total_sales := thursday_sales + friday_sales + saturday_sales + sunday_sales
  total_sales - original_plan = 325

theorem excess_meat_sales : meat_market_sales 210 130 500 := by
  sorry

end NUMINAMATH_CALUDE_excess_meat_sales_l3745_374540


namespace NUMINAMATH_CALUDE_gcf_of_60_and_90_l3745_374537

theorem gcf_of_60_and_90 : Nat.gcd 60 90 = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_60_and_90_l3745_374537


namespace NUMINAMATH_CALUDE_fraction_conversion_equivalence_l3745_374523

theorem fraction_conversion_equivalence (x : ℚ) :
  (x + 1) / (2/5) - ((1/5) * x - 1) / (7/10) = 1 ↔
  (10 * x + 10) / 4 - (2 * x - 10) / 7 = 1 :=
by sorry

end NUMINAMATH_CALUDE_fraction_conversion_equivalence_l3745_374523


namespace NUMINAMATH_CALUDE_herman_breakfast_cost_l3745_374557

/-- Calculates the total amount spent on breakfast during a project --/
def total_breakfast_cost (team_size : ℕ) (days_per_week : ℕ) (meal_cost : ℚ) (project_duration : ℕ) : ℚ :=
  (team_size : ℚ) * (days_per_week : ℚ) * meal_cost * (project_duration : ℚ)

/-- Proves that Herman's total breakfast cost for the project is $1,280.00 --/
theorem herman_breakfast_cost :
  let team_size : ℕ := 4  -- Herman and 3 team members
  let days_per_week : ℕ := 5
  let meal_cost : ℚ := 4
  let project_duration : ℕ := 16
  total_breakfast_cost team_size days_per_week meal_cost project_duration = 1280 := by
  sorry

end NUMINAMATH_CALUDE_herman_breakfast_cost_l3745_374557


namespace NUMINAMATH_CALUDE_product_theorem_l3745_374561

theorem product_theorem (x : ℝ) : 3.6 * x = 10.08 → 15 * x = 42 := by
  sorry

end NUMINAMATH_CALUDE_product_theorem_l3745_374561


namespace NUMINAMATH_CALUDE_unique_parallel_line_l3745_374552

/-- Two planes are parallel -/
def parallel_planes (α β : Set (Fin 3 → ℝ)) : Prop := sorry

/-- A line is contained in a plane -/
def line_in_plane (l : Set (Fin 3 → ℝ)) (p : Set (Fin 3 → ℝ)) : Prop := sorry

/-- A point is in a plane -/
def point_in_plane (x : Fin 3 → ℝ) (p : Set (Fin 3 → ℝ)) : Prop := sorry

/-- Two lines are parallel -/
def parallel_lines (l₁ l₂ : Set (Fin 3 → ℝ)) : Prop := sorry

/-- The set of all lines in a plane passing through a point -/
def lines_through_point (p : Set (Fin 3 → ℝ)) (x : Fin 3 → ℝ) : Set (Set (Fin 3 → ℝ)) := sorry

theorem unique_parallel_line 
  (α β : Set (Fin 3 → ℝ)) 
  (a : Set (Fin 3 → ℝ)) 
  (B : Fin 3 → ℝ) 
  (h₁ : parallel_planes α β) 
  (h₂ : line_in_plane a α) 
  (h₃ : point_in_plane B β) : 
  ∃! l, l ∈ lines_through_point β B ∧ parallel_lines l a := by
  sorry

end NUMINAMATH_CALUDE_unique_parallel_line_l3745_374552


namespace NUMINAMATH_CALUDE_pizza_stand_total_slices_l3745_374528

/-- Given the conditions of the pizza stand problem, prove that the total number of slices sold is 5000. -/
theorem pizza_stand_total_slices : 
  let small_price : ℕ := 150
  let large_price : ℕ := 250
  let total_revenue : ℕ := 1050000
  let small_slices_sold : ℕ := 2000
  let large_slices_sold : ℕ := (total_revenue - small_price * small_slices_sold) / large_price
  small_slices_sold + large_slices_sold = 5000 := by
sorry


end NUMINAMATH_CALUDE_pizza_stand_total_slices_l3745_374528


namespace NUMINAMATH_CALUDE_cubic_expansion_l3745_374579

theorem cubic_expansion (x : ℝ) : 
  3*x^3 - 10*x^2 + 13 = 3*(x-2)^3 + 8*(x-2)^2 - 4*(x-2) - 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expansion_l3745_374579


namespace NUMINAMATH_CALUDE_chinese_character_equation_l3745_374543

theorem chinese_character_equation :
  ∃! (a b c d : Nat),
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    d + d + d + d = 48 ∧
    1000 * a + 100 * b + 10 * c + d = 1468 :=
by
  sorry

end NUMINAMATH_CALUDE_chinese_character_equation_l3745_374543


namespace NUMINAMATH_CALUDE_original_car_price_l3745_374556

/-- Represents the original cost price of the car -/
def original_price : ℝ := sorry

/-- Represents the price after the first sale (11% loss) -/
def first_sale_price : ℝ := original_price * (1 - 0.11)

/-- Represents the price after the second sale (15% gain) -/
def second_sale_price : ℝ := first_sale_price * (1 + 0.15)

/-- Represents the final selling price -/
def final_sale_price : ℝ := 75000

/-- Theorem stating the original price of the car -/
theorem original_car_price : 
  ∃ (price : ℝ), original_price = price ∧ 
  (price ≥ 58744) ∧ (price ≤ 58745) ∧
  (second_sale_price * (1 + 0.25) = final_sale_price) :=
sorry

end NUMINAMATH_CALUDE_original_car_price_l3745_374556


namespace NUMINAMATH_CALUDE_inequality_proof_l3745_374512

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 + 3 / (a * b + b * c + c * a) ≥ 6 / (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3745_374512


namespace NUMINAMATH_CALUDE_expression_factorization_l3745_374526

theorem expression_factorization (x : ℝ) :
  (16 * x^6 + 36 * x^4 - 9) - (4 * x^6 - 6 * x^4 - 9) = 6 * x^4 * (2 * x^2 + 7) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l3745_374526


namespace NUMINAMATH_CALUDE_infinite_pairs_exist_infinitely_many_pairs_l3745_374506

/-- Recursive definition of the sequence a_n -/
def a : ℕ → ℕ
  | 0 => 4
  | 1 => 11
  | (n + 2) => 3 * a (n + 1) - a n

/-- Theorem stating the existence of infinitely many pairs satisfying the given properties -/
theorem infinite_pairs_exist : ∀ n : ℕ, n ≥ 1 → 
  (a n < a (n + 1)) ∧ 
  (Nat.gcd (a n) (a (n + 1)) = 1) ∧
  (a n ∣ a (n + 1)^2 - 5) ∧
  (a (n + 1) ∣ a n^2 - 5) := by
  sorry

/-- Corollary: There exist infinitely many pairs of positive integers satisfying the properties -/
theorem infinitely_many_pairs : 
  ∃ f : ℕ → ℕ × ℕ, ∀ n : ℕ, 
    let (a, b) := f n
    (a > b) ∧
    (Nat.gcd a b = 1) ∧
    (a ∣ b^2 - 5) ∧
    (b ∣ a^2 - 5) := by
  sorry

end NUMINAMATH_CALUDE_infinite_pairs_exist_infinitely_many_pairs_l3745_374506


namespace NUMINAMATH_CALUDE_machine_production_time_l3745_374570

theorem machine_production_time (x : ℝ) (T : ℝ) : T = 10 :=
  let machine_B_rate := 2 * x / 5
  let combined_rate := x / 2
  have h1 : x / T + machine_B_rate = combined_rate := by sorry
  sorry

end NUMINAMATH_CALUDE_machine_production_time_l3745_374570


namespace NUMINAMATH_CALUDE_root_product_sum_l3745_374534

theorem root_product_sum (x₁ x₂ x₃ : ℝ) : 
  x₁ < x₂ ∧ x₂ < x₃ ∧
  (∀ x, Real.sqrt 2021 * x^3 - 4043 * x^2 + x + 2 = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) →
  x₂ * (x₁ + x₃) = 2 * Real.sqrt 2021 :=
by sorry

end NUMINAMATH_CALUDE_root_product_sum_l3745_374534


namespace NUMINAMATH_CALUDE_omega_range_l3745_374562

theorem omega_range (ω : ℝ) (h_pos : ω > 0) : 
  (∀ x ∈ Set.Ioo (2 * Real.pi / 3) (4 * Real.pi / 3), 
    Monotone (fun x => Real.cos (ω * x + Real.pi / 3))) → 
  1 ≤ ω ∧ ω ≤ 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_omega_range_l3745_374562


namespace NUMINAMATH_CALUDE_system_solution_unique_l3745_374583

theorem system_solution_unique :
  ∃! (x y : ℝ), x - y = 1 ∧ 2 * x + 3 * y = 7 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l3745_374583


namespace NUMINAMATH_CALUDE_carlos_cookie_count_l3745_374554

/-- Represents the shape of a cookie -/
inductive CookieShape
  | Rectangle
  | Square

/-- Represents a cookie with its shape and area -/
structure Cookie where
  shape : CookieShape
  area : ℝ

/-- Represents a batch of cookies -/
structure CookieBatch where
  shape : CookieShape
  totalArea : ℝ
  count : ℕ

/-- The theorem to be proved -/
theorem carlos_cookie_count 
  (anne_batch : CookieBatch)
  (carlos_batch : CookieBatch)
  (h1 : anne_batch.shape = CookieShape.Rectangle)
  (h2 : carlos_batch.shape = CookieShape.Square)
  (h3 : anne_batch.totalArea = 180)
  (h4 : anne_batch.count = 15)
  (h5 : anne_batch.totalArea = carlos_batch.totalArea) :
  carlos_batch.count = 20 := by
  sorry


end NUMINAMATH_CALUDE_carlos_cookie_count_l3745_374554


namespace NUMINAMATH_CALUDE_hyperbola_parameter_sum_l3745_374560

/-- Theorem about the sum of parameters for a specific hyperbola -/
theorem hyperbola_parameter_sum :
  let center : ℝ × ℝ := (-3, 1)
  let focus : ℝ × ℝ := (-3 + Real.sqrt 50, 1)
  let vertex : ℝ × ℝ := (-7, 1)
  let h : ℝ := center.1
  let k : ℝ := center.2
  let a : ℝ := |vertex.1 - center.1|
  let c : ℝ := |focus.1 - center.1|
  let b : ℝ := Real.sqrt (c^2 - a^2)
  h + k + a + b = 2 + Real.sqrt 34 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_parameter_sum_l3745_374560


namespace NUMINAMATH_CALUDE_max_product_sum_2000_l3745_374507

theorem max_product_sum_2000 :
  (∃ (a b : ℤ), a + b = 2000 ∧ ∀ (x y : ℤ), x + y = 2000 → x * y ≤ a * b) ∧
  (∀ (a b : ℤ), a + b = 2000 → a * b ≤ 1000000) ∧
  (∃ (a b : ℤ), a + b = 2000 ∧ a * b = 1000000) :=
by sorry

end NUMINAMATH_CALUDE_max_product_sum_2000_l3745_374507


namespace NUMINAMATH_CALUDE_max_sum_on_circle_max_sum_achievable_l3745_374509

theorem max_sum_on_circle (x y : ℤ) : x^2 + y^2 = 100 → x + y ≤ 14 := by
  sorry

theorem max_sum_achievable : ∃ x y : ℤ, x^2 + y^2 = 100 ∧ x + y = 14 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_on_circle_max_sum_achievable_l3745_374509


namespace NUMINAMATH_CALUDE_value_of_5_minus_c_l3745_374591

theorem value_of_5_minus_c (c d : ℤ) 
  (eq1 : 5 + c = 6 - d) 
  (eq2 : 7 + d = 10 + c) : 
  5 - c = 6 := by
  sorry

end NUMINAMATH_CALUDE_value_of_5_minus_c_l3745_374591


namespace NUMINAMATH_CALUDE_vertical_line_equation_l3745_374517

/-- A line passing through the point (-2,1) with an undefined slope (vertical line) has the equation x + 2 = 0 -/
theorem vertical_line_equation : 
  ∀ (l : Set (ℝ × ℝ)), 
  (∀ (x y : ℝ), (x, y) ∈ l ↔ x = -2) → 
  (-2, 1) ∈ l → 
  (∀ (x y : ℝ), (x, y) ∈ l ↔ x + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_vertical_line_equation_l3745_374517


namespace NUMINAMATH_CALUDE_circle_equation_from_ellipse_and_hyperbola_l3745_374595

/-- Given an ellipse and a hyperbola, prove that a circle centered at the right focus of the ellipse
    and tangent to the asymptotes of the hyperbola has the equation x^2 + y^2 - 10x + 9 = 0 -/
theorem circle_equation_from_ellipse_and_hyperbola 
  (ellipse : ∀ x y : ℝ, x^2 / 169 + y^2 / 144 = 1 → Set ℝ × ℝ)
  (hyperbola : ∀ x y : ℝ, x^2 / 9 - y^2 / 16 = 1 → Set ℝ × ℝ)
  (circle_center : ℝ × ℝ)
  (is_right_focus : circle_center = (5, 0))
  (is_tangent_to_asymptotes : ∀ x y : ℝ, (y = 4/3 * x ∨ y = -4/3 * x) → 
    ((x - circle_center.1)^2 + (y - circle_center.2)^2 = 16)) :
  ∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | p.1^2 + p.2^2 - 10 * p.1 + 9 = 0} ↔ 
    (x - circle_center.1)^2 + (y - circle_center.2)^2 = 16 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_from_ellipse_and_hyperbola_l3745_374595


namespace NUMINAMATH_CALUDE_fraction_to_decimal_conversion_l3745_374594

theorem fraction_to_decimal_conversion :
  ∃ (d : ℚ), (d.num / d.den = 7 / 12) ∧ (abs (d - 0.58333) < 0.000005) := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_conversion_l3745_374594


namespace NUMINAMATH_CALUDE_longest_chord_line_eq_l3745_374589

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- Represents a line in 2D space -/
structure Line where
  equation : ℝ → ℝ → Prop

/-- Given a circle and a point inside it, returns the line containing the longest chord passing through the point -/
def longestChordLine (c : Circle) (m : Point) : Line :=
  sorry

/-- The theorem stating that the longest chord line passing through M(3, -1) in the given circle has the equation x + 2y - 2 = 0 -/
theorem longest_chord_line_eq (c : Circle) (m : Point) :
  c.equation = (fun x y => x^2 + y^2 - 4*x + y - 2 = 0) →
  m = ⟨3, -1⟩ →
  (longestChordLine c m).equation = (fun x y => x + 2*y - 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_longest_chord_line_eq_l3745_374589


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3745_374569

-- Define the sets M and N
def M : Set ℝ := {x | -2 < x ∧ x < 2}
def N : Set ℝ := {x | x^2 - 2*x - 3 < 0}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x | -1 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3745_374569


namespace NUMINAMATH_CALUDE_point_not_in_region_l3745_374575

def plane_region (x y : ℝ) : Prop := 3 * x + 2 * y < 6

theorem point_not_in_region : ¬ plane_region 2 0 := by
  sorry

end NUMINAMATH_CALUDE_point_not_in_region_l3745_374575


namespace NUMINAMATH_CALUDE_divisibility_by_six_l3745_374566

theorem divisibility_by_six (a b c : ℤ) (h : 18 ∣ (a^3 + b^3 + c^3)) : 6 ∣ (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_six_l3745_374566


namespace NUMINAMATH_CALUDE_worksheet_grading_problem_l3745_374530

theorem worksheet_grading_problem 
  (problems_per_worksheet : ℕ)
  (worksheets_graded : ℕ)
  (remaining_problems : ℕ)
  (h1 : problems_per_worksheet = 4)
  (h2 : worksheets_graded = 5)
  (h3 : remaining_problems = 16) :
  worksheets_graded + (remaining_problems / problems_per_worksheet) = 9 := by
  sorry

end NUMINAMATH_CALUDE_worksheet_grading_problem_l3745_374530


namespace NUMINAMATH_CALUDE_profit_percent_l3745_374519

theorem profit_percent (P : ℝ) (h : P > 0) : 
  (2 / 3 * P) * (1 + (-0.2)) = 0.8 * ((5 / 6) * P) → 
  (P - (5 / 6 * P)) / (5 / 6 * P) = 0.2 := by
sorry

end NUMINAMATH_CALUDE_profit_percent_l3745_374519


namespace NUMINAMATH_CALUDE_A_intersection_B_eq_A_l3745_374592

def A (k : ℝ) : Set ℝ := {x | k + 1 ≤ x ∧ x ≤ 2 * k}
def B : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem A_intersection_B_eq_A (k : ℝ) : A k ∩ B = A k ↔ k ∈ Set.Iic (3/2) :=
sorry

end NUMINAMATH_CALUDE_A_intersection_B_eq_A_l3745_374592


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3745_374532

def inequality_system (x : ℝ) : Prop :=
  x > -6 - 2*x ∧ x ≤ (3 + x) / 4

theorem inequality_system_solution :
  ∀ x : ℝ, inequality_system x ↔ -2 < x ∧ x ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3745_374532
