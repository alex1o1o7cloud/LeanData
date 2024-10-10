import Mathlib

namespace function_shift_l1936_193691

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem function_shift (x : ℝ) : f (x + 1) = x^2 - 2*x - 3 → f x = x^2 - 4*x := by
  sorry

end function_shift_l1936_193691


namespace power_of_product_l1936_193692

theorem power_of_product (a b : ℝ) : (a * b^3)^3 = a^3 * b^9 := by
  sorry

end power_of_product_l1936_193692


namespace odd_function_product_nonpositive_l1936_193616

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- State the theorem
theorem odd_function_product_nonpositive (f : ℝ → ℝ) (h : OddFunction f) :
  ∀ x : ℝ, f x * f (-x) ≤ 0 :=
by sorry

end odd_function_product_nonpositive_l1936_193616


namespace train_crossing_time_l1936_193654

/-- Proves that given two trains of equal length, where one train takes 15 seconds to cross a
    telegraph post, and they cross each other traveling in opposite directions in 7.5 seconds,
    the other train will take 5 seconds to cross the telegraph post. -/
theorem train_crossing_time
  (train_length : ℝ)
  (second_train_time : ℝ)
  (crossing_time : ℝ)
  (h1 : train_length = 120)
  (h2 : second_train_time = 15)
  (h3 : crossing_time = 7.5) :
  train_length / (train_length / second_train_time + train_length / crossing_time - train_length / second_train_time) = 5 := by
  sorry

#check train_crossing_time

end train_crossing_time_l1936_193654


namespace cheddar_cheese_sticks_l1936_193678

theorem cheddar_cheese_sticks (mozzarella : ℕ) (pepperjack : ℕ) (p_pepperjack : ℚ) : ℕ :=
  let total := pepperjack * 2
  let cheddar := total - mozzarella - pepperjack
  by
    have h1 : mozzarella = 30 := by sorry
    have h2 : pepperjack = 45 := by sorry
    have h3 : p_pepperjack = 1/2 := by sorry
    exact 15

#check cheddar_cheese_sticks

end cheddar_cheese_sticks_l1936_193678


namespace mortgage_more_beneficial_l1936_193653

/-- Represents the annual dividend rate of the preferred shares -/
def dividend_rate : ℝ := 0.17

/-- Represents the annual interest rate of the mortgage loan -/
def mortgage_rate : ℝ := 0.125

/-- Theorem stating that the net return from keeping shares and taking a mortgage is positive -/
theorem mortgage_more_beneficial : dividend_rate - mortgage_rate > 0 := by
  sorry

end mortgage_more_beneficial_l1936_193653


namespace two_zeros_iff_a_positive_l1936_193686

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ := (x - 2) * Real.exp x + a * (x - 1)^2

-- Define the property of having two zeros
def has_two_zeros (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0

-- Theorem statement
theorem two_zeros_iff_a_positive :
  ∀ a : ℝ, has_two_zeros (f a) ↔ a > 0 :=
sorry

end two_zeros_iff_a_positive_l1936_193686


namespace blue_balls_count_l1936_193631

theorem blue_balls_count (total : ℕ) (removed : ℕ) (prob : ℚ) : 
  total = 35 → 
  removed = 5 → 
  prob = 5 / 21 → 
  (∃ initial : ℕ, 
    initial ≤ total ∧ 
    (initial - removed : ℚ) / (total - removed : ℚ) = prob ∧ 
    initial = 12) :=
by sorry

end blue_balls_count_l1936_193631


namespace fund_raising_exceeded_goal_l1936_193671

def fund_raising (ken_amount : ℝ) : Prop :=
  let mary_amount := 5 * ken_amount
  let scott_amount := mary_amount / 3
  let total_collected := ken_amount + mary_amount + scott_amount
  let goal := 4000
  ken_amount = 600 → total_collected - goal = 600

theorem fund_raising_exceeded_goal : fund_raising 600 := by
  sorry

end fund_raising_exceeded_goal_l1936_193671


namespace sufficient_not_necessary_condition_l1936_193666

/-- Given two functions f and g, prove that |k| ≤ 2 is a sufficient but not necessary condition
for f(x) ≥ g(x) to hold for all x ∈ ℝ. -/
theorem sufficient_not_necessary_condition (k : ℝ) :
  (∀ x : ℝ, x^2 - 2*x + 3 ≥ k*x - 1) ↔ -6 ≤ k ∧ k ≤ 2 :=
sorry

end sufficient_not_necessary_condition_l1936_193666


namespace smallest_number_problem_l1936_193690

theorem smallest_number_problem (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a + b + c) / 3 = 30 →
  b = 29 →
  max a (max b c) = b + 8 →
  min a (min b c) = 24 :=
sorry

end smallest_number_problem_l1936_193690


namespace largest_number_l1936_193610

/-- Represents a real number with a repeating decimal expansion -/
def RepeatingDecimal (whole : ℕ) (nonRepeating : List ℕ) (repeating : List ℕ) : ℚ :=
  sorry

/-- The number 8.12356 -/
def num1 : ℚ := 8.12356

/-- The number 8.123$\overline{5}$ -/
def num2 : ℚ := RepeatingDecimal 8 [1, 2, 3] [5]

/-- The number 8.12$\overline{356}$ -/
def num3 : ℚ := RepeatingDecimal 8 [1, 2] [3, 5, 6]

/-- The number 8.1$\overline{2356}$ -/
def num4 : ℚ := RepeatingDecimal 8 [1] [2, 3, 5, 6]

/-- The number 8.$\overline{12356}$ -/
def num5 : ℚ := RepeatingDecimal 8 [] [1, 2, 3, 5, 6]

theorem largest_number : 
  num2 > num1 ∧ num2 > num3 ∧ num2 > num4 ∧ num2 > num5 :=
sorry

end largest_number_l1936_193610


namespace youngest_child_age_l1936_193608

def mother_charge : ℝ := 5.05
def child_charge_per_year : ℝ := 0.55
def total_bill : ℝ := 11.05

def is_valid_age_combination (twin_age : ℕ) (youngest_age : ℕ) : Prop :=
  twin_age > youngest_age ∧
  mother_charge + child_charge_per_year * (2 * twin_age + youngest_age) = total_bill

theorem youngest_child_age :
  ∀ youngest_age : ℕ,
    (∃ twin_age : ℕ, is_valid_age_combination twin_age youngest_age) ↔
    (youngest_age = 1 ∨ youngest_age = 3) :=
by sorry

end youngest_child_age_l1936_193608


namespace trigonometric_identities_l1936_193602

theorem trigonometric_identities :
  let a := Real.sqrt 2 / 2 * (Real.cos (15 * π / 180) - Real.sin (15 * π / 180))
  let b := Real.cos (π / 12) ^ 2 - Real.sin (π / 12) ^ 2
  let c := Real.tan (22.5 * π / 180) / (1 - Real.tan (22.5 * π / 180) ^ 2)
  let d := Real.sin (15 * π / 180) * Real.cos (15 * π / 180)
  (a = 1/2) ∧ 
  (c = 1/2) ∧ 
  (b ≠ 1/2) ∧ 
  (d ≠ 1/2) := by
sorry

end trigonometric_identities_l1936_193602


namespace triangle_angles_from_height_intersections_l1936_193635

/-- Given an acute-angled triangle ABC with circumscribed circle,
    let p, q, r be positive real numbers representing the ratio of arc lengths
    formed by the intersections of the extended heights with the circle.
    This theorem states the relationship between these ratios and the angles of the triangle. -/
theorem triangle_angles_from_height_intersections
  (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  let α := Real.pi / 2 * ((q + r) / (p + q + r))
  let β := Real.pi / 2 * (q / (p + q + r))
  let γ := Real.pi / 2 * (r / (p + q + r))
  α + β + γ = Real.pi ∧ 0 < α ∧ α < Real.pi/2 ∧ 0 < β ∧ β < Real.pi/2 ∧ 0 < γ ∧ γ < Real.pi/2 := by
  sorry

end triangle_angles_from_height_intersections_l1936_193635


namespace total_jokes_after_four_weeks_l1936_193606

def total_jokes (initial_jessy initial_alan initial_tom initial_emily : ℕ)
                (rate_jessy rate_alan rate_tom rate_emily : ℕ)
                (weeks : ℕ) : ℕ :=
  let jessy := initial_jessy * (rate_jessy ^ weeks - 1) / (rate_jessy - 1)
  let alan := initial_alan * (rate_alan ^ weeks - 1) / (rate_alan - 1)
  let tom := initial_tom * (rate_tom ^ weeks - 1) / (rate_tom - 1)
  let emily := initial_emily * (rate_emily ^ weeks - 1) / (rate_emily - 1)
  jessy + alan + tom + emily

theorem total_jokes_after_four_weeks :
  total_jokes 11 7 5 3 3 2 4 4 4 = 1225 := by
  sorry

end total_jokes_after_four_weeks_l1936_193606


namespace special_triangle_sides_l1936_193600

/-- A triangle with an inscribed circle that passes through trisection points of a median -/
structure SpecialTriangle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The circle passes through trisection points of a median -/
  trisects_median : Bool
  /-- The sides of the triangle -/
  side_a : ℝ
  side_b : ℝ
  side_c : ℝ

/-- The theorem about the special triangle -/
theorem special_triangle_sides (t : SpecialTriangle) 
  (h_radius : t.r = 3 * Real.sqrt 2)
  (h_trisects : t.trisects_median = true) :
  t.side_a = 5 * Real.sqrt 7 ∧ 
  t.side_b = 13 * Real.sqrt 7 ∧ 
  t.side_c = 10 * Real.sqrt 7 := by
  sorry

end special_triangle_sides_l1936_193600


namespace code_transformation_correct_l1936_193670

def initial_code : List (Fin 10 × Fin 10 × Fin 10 × Fin 10) :=
  [(4, 0, 2, 2), (0, 7, 1, 0), (4, 1, 9, 9)]

def complement_to_nine (n : Fin 10) : Fin 10 :=
  9 - n

def apply_rule (segment : Fin 10 × Fin 10 × Fin 10 × Fin 10) : Fin 10 × Fin 10 × Fin 10 × Fin 10 :=
  let (a, b, c, d) := segment
  (a, complement_to_nine b, c, complement_to_nine d)

def new_code : List (Fin 10 × Fin 10 × Fin 10 × Fin 10) :=
  initial_code.map apply_rule

theorem code_transformation_correct :
  new_code = [(4, 9, 2, 7), (0, 2, 1, 9), (4, 8, 9, 0)] :=
by sorry

end code_transformation_correct_l1936_193670


namespace geometric_mean_minimum_l1936_193643

theorem geometric_mean_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_geom_mean : Real.sqrt 3 = Real.sqrt (3^a * 3^b)) :
  (1/a + 1/b) ≥ 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧
    Real.sqrt 3 = Real.sqrt (3^a₀ * 3^b₀) ∧ 1/a₀ + 1/b₀ = 4 := by
  sorry

end geometric_mean_minimum_l1936_193643


namespace polynomial_expansion_equality_l1936_193611

theorem polynomial_expansion_equality (x : ℝ) :
  (3*x - 2) * (6*x^8 + 3*x^7 - 2*x^3 + x) = 18*x^9 - 3*x^8 - 6*x^7 - 6*x^4 - 4*x^3 + x :=
by
  sorry

end polynomial_expansion_equality_l1936_193611


namespace money_sharing_l1936_193619

theorem money_sharing (amanda ben carlos total : ℕ) : 
  amanda + ben + carlos = total →
  amanda * 5 = ben * 3 →
  carlos * 5 = ben * 12 →
  ben = 25 →
  total = 100 := by
sorry

end money_sharing_l1936_193619


namespace rectangle_width_decrease_l1936_193661

theorem rectangle_width_decrease (L W : ℝ) (L' W' : ℝ) (h1 : L' = 1.3 * L) (h2 : L * W = L' * W') : 
  (W - W') / W = 23.08 / 100 :=
sorry

end rectangle_width_decrease_l1936_193661


namespace triangle_theorem_l1936_193665

noncomputable section

theorem triangle_theorem (a b c A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  a * Real.sin A = 4 * b * Real.sin B →
  a * c = Real.sqrt 5 * (a^2 - b^2 - c^2) →
  Real.cos A = -Real.sqrt 5 / 5 ∧
  Real.sin (2 * B - A) = -2 * Real.sqrt 5 / 5 := by
  sorry

end

end triangle_theorem_l1936_193665


namespace pizza_toppings_combinations_l1936_193683

def total_toppings : ℕ := 9
def toppings_to_select : ℕ := 4
def required_toppings : ℕ := 2

theorem pizza_toppings_combinations :
  (Nat.choose total_toppings toppings_to_select) -
  (Nat.choose (total_toppings - required_toppings) toppings_to_select) = 91 := by
  sorry

end pizza_toppings_combinations_l1936_193683


namespace additional_money_needed_l1936_193624

def new_computer_cost : ℕ := 80
def initial_savings : ℕ := 50
def old_computer_sale : ℕ := 20

theorem additional_money_needed : 
  new_computer_cost - (initial_savings + old_computer_sale) = 10 := by
  sorry

end additional_money_needed_l1936_193624


namespace perpendicular_bisector_of_common_chord_l1936_193645

-- Define the curves C1 and C2
def C1 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

def C2 (x y : ℝ) : Prop := x^2 + y^2 = 4*y

-- Define the polar equation of the perpendicular bisector
def perpendicular_bisector (ρ θ : ℝ) : Prop :=
  ρ * Real.cos (θ - Real.pi/4) = Real.sqrt 2

-- Theorem statement
theorem perpendicular_bisector_of_common_chord :
  ∀ (x y ρ θ : ℝ), C1 x y → C2 x y →
  x = ρ * Real.cos θ → y = ρ * Real.sin θ →
  perpendicular_bisector ρ θ :=
sorry

end perpendicular_bisector_of_common_chord_l1936_193645


namespace quadratic_root_proof_l1936_193697

theorem quadratic_root_proof : let x : ℝ := (-15 - Real.sqrt 181) / 8
  ∀ u : ℝ, u = 2.75 → 4 * x^2 + 15 * x + u = 0 :=
by
  sorry

end quadratic_root_proof_l1936_193697


namespace unknown_blanket_rate_l1936_193622

theorem unknown_blanket_rate (price1 price2 avg_price : ℕ) 
  (count1 count2 count_unknown : ℕ) (total_count : ℕ) :
  price1 = 100 →
  price2 = 150 →
  avg_price = 150 →
  count1 = 2 →
  count2 = 5 →
  count_unknown = 2 →
  total_count = count1 + count2 + count_unknown →
  ∃ (unknown_price : ℕ), 
    (count1 * price1 + count2 * price2 + count_unknown * unknown_price) / total_count = avg_price ∧
    unknown_price = 200 :=
by sorry

end unknown_blanket_rate_l1936_193622


namespace intersection_range_is_correct_l1936_193699

/-- Line l with parameter t -/
structure Line where
  a : ℝ
  x : ℝ → ℝ
  y : ℝ → ℝ
  h1 : ∀ t, x t = a - 2 * t * (y t)
  h2 : ∀ t, y t = -4 * t

/-- Circle C with parameter θ -/
structure Circle where
  x : ℝ → ℝ
  y : ℝ → ℝ
  h1 : ∀ θ, x θ = 4 * Real.cos θ
  h2 : ∀ θ, y θ = 4 * Real.sin θ

/-- The range of a for which line l intersects circle C -/
def intersectionRange (l : Line) (c : Circle) : Set ℝ :=
  { a | ∃ t θ, l.x t = c.x θ ∧ l.y t = c.y θ }

theorem intersection_range_is_correct (l : Line) (c : Circle) :
  intersectionRange l c = Set.Icc (-4 * Real.sqrt 5) (4 * Real.sqrt 5) := by
  sorry

#check intersection_range_is_correct

end intersection_range_is_correct_l1936_193699


namespace divisibility_of_prime_square_minus_one_l1936_193676

theorem divisibility_of_prime_square_minus_one (p : ℕ) (h_prime : Nat.Prime p) (h_ge_five : p ≥ 5) :
  24 ∣ (p^2 - 1) :=
by sorry

end divisibility_of_prime_square_minus_one_l1936_193676


namespace product_of_fractions_l1936_193637

theorem product_of_fractions : 
  (1 + 1/2) * (1 + 1/3) * (1 + 1/4) * (1 + 1/5) * (1 + 1/6) * (1 + 1/7) = 8 := by
sorry

end product_of_fractions_l1936_193637


namespace min_value_of_P_sum_l1936_193646

def P (τ : ℝ) : ℝ := (τ + 1)^3

theorem min_value_of_P_sum (x y : ℝ) (h : x + y = 0) :
  ∃ (m : ℝ), m = 2 ∧ ∀ (a b : ℝ), a + b = 0 → P a + P b ≥ m :=
by sorry

end min_value_of_P_sum_l1936_193646


namespace cubic_function_properties_l1936_193609

-- Define the cubic function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x - 2

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3*x^2 - 3

-- Theorem stating the properties of f(x)
theorem cubic_function_properties :
  (∀ x, f' x = 0 ↔ x = 1 ∨ x = -1) ∧
  f (-2) = -4 ∧
  f (-1) = 0 ∧
  f 1 = -4 ∧
  (∀ x, x < -1 → f' x > 0) ∧
  (∀ x, x > 1 → f' x > 0) ∧
  (∀ x, -1 < x ∧ x < 1 → f' x < 0) :=
by sorry

end cubic_function_properties_l1936_193609


namespace function_property_l1936_193617

theorem function_property (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, 2 * f x - f (-x) = 3 * x + 1) : 
  f 1 = 2 := by
  sorry

end function_property_l1936_193617


namespace four_divisions_for_400_to_25_l1936_193648

/-- The number of divisions needed to reduce a collection of books to a target group size -/
def divisions_needed (total_books : ℕ) (target_group_size : ℕ) : ℕ :=
  if total_books ≤ target_group_size then 0
  else 1 + divisions_needed (total_books / 2) target_group_size

/-- Theorem stating that 4 divisions are needed to reduce 400 books to groups of 25 -/
theorem four_divisions_for_400_to_25 :
  divisions_needed 400 25 = 4 := by
sorry

end four_divisions_for_400_to_25_l1936_193648


namespace correct_equation_representation_l1936_193687

/-- Represents a rectangular field with width and length in steps -/
structure RectangularField where
  width : ℝ
  length : ℝ

/-- The area of a rectangular field in square steps -/
def area (field : RectangularField) : ℝ :=
  field.width * field.length

/-- Theorem stating that the equation x(x+12) = 864 correctly represents the problem -/
theorem correct_equation_representation (x : ℝ) :
  let field := RectangularField.mk x (x + 12)
  area field = 864 → x * (x + 12) = 864 := by
  sorry

end correct_equation_representation_l1936_193687


namespace circle_angle_theorem_l1936_193693

-- Define the circle and angles
def Circle (F : Point) : Prop := sorry

def angle (A B C : Point) : ℝ := sorry

-- State the theorem
theorem circle_angle_theorem (F A B C D E : Point) :
  Circle F →
  angle B F C = 2 * angle A F B →
  angle C F D = 3 * angle A F B →
  angle D F E = 4 * angle A F B →
  angle E F A = 5 * angle A F B →
  angle B F C = 48 := by
  sorry

end circle_angle_theorem_l1936_193693


namespace two_digit_reverse_sum_square_l1936_193612

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem two_digit_reverse_sum_square :
  {n : ℕ | is_two_digit n ∧ is_perfect_square (n + reverse_digits n)} =
  {29, 38, 47, 56, 65, 74, 83, 92} := by sorry

end two_digit_reverse_sum_square_l1936_193612


namespace all_PQ_pass_through_common_point_l1936_193668

-- Define the circle S
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a line
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

-- Define the problem setup
structure Setup where
  S : Circle
  A : ℝ × ℝ
  B : ℝ × ℝ
  L : Line
  c : ℝ

-- Define the condition for X and Y
def satisfiesCondition (setup : Setup) (X Y : ℝ × ℝ) : Prop :=
  X ≠ Y ∧ 
  (X.1 - setup.A.1) * (Y.1 - setup.A.1) + (X.2 - setup.A.2) * (Y.2 - setup.A.2) = setup.c

-- Define the intersection points P and Q
def getIntersectionP (setup : Setup) (X : ℝ × ℝ) : ℝ × ℝ := sorry
def getIntersectionQ (setup : Setup) (Y : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the line PQ
def linePQ (P Q : ℝ × ℝ) : Line := ⟨P, Q⟩

-- Theorem statement
theorem all_PQ_pass_through_common_point (setup : Setup) :
  ∃ (commonPoint : ℝ × ℝ), ∀ (X Y : ℝ × ℝ),
    satisfiesCondition setup X Y →
    let P := getIntersectionP setup X
    let Q := getIntersectionQ setup Y
    let PQ := linePQ P Q
    -- The common point lies on line PQ
    (commonPoint.1 - PQ.point1.1) * (PQ.point2.2 - PQ.point1.2) = 
    (commonPoint.2 - PQ.point1.2) * (PQ.point2.1 - PQ.point1.1) :=
sorry

end all_PQ_pass_through_common_point_l1936_193668


namespace inequality_propositions_l1936_193626

theorem inequality_propositions :
  ∃ (correct : Finset (Fin 4)), correct.card = 2 ∧
  (∀ i, i ∈ correct ↔
    (i = 0 ∧ (∀ a b c : ℝ, a * c^2 > b * c^2 → a > b)) ∨
    (i = 1 ∧ (∀ a b c d : ℝ, a > b → c > d → a + c > b + d)) ∨
    (i = 2 ∧ (∀ a b c d : ℝ, a > b → c > d → a * c > b * d)) ∨
    (i = 3 ∧ (∀ a b : ℝ, a > b → 1 / a > 1 / b))) :=
by sorry

end inequality_propositions_l1936_193626


namespace unfair_coin_probability_l1936_193696

/-- The probability of getting exactly k successes in n independent Bernoulli trials -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

/-- The probability of flipping exactly 3 heads in 8 flips of an unfair coin -/
theorem unfair_coin_probability : 
  binomial_probability 8 3 (1/3) = 1792/6561 := by
  sorry


end unfair_coin_probability_l1936_193696


namespace octagon_diagonals_l1936_193650

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon is a polygon with 8 sides -/
def octagon_sides : ℕ := 8

/-- Theorem: The number of diagonals in an octagon is 20 -/
theorem octagon_diagonals : num_diagonals octagon_sides = 20 := by
  sorry

end octagon_diagonals_l1936_193650


namespace unique_integer_solution_l1936_193639

theorem unique_integer_solution : ∃! (x y : ℤ), x^4 + y^2 - 4*y + 4 = 4 := by sorry

end unique_integer_solution_l1936_193639


namespace melanie_initial_dimes_l1936_193664

/-- The number of dimes Melanie initially had in her bank -/
def initial_dimes : ℕ := sorry

/-- The number of dimes Melanie's dad gave her -/
def dimes_from_dad : ℕ := 8

/-- The number of dimes Melanie gave to her mother -/
def dimes_to_mother : ℕ := 4

/-- The number of dimes Melanie has now -/
def current_dimes : ℕ := 11

theorem melanie_initial_dimes : 
  initial_dimes + dimes_from_dad - dimes_to_mother = current_dimes := by sorry

end melanie_initial_dimes_l1936_193664


namespace f_properties_l1936_193655

noncomputable def f (x : ℝ) := Real.sqrt 3 * (Real.sin x ^ 2 - Real.cos x ^ 2) - 2 * Real.sin x * Real.cos x

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S ∧ S < T → ∃ y, f (y + S) ≠ f y) ∧
  (∀ x ∈ Set.Icc (-Real.pi/3) (Real.pi/12), ∀ y ∈ Set.Icc (-Real.pi/3) (Real.pi/12), x ≤ y → f y ≤ f x) ∧
  (∀ x ∈ Set.Icc (Real.pi/12) (Real.pi/3), ∀ y ∈ Set.Icc (Real.pi/12) (Real.pi/3), x ≤ y → f x ≤ f y) :=
by sorry

end f_properties_l1936_193655


namespace inequality_proof_l1936_193641

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) :
  (|x^2 + y^2|) / (x + y) < (|x^2 - y^2|) / (x - y) := by
  sorry

end inequality_proof_l1936_193641


namespace cinema_chairs_l1936_193662

/-- The total number of chairs in a cinema with a given number of rows and chairs per row. -/
def total_chairs (rows : ℕ) (chairs_per_row : ℕ) : ℕ := rows * chairs_per_row

/-- Theorem: The total number of chairs in a cinema with 4 rows and 8 chairs per row is 32. -/
theorem cinema_chairs : total_chairs 4 8 = 32 := by
  sorry

end cinema_chairs_l1936_193662


namespace square_area_ratio_l1936_193615

/-- The ratio of the area of a square with side length x to the area of a square with side length 3x is 1/9 -/
theorem square_area_ratio (x : ℝ) (hx : x > 0) : 
  (x^2) / ((3*x)^2) = 1 / 9 := by
sorry

end square_area_ratio_l1936_193615


namespace line_parallel_to_parallel_plane_l1936_193640

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (containedIn : Line → Plane → Prop)
variable (parallelTo : Line → Plane → Prop)
variable (planeparallel : Plane → Plane → Prop)

-- State the theorem
theorem line_parallel_to_parallel_plane 
  (m : Line) (α β : Plane) :
  containedIn m α → planeparallel α β → parallelTo m β := by
  sorry

end line_parallel_to_parallel_plane_l1936_193640


namespace tree_scenario_result_l1936_193663

/-- Represents the number of caterpillars and leaves eaten in a tree scenario -/
def tree_scenario (initial_caterpillars storm_fallen hatched_eggs 
                   baby_leaves_eaten cocoon_left moth_ratio
                   moth_daily_consumption days : ℕ) : ℕ × ℕ :=
  let remaining_after_storm := initial_caterpillars - storm_fallen
  let total_after_hatch := remaining_after_storm + hatched_eggs
  let remaining_after_cocoon := total_after_hatch - cocoon_left
  let moth_caterpillars := remaining_after_cocoon / 2
  let total_leaves_eaten := baby_leaves_eaten + 
    moth_caterpillars * moth_daily_consumption * days
  (remaining_after_cocoon, total_leaves_eaten)

/-- Theorem stating the result of the tree scenario -/
theorem tree_scenario_result : 
  tree_scenario 14 3 6 18 9 2 4 7 = (8, 130) :=
by sorry

end tree_scenario_result_l1936_193663


namespace arithmetic_mean_after_removal_l1936_193604

theorem arithmetic_mean_after_removal (S : Finset ℝ) (x y : ℝ) :
  S.card = 60 →
  x ∈ S →
  y ∈ S →
  x = 60 →
  y = 75 →
  (S.sum id) / S.card = 45 →
  ((S.sum id - (x + y)) / (S.card - 2) : ℝ) = 465 / 106 := by
  sorry

end arithmetic_mean_after_removal_l1936_193604


namespace inequality_for_positive_reals_l1936_193623

theorem inequality_for_positive_reals : ∀ x : ℝ, x > 0 → x + 4/x ≥ 4 := by
  sorry

end inequality_for_positive_reals_l1936_193623


namespace power_fraction_multiply_simplify_fraction_two_thirds_cubed_times_half_l1936_193689

theorem power_fraction_multiply (a b c d : ℚ) :
  (a / b) ^ 3 * (c / d) = (a ^ 3 * c) / (b ^ 3 * d) :=
by sorry

theorem simplify_fraction_two_thirds_cubed_times_half :
  (2 / 3 : ℚ) ^ 3 * (1 / 2) = 4 / 27 :=
by sorry

end power_fraction_multiply_simplify_fraction_two_thirds_cubed_times_half_l1936_193689


namespace probability_123_in_10_rolls_l1936_193620

theorem probability_123_in_10_rolls (n : ℕ) (h : n = 10) :
  let total_outcomes := 6^n
  let favorable_outcomes := 8 * 6^7 - 15 * 6^4 + 4 * 6
  (favorable_outcomes : ℚ) / total_outcomes = 2220072 / 6^10 :=
by sorry

end probability_123_in_10_rolls_l1936_193620


namespace parabola_ellipse_focus_coincide_l1936_193603

/-- The value of p for which the focus of the parabola y^2 = -2px coincides with the left focus of the ellipse (x^2/16) + (y^2/12) = 1 -/
theorem parabola_ellipse_focus_coincide : ∃ p : ℝ,
  (∀ x y : ℝ, y^2 = -2*p*x → (x^2/16 + y^2/12 = 1 → x = -2)) →
  p = 4 := by
  sorry

end parabola_ellipse_focus_coincide_l1936_193603


namespace library_wall_leftover_space_l1936_193675

theorem library_wall_leftover_space
  (wall_length : ℝ)
  (desk_length : ℝ)
  (bookcase_length : ℝ)
  (min_spacing : ℝ)
  (h_wall : wall_length = 15)
  (h_desk : desk_length = 2)
  (h_bookcase : bookcase_length = 1.5)
  (h_spacing : min_spacing = 0.5)
  : ∃ (n : ℕ), 
    n * (desk_length + bookcase_length + min_spacing) ≤ wall_length ∧
    (n + 1) * (desk_length + bookcase_length + min_spacing) > wall_length ∧
    wall_length - n * (desk_length + bookcase_length + min_spacing) = 3 :=
by sorry

end library_wall_leftover_space_l1936_193675


namespace angle_measure_l1936_193679

theorem angle_measure (α : Real) : 
  (90 - α) + (90 - (180 - α)) = 90 → α = 45 := by
  sorry

end angle_measure_l1936_193679


namespace hyperbola_triangle_perimeter_l1936_193636

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola with foci F₁ and F₂, and real axis length 2a -/
structure Hyperbola where
  F₁ : Point
  F₂ : Point
  a : ℝ

/-- Theorem: Perimeter of triangle ABF₂ in a hyperbola -/
theorem hyperbola_triangle_perimeter 
  (h : Hyperbola) 
  (A B : Point) 
  (m : ℝ) 
  (h_line : A.x = B.x ∧ A.x = h.F₁.x) -- A, B, and F₁ are collinear
  (h_on_hyperbola : 
    |A.x - h.F₂.x| + |A.y - h.F₂.y| - |A.x - h.F₁.x| - |A.y - h.F₁.y| = 2 * h.a ∧
    |B.x - h.F₂.x| + |B.y - h.F₂.y| - |B.x - h.F₁.x| - |B.y - h.F₁.y| = 2 * h.a)
  (h_AB : |A.x - B.x| + |A.y - B.y| = m) :
  |A.x - h.F₂.x| + |A.y - h.F₂.y| + |B.x - h.F₂.x| + |B.y - h.F₂.y| + m = 4 * h.a + 2 * m := by
  sorry


end hyperbola_triangle_perimeter_l1936_193636


namespace bargain_bin_books_theorem_l1936_193642

/-- Calculates the number of books in the bargain bin after two weeks of sales and additions. -/
def books_after_two_weeks (initial : ℕ) (sold_week1 sold_week2 added_week1 added_week2 : ℕ) : ℕ :=
  initial - sold_week1 + added_week1 - sold_week2 + added_week2

/-- Theorem stating that given the initial number of books and the changes during two weeks,
    the final number of books in the bargain bin is 391. -/
theorem bargain_bin_books_theorem :
  books_after_two_weeks 500 115 289 65 230 = 391 := by
  sorry

#eval books_after_two_weeks 500 115 289 65 230

end bargain_bin_books_theorem_l1936_193642


namespace two_digit_multiplication_l1936_193601

theorem two_digit_multiplication (a b c d : ℕ) :
  (a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10) →
  ((b = d ∧ a + c = 10) ∨ (a = c ∧ b + d = 10) ∨ (c = d ∧ a + b = 10)) →
  (10 * a + b) * (10 * c + d) = 
    (if b = d ∧ a + c = 10 then 100 * (a^2 + a) + b * d
     else if a = c ∧ b + d = 10 then 100 * a * c + 100 * b + b^2
     else 100 * a * c + 100 * c + b * c) :=
by sorry

end two_digit_multiplication_l1936_193601


namespace paper_parts_cannot_reach_2020_can_reach_2023_l1936_193607

def paper_sequence : Nat → Nat
  | 0 => 1
  | n + 1 => paper_sequence n + 2

theorem paper_parts (n : Nat) : 
  paper_sequence n = 2 * n + 1 := by sorry

theorem cannot_reach_2020 : 
  ∀ n, paper_sequence n ≠ 2020 := by sorry

theorem can_reach_2023 : 
  ∃ n, paper_sequence n = 2023 := by sorry

end paper_parts_cannot_reach_2020_can_reach_2023_l1936_193607


namespace solve_linear_equation_l1936_193680

theorem solve_linear_equation :
  ∃ x : ℝ, 3 * x - 7 = 2 * x + 5 ∧ x = 12 := by sorry

end solve_linear_equation_l1936_193680


namespace equation_solution_l1936_193649

theorem equation_solution :
  ∃ x : ℚ, (3 * x + 4 * x = 600 - (5 * x + 6 * x)) ∧ (x = 100 / 3) := by
  sorry

end equation_solution_l1936_193649


namespace confidence_level_interpretation_l1936_193681

theorem confidence_level_interpretation 
  (confidence_level : ℝ) 
  (hypothesis_test : Type) 
  (is_valid_test : hypothesis_test → Prop) 
  (test_result : hypothesis_test → Bool) 
  (h_confidence : confidence_level = 0.95) :
  ∃ (error_probability : ℝ), 
    error_probability = 1 - confidence_level ∧ 
    error_probability = 0.05 := by
  sorry

end confidence_level_interpretation_l1936_193681


namespace cone_volume_l1936_193652

theorem cone_volume (central_angle : Real) (sector_area : Real) :
  central_angle = 120 * Real.pi / 180 →
  sector_area = 3 * Real.pi →
  ∃ (volume : Real), volume = (2 * Real.sqrt 2 / 3) * Real.pi :=
by sorry

end cone_volume_l1936_193652


namespace square_root_equation_solution_l1936_193632

theorem square_root_equation_solution (x : ℝ) :
  Real.sqrt (2 - 5 * x + x^2) = 9 ↔ x = (5 + Real.sqrt 341) / 2 ∨ x = (5 - Real.sqrt 341) / 2 := by
  sorry

end square_root_equation_solution_l1936_193632


namespace isosceles_triangle_area_l1936_193614

/-- An isosceles triangle with given altitude and perimeter -/
structure IsoscelesTriangle where
  altitude : ℝ
  perimeter : ℝ

/-- The area of an isosceles triangle -/
def area (t : IsoscelesTriangle) : ℝ :=
  sorry

/-- Theorem: The area of an isosceles triangle with altitude 10 and perimeter 40 is 75 -/
theorem isosceles_triangle_area :
  ∀ t : IsoscelesTriangle, t.altitude = 10 ∧ t.perimeter = 40 → area t = 75 := by
  sorry

end isosceles_triangle_area_l1936_193614


namespace arithmetic_sequence_problem_l1936_193605

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_a2 : a 2 = 3) 
  (h_sum : a 3 + a 4 = 9) : 
  a 1 * a 6 = 14 := by
sorry

end arithmetic_sequence_problem_l1936_193605


namespace area_ratio_is_9_32_l1936_193656

-- Define the triangle XYZ
structure Triangle :=
  (XY YZ XZ : ℝ)

-- Define the points M, N, O
structure Points (t : Triangle) :=
  (p q r : ℝ)
  (p_pos : p > 0)
  (q_pos : q > 0)
  (r_pos : r > 0)
  (sum_eq : p + q + r = 3/4)
  (sum_sq_eq : p^2 + q^2 + r^2 = 1/2)

-- Define the function to calculate the ratio of areas
def areaRatio (t : Triangle) (pts : Points t) : ℝ :=
  -- The actual calculation of the ratio would go here
  sorry

-- State the theorem
theorem area_ratio_is_9_32 (t : Triangle) (pts : Points t) 
  (h1 : t.XY = 12) (h2 : t.YZ = 16) (h3 : t.XZ = 20) : 
  areaRatio t pts = 9/32 := by
  sorry

end area_ratio_is_9_32_l1936_193656


namespace division_problem_l1936_193651

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) : 
  dividend = 127 → 
  divisor = 25 → 
  remainder = 2 → 
  dividend = divisor * quotient + remainder → 
  quotient = 5 := by
sorry

end division_problem_l1936_193651


namespace least_number_of_marbles_l1936_193682

theorem least_number_of_marbles (x : ℕ) : x = 50 ↔ 
  x > 0 ∧ 
  x % 6 = 2 ∧ 
  x % 4 = 3 ∧ 
  ∀ y : ℕ, y > 0 ∧ y % 6 = 2 ∧ y % 4 = 3 → x ≤ y :=
by sorry

end least_number_of_marbles_l1936_193682


namespace quadratic_roots_property_l1936_193694

theorem quadratic_roots_property (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   x₁^2 + a*x₁ + 4 = 0 ∧ 
   x₂^2 + a*x₂ + 4 = 0 ∧ 
   x₁^2 - 20/(3*x₂^3) = x₂^2 - 20/(3*x₁^3)) → 
  a = -10 :=
by sorry

end quadratic_roots_property_l1936_193694


namespace polynomial_evaluation_l1936_193688

theorem polynomial_evaluation (x : ℝ) (h : x = 4) : x^4 + x^3 + x^2 + x + 1 = 341 := by
  sorry

end polynomial_evaluation_l1936_193688


namespace quadratic_max_l1936_193647

-- Define the quadratic function
def f (x : ℝ) : ℝ := -(x - 2)^2 - 3

-- State the theorem
theorem quadratic_max (x : ℝ) : 
  (∀ y : ℝ, f y ≤ f 2) ∧ f 2 = -3 := by sorry

end quadratic_max_l1936_193647


namespace solution_sets_equivalent_l1936_193618

theorem solution_sets_equivalent : 
  {x : ℝ | |8*x + 9| < 7} = {x : ℝ | -4*x^2 - 9*x - 2 > 0} := by sorry

end solution_sets_equivalent_l1936_193618


namespace semicircles_area_ratio_l1936_193628

theorem semicircles_area_ratio (r : ℝ) (hr : r > 0) : 
  let circle_area := π * r^2
  let semicircle1_area := π * (r/2)^2 / 2
  let semicircle2_area := π * (r/3)^2 / 2
  (semicircle1_area + semicircle2_area) / circle_area = 13/72 := by
  sorry

end semicircles_area_ratio_l1936_193628


namespace square_pyramid_volume_l1936_193695

/-- The volume of a square pyramid inscribed in a cube -/
theorem square_pyramid_volume (cube_side_length : ℝ) (pyramid_volume : ℝ) :
  cube_side_length = 3 →
  pyramid_volume = (1 / 3) * (cube_side_length ^ 3) →
  pyramid_volume = 9 := by
sorry

end square_pyramid_volume_l1936_193695


namespace log_sum_sqrt_equality_l1936_193677

theorem log_sum_sqrt_equality : Real.sqrt (Real.log 8 / Real.log 4 + Real.log 16 / Real.log 8) = Real.sqrt (17 / 6) := by
  sorry

end log_sum_sqrt_equality_l1936_193677


namespace sqrt_expression_equals_eight_l1936_193674

theorem sqrt_expression_equals_eight :
  (3 * Real.sqrt 48 - 2 * Real.sqrt 12) / Real.sqrt 3 = 8 := by
  sorry

end sqrt_expression_equals_eight_l1936_193674


namespace total_money_l1936_193634

-- Define Tim's and Alice's money as fractions of a dollar
def tim_money : ℚ := 5/8
def alice_money : ℚ := 2/5

-- Theorem statement
theorem total_money :
  tim_money + alice_money = 1.025 := by sorry

end total_money_l1936_193634


namespace modular_arithmetic_proof_l1936_193667

theorem modular_arithmetic_proof :
  ∃ (a b : ℤ), (a * 7 ≡ 1 [ZMOD 63]) ∧ 
               (b * 13 ≡ 1 [ZMOD 63]) ∧ 
               ((3 * a + 5 * b) % 63 = 13) := by
  sorry

end modular_arithmetic_proof_l1936_193667


namespace unique_solution_quadratic_inequality_l1936_193684

theorem unique_solution_quadratic_inequality (a : ℝ) : 
  (∃! x : ℝ, 0 ≤ x^2 - a*x + a ∧ x^2 - a*x + a ≤ 1) → a = 2 := by
  sorry

end unique_solution_quadratic_inequality_l1936_193684


namespace largest_difference_l1936_193673

def A : ℕ := 3 * 1005^1006
def B : ℕ := 1005^1006
def C : ℕ := 1004 * 1005^1005
def D : ℕ := 3 * 1005^1005
def E : ℕ := 1005^1005
def F : ℕ := 1005^1004

theorem largest_difference (A B C D E F : ℕ) 
  (hA : A = 3 * 1005^1006)
  (hB : B = 1005^1006)
  (hC : C = 1004 * 1005^1005)
  (hD : D = 3 * 1005^1005)
  (hE : E = 1005^1005)
  (hF : F = 1005^1004) :
  (A - B > B - C) ∧ (A - B > C - D) ∧ (A - B > D - E) ∧ (A - B > E - F) :=
sorry

end largest_difference_l1936_193673


namespace lcm_14_18_20_l1936_193629

theorem lcm_14_18_20 : Nat.lcm 14 (Nat.lcm 18 20) = 1260 := by sorry

end lcm_14_18_20_l1936_193629


namespace exact_arrival_speed_l1936_193660

theorem exact_arrival_speed 
  (d : ℝ) (t : ℝ) 
  (h1 : d = 30 * (t + 1/12)) 
  (h2 : d = 70 * (t - 1/12)) : 
  d / t = 42 := by
sorry

end exact_arrival_speed_l1936_193660


namespace triangle_property_l1936_193669

theorem triangle_property (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Side lengths are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Law of sines
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C →
  -- Given condition
  Real.sin A ^ 2 - Real.sin B ^ 2 - Real.sin C ^ 2 = Real.sin B * Real.sin C →
  -- BC = 3
  a = 3 →
  -- Prove A = 2π/3
  A = 2 * π / 3 ∧
  -- Prove maximum perimeter is 3 + 2√3
  (b + c ≤ 2 * Real.sqrt 3 ∧ a + b + c ≤ 3 + 2 * Real.sqrt 3) := by
  sorry

end triangle_property_l1936_193669


namespace rightmost_three_digits_of_7_to_1997_l1936_193633

theorem rightmost_three_digits_of_7_to_1997 :
  7^1997 % 1000 = 207 := by
  sorry

end rightmost_three_digits_of_7_to_1997_l1936_193633


namespace convex_polygon_sides_l1936_193625

/-- The number of sides in a convex polygon where the sum of n-1 internal angles is 2009 degrees -/
def polygon_sides : ℕ := 14

theorem convex_polygon_sides :
  ∀ n : ℕ,
  n > 2 →
  (n - 1) * 180 < 2009 →
  n * 180 > 2009 →
  n = polygon_sides :=
by sorry

end convex_polygon_sides_l1936_193625


namespace brad_lemonade_profit_l1936_193638

/-- Calculates the net profit from a lemonade stand given the specified conditions. -/
def lemonade_stand_profit (
  glasses_per_gallon : ℕ)
  (cost_per_gallon : ℚ)
  (gallons_made : ℕ)
  (price_per_glass : ℚ)
  (glasses_drunk : ℕ)
  (glasses_unsold : ℕ) : ℚ :=
  let total_glasses := glasses_per_gallon * gallons_made
  let glasses_sold := total_glasses - (glasses_drunk + glasses_unsold)
  let total_cost := cost_per_gallon * gallons_made
  let total_revenue := price_per_glass * glasses_sold
  total_revenue - total_cost

/-- Theorem stating that Brad's net profit is $14.00 given the specified conditions. -/
theorem brad_lemonade_profit :
  lemonade_stand_profit 16 3.5 2 1 5 6 = 14 := by
  sorry

end brad_lemonade_profit_l1936_193638


namespace square_root_identity_specific_square_roots_l1936_193698

theorem square_root_identity (n : ℕ) :
  Real.sqrt (1 - (2 * n + 1) / ((n + 1) ^ 2)) = n / (n + 1) :=
sorry

theorem specific_square_roots :
  Real.sqrt (1 - 9 / 25) = 4 / 5 ∧ Real.sqrt (1 - 15 / 64) = 7 / 8 :=
sorry

end square_root_identity_specific_square_roots_l1936_193698


namespace min_segment_length_l1936_193627

/-- A cube with edge length 1 -/
structure Cube :=
  (edge_length : ℝ)
  (edge_length_eq : edge_length = 1)

/-- A point on the diagonal A₁D of the cube -/
structure PointM (cube : Cube) :=
  (coords : ℝ × ℝ × ℝ)

/-- A point on the edge CD₁ of the cube -/
structure PointN (cube : Cube) :=
  (coords : ℝ × ℝ × ℝ)

/-- The condition that MN is parallel to A₁ACC₁ -/
def is_parallel_to_diagonal_face (cube : Cube) (m : PointM cube) (n : PointN cube) : Prop :=
  sorry

/-- The length of segment MN -/
def segment_length (cube : Cube) (m : PointM cube) (n : PointN cube) : ℝ :=
  sorry

/-- The main theorem -/
theorem min_segment_length (cube : Cube) :
  ∃ (m : PointM cube) (n : PointN cube),
    is_parallel_to_diagonal_face cube m n ∧
    ∀ (m' : PointM cube) (n' : PointN cube),
      is_parallel_to_diagonal_face cube m' n' →
      segment_length cube m n ≤ segment_length cube m' n' ∧
      segment_length cube m n = Real.sqrt 3 / 3 :=
sorry

end min_segment_length_l1936_193627


namespace chord_equation_l1936_193672

/-- Given an ellipse and a point M, prove the equation of the line containing the chord with midpoint M -/
theorem chord_equation (x y : ℝ) :
  (x^2 / 4 + y^2 = 1) →  -- Ellipse equation
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    (x₁^2 / 4 + y₁^2 = 1) ∧  -- Point (x₁, y₁) is on the ellipse
    (x₂^2 / 4 + y₂^2 = 1) ∧  -- Point (x₂, y₂) is on the ellipse
    ((x₁ + x₂) / 2 = 1) ∧    -- x-coordinate of midpoint M
    ((y₁ + y₂) / 2 = 1/2) ∧  -- y-coordinate of midpoint M
    (y - 1/2 = -(1/2) * (x - 1))) →  -- Equation of the line through M with slope -1/2
  x + 2*y - 2 = 0  -- Resulting equation of the line
:= by sorry

end chord_equation_l1936_193672


namespace ice_cream_permutations_l1936_193685

/-- The number of distinct permutations of n items, where some items may be identical -/
def distinctPermutations (n : ℕ) (itemCounts : List ℕ) : ℕ :=
  Nat.factorial n / (itemCounts.map Nat.factorial).prod

theorem ice_cream_permutations :
  distinctPermutations 4 [2, 1, 1] = 12 := by
  sorry

end ice_cream_permutations_l1936_193685


namespace units_digit_of_4_pow_3_pow_5_l1936_193621

theorem units_digit_of_4_pow_3_pow_5 : (4^(3^5)) % 10 = 4 := by
  sorry

end units_digit_of_4_pow_3_pow_5_l1936_193621


namespace remainder_problem_l1936_193658

theorem remainder_problem (m n : ℕ) (h1 : m > n) (h2 : n % 6 = 3) (h3 : (m - n) % 6 = 5) :
  m % 6 = 2 := by
  sorry

end remainder_problem_l1936_193658


namespace complex_fraction_equality_l1936_193630

theorem complex_fraction_equality : 
  1 / ( 3 + 1 / ( 3 + 1 / ( 3 - 1 / 3 ) ) ) = 27/89 := by
  sorry

end complex_fraction_equality_l1936_193630


namespace duck_park_population_l1936_193659

theorem duck_park_population (initial_ducks : ℕ) (arriving_ducks : ℕ) (leaving_geese : ℕ) : 
  initial_ducks = 25 →
  arriving_ducks = 4 →
  leaving_geese = 10 →
  (initial_ducks * 2 - 10) - leaving_geese - (initial_ducks + arriving_ducks) = 1 :=
by sorry

end duck_park_population_l1936_193659


namespace min_value_theorem_l1936_193644

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1) :
  (1 / a + 3 / b) ≥ 16 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 3 * b₀ = 1 ∧ 1 / a₀ + 3 / b₀ = 16 := by
  sorry

end min_value_theorem_l1936_193644


namespace largest_five_digit_sum_20_l1936_193613

/-- A function that returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A function that checks if a number is five-digit -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

theorem largest_five_digit_sum_20 : 
  ∀ n : ℕ, is_five_digit n → sum_of_digits n = 20 → n ≤ 99200 := by sorry

end largest_five_digit_sum_20_l1936_193613


namespace break_time_is_30_minutes_l1936_193657

/-- Represents the travel scenario with three train stations -/
structure TravelScenario where
  /-- Time between each station in hours -/
  station_distance : ℝ
  /-- Total travel time including break in minutes -/
  total_time : ℝ

/-- Calculates the break time at the second station -/
def break_time (scenario : TravelScenario) : ℝ :=
  scenario.total_time - 2 * (scenario.station_distance * 60)

/-- Theorem stating that the break time is 30 minutes -/
theorem break_time_is_30_minutes (scenario : TravelScenario) 
  (h1 : scenario.station_distance = 2)
  (h2 : scenario.total_time = 270) : 
  break_time scenario = 30 := by
  sorry

end break_time_is_30_minutes_l1936_193657
