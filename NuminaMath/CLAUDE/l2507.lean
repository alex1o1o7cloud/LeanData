import Mathlib

namespace motorboat_travel_time_l2507_250768

/-- Represents the scenario of a motorboat and kayak traveling on a river -/
structure RiverTravel where
  r : ℝ  -- Speed of the river current (and kayak's speed)
  m : ℝ  -- Speed of the motorboat relative to the river
  t : ℝ  -- Time for motorboat to travel from X to Y
  total_time : ℝ  -- Total time until motorboat meets kayak

/-- The theorem representing the problem -/
theorem motorboat_travel_time (rt : RiverTravel) : 
  rt.m = rt.r ∧ rt.total_time = 8 → rt.t = 4 := by
  sorry

#check motorboat_travel_time

end motorboat_travel_time_l2507_250768


namespace regularPolygonProperties_givenPolygonSatisfiesProperties_l2507_250789

-- Define a regular polygon
structure RegularPolygon where
  sides : ℕ
  exteriorAngle : ℝ
  interiorAngle : ℝ

-- Define the properties of the given regular polygon
def givenPolygon : RegularPolygon where
  sides := 20
  exteriorAngle := 18
  interiorAngle := 162

-- Theorem statement
theorem regularPolygonProperties (p : RegularPolygon) 
  (h1 : p.exteriorAngle = 18) : 
  p.sides = 20 ∧ p.interiorAngle = 162 := by
  sorry

-- Proof that the given polygon satisfies the theorem
theorem givenPolygonSatisfiesProperties : 
  givenPolygon.sides = 20 ∧ givenPolygon.interiorAngle = 162 := by
  apply regularPolygonProperties givenPolygon
  rfl

end regularPolygonProperties_givenPolygonSatisfiesProperties_l2507_250789


namespace sequence_sum_l2507_250704

/-- Given a sequence {a_n} where a_1 = 1 and S_n = n^2 * a_n for all positive integers n,
    prove that S_n = 2n / (n+1) for all positive integers n. -/
theorem sequence_sum (a : ℕ → ℚ) (S : ℕ → ℚ) :
  a 1 = 1 →
  (∀ n : ℕ, n > 0 → S n = n^2 * a n) →
  ∀ n : ℕ, n > 0 → S n = 2 * n / (n + 1) := by
  sorry

end sequence_sum_l2507_250704


namespace tangents_form_diameter_l2507_250750

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

-- Define the circle E
def circle_E (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define a point on the circle E
def point_on_E (P : ℝ × ℝ) : Prop :=
  circle_E P.1 P.2

-- Define tangent lines from P to C
def tangent_to_C (P : ℝ × ℝ) (l : ℝ → ℝ) : Prop :=
  ∃ (Q : ℝ × ℝ), ellipse_C Q.1 Q.2 ∧ l Q.1 = Q.2 ∧
  ∀ (x y : ℝ), ellipse_C x y → (y - l x) * (Q.2 - l Q.1) ≥ 0

-- Define intersection points of tangents with E
def intersect_E (P : ℝ × ℝ) (l : ℝ → ℝ) (M : ℝ × ℝ) : Prop :=
  M ≠ P ∧ circle_E M.1 M.2 ∧ l M.1 = M.2

-- Main theorem
theorem tangents_form_diameter (P M N : ℝ × ℝ) (l₁ l₂ : ℝ → ℝ) :
  point_on_E P →
  tangent_to_C P l₁ →
  tangent_to_C P l₂ →
  intersect_E P l₁ M →
  intersect_E P l₂ N →
  (M.1 + N.1 = 0 ∧ M.2 + N.2 = 0) :=
sorry

end tangents_form_diameter_l2507_250750


namespace volleyball_team_score_l2507_250793

theorem volleyball_team_score :
  let lizzie_score : ℕ := 4
  let nathalie_score : ℕ := lizzie_score + 3
  let aimee_score : ℕ := 2 * (lizzie_score + nathalie_score)
  let teammates_score : ℕ := 17
  lizzie_score + nathalie_score + aimee_score + teammates_score = 50 :=
by sorry

end volleyball_team_score_l2507_250793


namespace empty_quadratic_inequality_solution_set_l2507_250728

/-- Given a quadratic inequality ax² + bx + c < 0 with a ≠ 0, 
    if the solution set is empty, then a > 0 and Δ ≤ 0, where Δ = b² - 4ac -/
theorem empty_quadratic_inequality_solution_set 
  (a b c : ℝ) (h1 : a ≠ 0) 
  (h2 : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0) : 
  a > 0 ∧ b^2 - 4*a*c ≤ 0 := by
  sorry

end empty_quadratic_inequality_solution_set_l2507_250728


namespace algebraic_simplification_l2507_250774

theorem algebraic_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (2 / a) * (a * b / 4) = b / 2 ∧
  -6 * a * b / ((3 * b^2) / (2 * a)) = -4 * a^2 / b := by sorry

end algebraic_simplification_l2507_250774


namespace axis_of_symmetry_l2507_250717

-- Define the function f with the given property
def f : ℝ → ℝ := sorry

-- Define the property that f(x) = f(6-x) for all x
axiom f_symmetry (x : ℝ) : f x = f (6 - x)

-- State the theorem: x = 3 is the axis of symmetry
theorem axis_of_symmetry :
  ∀ (x y : ℝ), f x = y ↔ f (6 - x) = y :=
by sorry

end axis_of_symmetry_l2507_250717


namespace triangle_determinant_zero_l2507_250762

theorem triangle_determinant_zero (A B C : ℝ) (h : A + B + C = π) : 
  let matrix : Matrix (Fin 3) (Fin 3) ℝ := ![
    ![Real.sin A ^ 2, Real.cos A / Real.sin A, Real.sin A],
    ![Real.sin B ^ 2, Real.cos B / Real.sin B, Real.sin B],
    ![Real.sin C ^ 2, Real.cos C / Real.sin C, Real.sin C]
  ]
  Matrix.det matrix = 0 := by
  sorry

end triangle_determinant_zero_l2507_250762


namespace julia_tag_game_l2507_250786

theorem julia_tag_game (monday tuesday : ℕ) 
  (h1 : monday = 12) 
  (h2 : tuesday = 7) : 
  monday + tuesday = 19 := by
  sorry

end julia_tag_game_l2507_250786


namespace xy_range_l2507_250712

theorem xy_range (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + 8/y = 1) :
  ∃ (m : ℝ), m = 64 ∧ xy ≥ m ∧ ∀ (z : ℝ), z > m → ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2/a + 8/b = 1 ∧ a * b = z :=
sorry

end xy_range_l2507_250712


namespace sunglasses_cap_probability_l2507_250703

theorem sunglasses_cap_probability 
  (total_sunglasses : ℕ) 
  (total_caps : ℕ) 
  (prob_cap_and_sunglasses : ℚ) :
  total_sunglasses = 50 →
  total_caps = 35 →
  prob_cap_and_sunglasses = 2/5 →
  (prob_cap_and_sunglasses * total_caps : ℚ) / total_sunglasses = 7/25 := by
  sorry

end sunglasses_cap_probability_l2507_250703


namespace min_value_of_f_on_interval_l2507_250782

-- Define the function f
def f (x : ℝ) : ℝ := x * (3 - x^2)

-- Define the interval
def interval : Set ℝ := { x | 0 ≤ x ∧ x ≤ Real.sqrt 2 }

-- State the theorem
theorem min_value_of_f_on_interval :
  ∃ (m : ℝ), m = 0 ∧ ∀ x ∈ interval, f x ≥ m :=
sorry

end min_value_of_f_on_interval_l2507_250782


namespace base10_512_equals_base6_2212_l2507_250737

-- Define a function to convert a list of digits in base 6 to a natural number
def base6ToNat (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => d + 6 * acc) 0

-- Define the theorem
theorem base10_512_equals_base6_2212 :
  512 = base6ToNat [2, 1, 2, 2] := by
  sorry


end base10_512_equals_base6_2212_l2507_250737


namespace function_composition_equality_l2507_250799

/-- Given two functions f and g, where f(x) = Ax^2 - 3B^3 and g(x) = Bx^2,
    if B ≠ 0 and f(g(2)) = 0, then A = 3B/16 -/
theorem function_composition_equality (A B : ℝ) (hB : B ≠ 0) :
  let f := fun x => A * x^2 - 3 * B^3
  let g := fun x => B * x^2
  f (g 2) = 0 → A = 3 * B / 16 := by
  sorry

end function_composition_equality_l2507_250799


namespace edward_money_left_l2507_250788

/-- The amount of money Edward initially had to spend --/
def initial_amount : ℚ := 1780 / 100

/-- The cost of one toy car before discount --/
def toy_car_cost : ℚ := 95 / 100

/-- The number of toy cars Edward bought --/
def num_toy_cars : ℕ := 4

/-- The discount rate on toy cars --/
def toy_car_discount_rate : ℚ := 15 / 100

/-- The cost of the race track before tax --/
def race_track_cost : ℚ := 600 / 100

/-- The tax rate on the race track --/
def race_track_tax_rate : ℚ := 8 / 100

/-- The theorem stating how much money Edward has left --/
theorem edward_money_left : 
  initial_amount - 
  (num_toy_cars * toy_car_cost * (1 - toy_car_discount_rate) + 
   race_track_cost * (1 + race_track_tax_rate)) = 809 / 100 := by
  sorry

end edward_money_left_l2507_250788


namespace product_ab_equals_twelve_l2507_250781

-- Define the set A
def A (a b : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ b}

-- Define the complement of A with respect to ℝ
def complement_A : Set ℝ := {x | x < 3 ∨ x > 4}

-- Theorem statement
theorem product_ab_equals_twelve (a b : ℝ) : 
  A a b ∪ complement_A = Set.univ → a * b = 12 := by
  sorry

end product_ab_equals_twelve_l2507_250781


namespace triangle_side_length_l2507_250769

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  -- Triangle ABC is oblique (implied by the other conditions)
  -- Side lengths opposite to angles A, B, C are a, b, c respectively
  A = π / 4 →
  Real.sin A + Real.sin (B - C) = 2 * Real.sqrt 2 * Real.sin (2 * C) →
  (1 / 2) * b * c * Real.sin A = 1 →
  a = Real.sqrt 5 := by
  sorry

end triangle_side_length_l2507_250769


namespace triangle_inequality_and_sqrt_sides_l2507_250770

/-- Given a triangle with side lengths a, b, c, prove the existence of a triangle
    with side lengths √a, √b, √c and the inequality involving these lengths. -/
theorem triangle_inequality_and_sqrt_sides {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a ≤ b + c) (hbc : b ≤ a + c) (hca : c ≤ a + b) :
  (∃ (u v w : ℝ), u > 0 ∧ v > 0 ∧ w > 0 ∧ 
    a = v + w ∧ b = u + w ∧ c = u + v) ∧
  (Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) ≤ a + b + c) ∧
  (a + b + c ≤ 2 * Real.sqrt (a * b) + 2 * Real.sqrt (b * c) + 2 * Real.sqrt (c * a)) :=
by sorry

end triangle_inequality_and_sqrt_sides_l2507_250770


namespace team_win_percentage_l2507_250727

theorem team_win_percentage (games_won : ℕ) (games_lost : ℕ) 
  (h : games_won / games_lost = 13 / 7) : 
  (games_won : ℚ) / (games_won + games_lost) * 100 = 65 := by
  sorry

end team_win_percentage_l2507_250727


namespace replaced_student_weight_l2507_250753

theorem replaced_student_weight
  (n : ℕ)
  (initial_total_weight : ℝ)
  (new_student_weight : ℝ)
  (average_decrease : ℝ)
  (h1 : n = 10)
  (h2 : new_student_weight = 60)
  (h3 : average_decrease = 6)
  (h4 : initial_total_weight - (initial_total_weight / n) + (new_student_weight / n) = initial_total_weight - n * average_decrease) :
  initial_total_weight / n - new_student_weight + n * average_decrease = 120 := by
sorry

end replaced_student_weight_l2507_250753


namespace entertainment_budget_percentage_l2507_250749

/-- Proves that given a budget of $1000, with 30% spent on food, 15% on accommodation,
    $300 on coursework materials, the remaining percentage spent on entertainment is 25%. -/
theorem entertainment_budget_percentage
  (total_budget : ℝ)
  (food_percentage : ℝ)
  (accommodation_percentage : ℝ)
  (coursework_materials : ℝ)
  (h1 : total_budget = 1000)
  (h2 : food_percentage = 30)
  (h3 : accommodation_percentage = 15)
  (h4 : coursework_materials = 300) :
  (total_budget - (food_percentage / 100 * total_budget + 
   accommodation_percentage / 100 * total_budget + coursework_materials)) / 
   total_budget * 100 = 25 := by
  sorry

#check entertainment_budget_percentage

end entertainment_budget_percentage_l2507_250749


namespace combined_tax_rate_calculation_l2507_250709

def john_tax_rate : ℚ := 30 / 100
def ingrid_tax_rate : ℚ := 40 / 100
def alice_tax_rate : ℚ := 25 / 100
def ben_tax_rate : ℚ := 35 / 100

def john_income : ℕ := 56000
def ingrid_income : ℕ := 74000
def alice_income : ℕ := 62000
def ben_income : ℕ := 80000

def total_tax : ℚ := john_tax_rate * john_income + ingrid_tax_rate * ingrid_income + 
                     alice_tax_rate * alice_income + ben_tax_rate * ben_income

def total_income : ℕ := john_income + ingrid_income + alice_income + ben_income

def combined_tax_rate : ℚ := total_tax / total_income

theorem combined_tax_rate_calculation : 
  combined_tax_rate = total_tax / total_income :=
by sorry

end combined_tax_rate_calculation_l2507_250709


namespace circle_point_x_coordinate_l2507_250751

theorem circle_point_x_coordinate :
  ∀ x : ℝ,
  let circle_center : ℝ × ℝ := (7, 0)
  let circle_radius : ℝ := 14
  let point_on_circle : ℝ × ℝ := (x, 10)
  (point_on_circle.1 - circle_center.1)^2 + (point_on_circle.2 - circle_center.2)^2 = circle_radius^2 →
  x = 7 + 4 * Real.sqrt 6 ∨ x = 7 - 4 * Real.sqrt 6 :=
by
  sorry

end circle_point_x_coordinate_l2507_250751


namespace percentage_greater_than_l2507_250763

theorem percentage_greater_than (X Y Z : ℝ) : 
  (X - Y) / (Y + Z) * 100 = 100 * (X - Y) / (Y + Z) :=
by sorry

end percentage_greater_than_l2507_250763


namespace books_taken_out_on_friday_l2507_250724

theorem books_taken_out_on_friday 
  (initial_books : ℕ) 
  (taken_out_tuesday : ℕ) 
  (brought_back_thursday : ℕ) 
  (final_books : ℕ) 
  (h1 : initial_books = 235)
  (h2 : taken_out_tuesday = 227)
  (h3 : brought_back_thursday = 56)
  (h4 : final_books = 29) :
  initial_books - taken_out_tuesday + brought_back_thursday - final_books = 35 :=
by sorry

end books_taken_out_on_friday_l2507_250724


namespace sixth_sampled_item_is_101_l2507_250771

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  totalItems : ℕ
  sampleSize : ℕ
  startNumber : ℕ

/-- Calculates the nth sampled item number in a systematic sampling -/
def nthSampledItem (s : SystematicSampling) (n : ℕ) : ℕ :=
  s.startNumber + (s.totalItems / s.sampleSize) * (n - 1)

/-- The main theorem to prove -/
theorem sixth_sampled_item_is_101 :
  let s : SystematicSampling := {
    totalItems := 1000,
    sampleSize := 50,
    startNumber := 1
  }
  nthSampledItem s 6 = 101 := by sorry

end sixth_sampled_item_is_101_l2507_250771


namespace digit_2009_is_zero_l2507_250720

/-- The function that returns the nth digit in the sequence formed by 
    writing successive natural numbers without spaces -/
def nthDigit (n : ℕ) : ℕ :=
  sorry

/-- The theorem stating that the 2009th digit in the sequence is 0 -/
theorem digit_2009_is_zero : nthDigit 2009 = 0 := by
  sorry

end digit_2009_is_zero_l2507_250720


namespace necessary_but_not_sufficient_condition_l2507_250721

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (x^2 - x < 0) → (-1 < x ∧ x < 1) ∧
  ∃ y : ℝ, -1 < y ∧ y < 1 ∧ ¬(y^2 - y < 0) :=
by sorry

end necessary_but_not_sufficient_condition_l2507_250721


namespace average_marks_chem_math_l2507_250734

/-- Given that the total marks in physics, chemistry, and mathematics is 140 more than
    the marks in physics, prove that the average mark in chemistry and mathematics is 70. -/
theorem average_marks_chem_math (P C M : ℕ) (h : P + C + M = P + 140) :
  (C + M) / 2 = 70 := by
  sorry

end average_marks_chem_math_l2507_250734


namespace contribution_is_180_l2507_250752

/-- Calculates the individual contribution for painting a wall --/
def calculate_contribution (paint_cost_per_gallon : ℚ) (coverage_per_gallon : ℚ) (total_area : ℚ) (num_coats : ℕ) : ℚ :=
  let total_gallons := (total_area / coverage_per_gallon) * num_coats
  let total_cost := total_gallons * paint_cost_per_gallon
  total_cost / 2

/-- Proves that each person's contribution is $180 --/
theorem contribution_is_180 :
  calculate_contribution 45 400 1600 2 = 180 := by
  sorry

end contribution_is_180_l2507_250752


namespace jaya_rank_from_bottom_l2507_250758

theorem jaya_rank_from_bottom (total_students : ℕ) (rank_from_top : ℕ) (rank_from_bottom : ℕ) : 
  total_students = 53 → 
  rank_from_top = 5 → 
  rank_from_bottom = total_students - rank_from_top + 1 →
  rank_from_bottom = 50 := by
sorry

end jaya_rank_from_bottom_l2507_250758


namespace range_of_function_l2507_250705

open Real

theorem range_of_function (x : ℝ) (h : 0 < x ∧ x < π/2) :
  let y := sin x - 2 * cos x + 32 / (125 * sin x * (1 - cos x))
  ∀ z, y ≥ z → z ≥ 2/5 := by sorry

end range_of_function_l2507_250705


namespace inverse_function_problem_l2507_250746

theorem inverse_function_problem (f : ℝ → ℝ) (hf : Function.Bijective f) :
  f 6 = 5 → f 5 = 1 → f 1 = 4 →
  (Function.invFun f) ((Function.invFun f 5) * (Function.invFun f 4)) = 6 := by
  sorry

end inverse_function_problem_l2507_250746


namespace elevator_weight_problem_l2507_250725

/-- Given 6 people in an elevator with an average weight of 154 lbs, 
    if a 7th person enters and the new average weight becomes 151 lbs, 
    then the weight of the 7th person is 133 lbs. -/
theorem elevator_weight_problem (initial_people : ℕ) (initial_avg_weight : ℝ) 
    (final_people : ℕ) (final_avg_weight : ℝ) : 
    initial_people = 6 → 
    initial_avg_weight = 154 → 
    final_people = 7 → 
    final_avg_weight = 151 → 
    (initial_people * initial_avg_weight + 
      (final_people - initial_people) * 
      ((final_people * final_avg_weight) - (initial_people * initial_avg_weight))) / 
      (final_people - initial_people) = 133 := by
  sorry

end elevator_weight_problem_l2507_250725


namespace taxi_fare_calculation_l2507_250745

/-- Represents the taxi fare structure -/
structure TaxiFare where
  startupFee : ℝ
  ratePerMile : ℝ

/-- Calculates the total fare for a given distance -/
def totalFare (tf : TaxiFare) (distance : ℝ) : ℝ :=
  tf.startupFee + tf.ratePerMile * distance

theorem taxi_fare_calculation (tf : TaxiFare) :
  tf.startupFee = 30 ∧ totalFare tf 60 = 150 → totalFare tf 90 = 210 := by
  sorry

end taxi_fare_calculation_l2507_250745


namespace scout_sunday_deliveries_l2507_250777

def base_pay : ℝ := 10
def tip_per_customer : ℝ := 5
def saturday_hours : ℝ := 4
def saturday_customers : ℝ := 5
def sunday_hours : ℝ := 5
def total_earnings : ℝ := 155

theorem scout_sunday_deliveries :
  ∃ (sunday_customers : ℝ),
    base_pay * (saturday_hours + sunday_hours) +
    tip_per_customer * (saturday_customers + sunday_customers) = total_earnings ∧
    sunday_customers = 8 := by
  sorry

end scout_sunday_deliveries_l2507_250777


namespace shaded_area_proof_l2507_250767

theorem shaded_area_proof (rectangle_length rectangle_width : ℝ)
  (triangle_a_leg1 triangle_a_leg2 : ℝ)
  (triangle_b_leg1 triangle_b_leg2 : ℝ)
  (h1 : rectangle_length = 14)
  (h2 : rectangle_width = 7)
  (h3 : triangle_a_leg1 = 8)
  (h4 : triangle_a_leg2 = 5)
  (h5 : triangle_b_leg1 = 6)
  (h6 : triangle_b_leg2 = 2) :
  rectangle_length * rectangle_width - 3 * ((1/2 * triangle_a_leg1 * triangle_a_leg2) + (1/2 * triangle_b_leg1 * triangle_b_leg2)) = 20 := by
  sorry

end shaded_area_proof_l2507_250767


namespace marble_selection_theorem_l2507_250700

def total_marbles : ℕ := 15
def red_marbles : ℕ := 1
def green_marbles : ℕ := 1
def blue_marbles : ℕ := 1
def yellow_marbles : ℕ := 1
def other_marbles : ℕ := 11
def marbles_to_choose : ℕ := 5

def choose_marbles (n k : ℕ) : ℕ := Nat.choose n k

theorem marble_selection_theorem :
  (choose_marbles 3 1 * choose_marbles 11 4) +
  (choose_marbles 3 2 * choose_marbles 11 3) +
  (choose_marbles 3 3 * choose_marbles 11 2) = 1540 := by
  sorry

#check marble_selection_theorem

end marble_selection_theorem_l2507_250700


namespace smallest_even_abundant_l2507_250739

/-- A number is abundant if the sum of its proper divisors is greater than the number itself. -/
def is_abundant (n : ℕ) : Prop :=
  (Finset.filter (· < n) (Finset.range n)).sum (λ i => if n % i = 0 then i else 0) > n

/-- A number is even if it's divisible by 2. -/
def is_even (n : ℕ) : Prop := n % 2 = 0

theorem smallest_even_abundant : ∀ n : ℕ, is_even n → is_abundant n → n ≥ 12 :=
sorry

end smallest_even_abundant_l2507_250739


namespace matrix_is_own_inverse_l2507_250715

/-- A matrix is its own inverse if and only if its square is the identity matrix. -/
theorem matrix_is_own_inverse (c d : ℚ) : 
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![4, -2; c, d]
  A * A = 1 ↔ c = 15/2 ∧ d = -4 := by
  sorry

end matrix_is_own_inverse_l2507_250715


namespace problem_statement_l2507_250738

theorem problem_statement (a : ℝ) (h : a/2 - 2/a = 5) : 
  (a^8 - 256) / (16 * a^4) * (2 * a) / (a^2 + 4) = 81 := by
  sorry

end problem_statement_l2507_250738


namespace symmetric_point_about_origin_l2507_250710

/-- Given a point P(-1, 2) in a rectangular coordinate system,
    its symmetric point about the origin has coordinates (1, -2). -/
theorem symmetric_point_about_origin :
  let P : ℝ × ℝ := (-1, 2)
  (- P.1, - P.2) = (1, -2) := by
  sorry

end symmetric_point_about_origin_l2507_250710


namespace least_value_quadratic_l2507_250701

theorem least_value_quadratic (x : ℝ) : 
  (∀ y : ℝ, 4 * y^2 + 8 * y + 3 = 1 → y ≥ -1) ∧ 
  (4 * (-1)^2 + 8 * (-1) + 3 = 1) := by
  sorry

end least_value_quadratic_l2507_250701


namespace edward_games_from_friend_l2507_250735

/-- The number of games Edward bought from his friend -/
def games_from_friend : ℕ := sorry

/-- The number of games Edward bought at the garage sale -/
def games_from_garage_sale : ℕ := 14

/-- The number of games that didn't work -/
def non_working_games : ℕ := 31

/-- The number of good games Edward ended up with -/
def good_games : ℕ := 24

theorem edward_games_from_friend :
  games_from_friend = 41 :=
by
  have h1 : games_from_friend + games_from_garage_sale - non_working_games = good_games := by sorry
  sorry

end edward_games_from_friend_l2507_250735


namespace nine_nine_nine_squared_plus_nine_nine_nine_l2507_250733

theorem nine_nine_nine_squared_plus_nine_nine_nine (n : ℕ) : 999 * 999 + 999 = 999000 := by
  sorry

end nine_nine_nine_squared_plus_nine_nine_nine_l2507_250733


namespace door_can_be_opened_l2507_250732

/-- Represents a device with toggle switches and a display -/
structure Device where
  combinations : Fin 32 → ℕ

/-- Represents the notebook used for communication -/
structure Notebook where
  pages : Fin 1001 → Option (Fin 32)

/-- Represents the state of the operation -/
structure OperationState where
  deviceA : Device
  deviceB : Device
  notebook : Notebook
  time : ℕ

/-- Checks if a matching combination is found -/
def isMatchingCombinationFound (state : OperationState) : Prop :=
  ∃ (i : Fin 32), state.deviceA.combinations i = state.deviceB.combinations i

/-- Defines the time constraints of the operation -/
def isWithinTimeConstraint (state : OperationState) : Prop :=
  state.time ≤ 75

/-- Theorem stating that a matching combination can be found within the time constraint -/
theorem door_can_be_opened (initialState : OperationState) :
  ∃ (finalState : OperationState),
    isMatchingCombinationFound finalState ∧
    isWithinTimeConstraint finalState :=
  sorry


end door_can_be_opened_l2507_250732


namespace initial_plus_bought_equals_total_l2507_250778

/-- The number of bottle caps William had initially -/
def initial_caps : ℕ := 2

/-- The number of bottle caps William bought -/
def bought_caps : ℕ := 41

/-- The total number of bottle caps William has after buying more -/
def total_caps : ℕ := 43

/-- Theorem stating that the initial number of bottle caps plus the bought ones equals the total -/
theorem initial_plus_bought_equals_total : 
  initial_caps + bought_caps = total_caps := by sorry

end initial_plus_bought_equals_total_l2507_250778


namespace common_factor_of_polynomials_l2507_250754

theorem common_factor_of_polynomials (m : ℝ) : 
  ∃ (k₁ k₂ k₃ : ℝ → ℝ), 
    (m * (m - 3) + 2 * (3 - m) = (m - 2) * k₁ m) ∧
    (m^2 - 4*m + 4 = (m - 2) * k₂ m) ∧
    (m^4 - 16 = (m - 2) * k₃ m) := by
  sorry

end common_factor_of_polynomials_l2507_250754


namespace lucky_money_distribution_l2507_250742

/-- Represents a distribution of lucky money among three grandsons -/
structure LuckyMoneyDistribution where
  grandson1 : ℕ
  grandson2 : ℕ
  grandson3 : ℕ

/-- Checks if a distribution satisfies the given conditions -/
def isValidDistribution (d : LuckyMoneyDistribution) : Prop :=
  ∃ (x y z : ℕ),
    (d.grandson1 = 10 * x ∧ d.grandson2 = 20 * y ∧ d.grandson3 = 50 * z) ∧
    (x = y * z) ∧
    (d.grandson1 + d.grandson2 + d.grandson3 = 300)

/-- The theorem stating the only valid distributions -/
theorem lucky_money_distribution :
  ∀ d : LuckyMoneyDistribution,
    isValidDistribution d →
    (d = ⟨100, 100, 100⟩ ∨ d = ⟨90, 60, 150⟩) :=
by sorry


end lucky_money_distribution_l2507_250742


namespace permutations_of_eight_distinct_objects_l2507_250761

theorem permutations_of_eight_distinct_objects : Nat.factorial 8 = 40320 := by
  sorry

end permutations_of_eight_distinct_objects_l2507_250761


namespace ant_movement_probability_l2507_250791

/-- A point in a 3D cubic lattice grid -/
structure Point3D where
  x : ℤ
  y : ℤ
  z : ℤ

/-- The number of adjacent points in a 3D cubic lattice -/
def adjacent_points : ℕ := 6

/-- The number of steps the ant takes -/
def num_steps : ℕ := 4

/-- The probability of moving to a specific adjacent point in one step -/
def step_probability : ℚ := 1 / adjacent_points

/-- 
  Theorem: The probability of an ant moving from point A to point B 
  (directly one floor above A) on a cubic lattice grid in exactly four steps, 
  where each step is to an adjacent point with equal probability, is 1/1296.
-/
theorem ant_movement_probability (A B : Point3D) 
  (h1 : B.x = A.x ∧ B.y = A.y ∧ B.z = A.z + 1) : 
  step_probability ^ num_steps = 1 / 1296 := by
  sorry

end ant_movement_probability_l2507_250791


namespace quadratic_polynomial_value_l2507_250718

/-- A quadratic polynomial -/
def QuadraticPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- Divisibility condition for the polynomial -/
def DivisibilityCondition (q : ℝ → ℝ) : Prop :=
  ∃ p : ℝ → ℝ, ∀ x : ℝ, q x^3 + x = p x * (x - 2) * (x + 2) * (x - 5)

theorem quadratic_polynomial_value (a b c : ℝ) :
  let q := QuadraticPolynomial a b c
  DivisibilityCondition q → q 10 = -139 * Real.rpow 2 (1/3) := by
  sorry

end quadratic_polynomial_value_l2507_250718


namespace point_in_region_l2507_250741

-- Define the point P
def P (a : ℝ) : ℝ × ℝ := (a, a + 1)

-- Define the inequality that represents the region
def in_region (x y a : ℝ) : Prop := x + a * y - 3 > 0

-- Theorem statement
theorem point_in_region (a : ℝ) :
  in_region (P a).1 (P a).2 a ↔ a < -3 ∨ a > 1 :=
sorry

end point_in_region_l2507_250741


namespace smallest_three_star_number_three_star_common_divisor_with_30_l2507_250773

/-- A three-star number is a three-digit positive integer that is the product of three distinct prime numbers. -/
def IsThreeStarNumber (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ ∃ p q r : ℕ, Prime p ∧ Prime q ∧ Prime r ∧ p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ n = p * q * r

/-- The smallest three-star number is 102. -/
theorem smallest_three_star_number : 
  IsThreeStarNumber 102 ∧ ∀ n, IsThreeStarNumber n → 102 ≤ n :=
sorry

/-- Every three-star number has a common divisor with 30 greater than 1. -/
theorem three_star_common_divisor_with_30 (n : ℕ) (h : IsThreeStarNumber n) : 
  ∃ d : ℕ, d > 1 ∧ d ∣ n ∧ d ∣ 30 :=
sorry

end smallest_three_star_number_three_star_common_divisor_with_30_l2507_250773


namespace new_train_distance_l2507_250755

theorem new_train_distance (old_distance : ℝ) (increase_percentage : ℝ) (new_distance : ℝ) : 
  old_distance = 180 → 
  increase_percentage = 0.5 → 
  new_distance = old_distance * (1 + increase_percentage) → 
  new_distance = 270 := by
sorry

end new_train_distance_l2507_250755


namespace greatest_divisor_with_remainders_l2507_250744

theorem greatest_divisor_with_remainders : ∃ (n : ℕ), 
  n > 0 ∧
  (∀ m : ℕ, m > 0 ∧ 
    (3815 % m = 31 ∧ 4521 % m = 33) → 
    m ≤ n) ∧
  3815 % n = 31 ∧ 
  4521 % n = 33 ∧
  n = 64 := by
sorry

end greatest_divisor_with_remainders_l2507_250744


namespace intersection_point_of_function_and_inverse_l2507_250766

theorem intersection_point_of_function_and_inverse (b a : ℤ) : 
  let f : ℝ → ℝ := λ x ↦ -2 * x + b
  let f_inv : ℝ → ℝ := Function.invFun f
  (∀ x, f (f_inv x) = x) ∧ (∀ x, f_inv (f x) = x) ∧ f 2 = a ∧ f_inv 2 = a
  → a = 2 := by
  sorry

end intersection_point_of_function_and_inverse_l2507_250766


namespace gift_cost_increase_l2507_250726

theorem gift_cost_increase (initial_friends : ℕ) (gift_cost : ℕ) (dropouts : ℕ) : 
  initial_friends = 10 → 
  gift_cost = 120 → 
  dropouts = 4 → 
  (gift_cost / (initial_friends - dropouts) : ℚ) - (gift_cost / initial_friends : ℚ) = 8 := by
  sorry

end gift_cost_increase_l2507_250726


namespace function_equation_solution_l2507_250736

/-- Given a function f : ℝ → ℝ satisfying the equation
    f(x) + f(2x+y) + 7xy = f(3x - 2y) + 3x^2 + 2
    for all real numbers x and y, prove that f(15) = 1202 -/
theorem function_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f x + f (2*x + y) + 7*x*y = f (3*x - 2*y) + 3*x^2 + 2) : 
  f 15 = 1202 := by
  sorry

end function_equation_solution_l2507_250736


namespace quadratic_function_properties_l2507_250730

/-- A quadratic function f(x) with leading coefficient a -/
def f (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- The condition that f(x) > -2x for x ∈ (1,3) -/
def condition_solution_set (a b c : ℝ) : Prop :=
  ∀ x, 1 < x ∧ x < 3 → f a b c x > -2 * x

/-- The condition that f(x) + 6a = 0 has two equal roots -/
def condition_equal_roots (a b c : ℝ) : Prop :=
  ∃ x : ℝ, f a b c x + 6 * a = 0 ∧
    ∀ y : ℝ, f a b c y + 6 * a = 0 → y = x

/-- The condition that the maximum value of f(x) is positive -/
def condition_positive_max (a b c : ℝ) : Prop :=
  ∃ m : ℝ, m > 0 ∧ ∀ x : ℝ, f a b c x ≤ m

theorem quadratic_function_properties
  (a b c : ℝ)
  (ha : a < 0)
  (hf : condition_solution_set a b c) :
  (condition_equal_roots a b c →
    f a b c = fun x ↦ -x^2 - x - 3/5) ∧
  (condition_positive_max a b c →
    a < -2 - Real.sqrt 5 ∨ (-2 + Real.sqrt 5 < a ∧ a < 0)) :=
sorry

end quadratic_function_properties_l2507_250730


namespace mikey_new_leaves_l2507_250707

/-- The number of new leaves that came to Mikey -/
def new_leaves (initial final : ℝ) : ℝ := final - initial

/-- Proof that Mikey received 112 new leaves -/
theorem mikey_new_leaves :
  let initial : ℝ := 356.0
  let final : ℝ := 468
  new_leaves initial final = 112 := by sorry

end mikey_new_leaves_l2507_250707


namespace largest_multiple_of_seven_under_hundred_l2507_250792

theorem largest_multiple_of_seven_under_hundred : 
  ∃ (n : ℕ), n = 98 ∧ 
  7 ∣ n ∧ 
  n < 100 ∧ 
  ∀ (m : ℕ), 7 ∣ m → m < 100 → m ≤ n :=
by sorry

end largest_multiple_of_seven_under_hundred_l2507_250792


namespace car_average_speed_l2507_250772

/-- The average speed of a car traveling 60 km in the first hour and 30 km in the second hour is 45 km/h. -/
theorem car_average_speed : 
  let speed1 : ℝ := 60 -- Speed in the first hour (km/h)
  let speed2 : ℝ := 30 -- Speed in the second hour (km/h)
  let time : ℝ := 2 -- Total time (hours)
  let total_distance : ℝ := speed1 + speed2 -- Total distance (km)
  let average_speed : ℝ := total_distance / time -- Average speed (km/h)
  average_speed = 45 := by
  sorry

end car_average_speed_l2507_250772


namespace marks_candies_l2507_250785

-- Define the number of people
def num_people : ℕ := 3

-- Define the number of candies each person will have after sharing
def shared_candies : ℕ := 30

-- Define Peter's candies
def peter_candies : ℕ := 25

-- Define John's candies
def john_candies : ℕ := 35

-- Theorem to prove Mark's candies
theorem marks_candies :
  shared_candies * num_people - (peter_candies + john_candies) = 30 := by
  sorry


end marks_candies_l2507_250785


namespace parallelogram_base_l2507_250780

/-- Given a parallelogram with area 78.88 cm² and height 8 cm, its base is 9.86 cm -/
theorem parallelogram_base (area : ℝ) (height : ℝ) (base : ℝ) : 
  area = 78.88 ∧ height = 8 ∧ area = base * height → base = 9.86 := by
  sorry

end parallelogram_base_l2507_250780


namespace urn_problem_solution_l2507_250779

/-- The number of blue balls in the second urn -/
def N : ℕ := 144

/-- The probability of drawing two balls of the same color -/
def same_color_probability : ℚ := 29/50

theorem urn_problem_solution :
  let urn1_green : ℕ := 4
  let urn1_blue : ℕ := 6
  let urn2_green : ℕ := 16
  let urn1_total : ℕ := urn1_green + urn1_blue
  let urn2_total : ℕ := urn2_green + N
  let same_green : ℕ := urn1_green * urn2_green
  let same_blue : ℕ := urn1_blue * N
  let total_outcomes : ℕ := urn1_total * urn2_total
  (same_green + same_blue : ℚ) / total_outcomes = same_color_probability :=
by sorry

end urn_problem_solution_l2507_250779


namespace math_textbooks_in_one_box_l2507_250706

def total_textbooks : ℕ := 15
def math_textbooks : ℕ := 4
def boxes : ℕ := 3
def books_per_box : ℕ := 5

def probability_all_math_in_one_box : ℚ := 769 / 100947

theorem math_textbooks_in_one_box :
  let total_ways := (total_textbooks.choose books_per_box) * 
                    ((total_textbooks - books_per_box).choose books_per_box) * 
                    ((total_textbooks - 2 * books_per_box).choose books_per_box)
  let favorable_ways := boxes * 
                        ((total_textbooks - math_textbooks).choose 1) * 
                        ((total_textbooks - math_textbooks - 1).choose books_per_box) * 
                        ((total_textbooks - math_textbooks - 1 - books_per_box).choose books_per_box)
  (favorable_ways : ℚ) / total_ways = probability_all_math_in_one_box := by
  sorry

end math_textbooks_in_one_box_l2507_250706


namespace factors_of_72_l2507_250702

theorem factors_of_72 : Finset.card (Nat.divisors 72) = 12 := by sorry

end factors_of_72_l2507_250702


namespace probability_second_white_given_first_white_l2507_250795

/-- The probability of drawing a white ball second, given that the first ball drawn is white,
    when there are 5 white balls and 4 black balls initially. -/
theorem probability_second_white_given_first_white :
  let total_balls : ℕ := 9
  let white_balls : ℕ := 5
  let black_balls : ℕ := 4
  let prob_first_white : ℚ := white_balls / total_balls
  let prob_both_white : ℚ := (white_balls * (white_balls - 1)) / (total_balls * (total_balls - 1))
  prob_both_white / prob_first_white = 1 / 2 := by sorry

end probability_second_white_given_first_white_l2507_250795


namespace two_color_theorem_l2507_250723

/-- A type representing a region in the plane --/
def Region : Type := ℕ

/-- A type representing a color (either 0 or 1) --/
def Color : Type := Fin 2

/-- A function that determines if two regions are adjacent --/
def adjacent (n : ℕ) (r1 r2 : Region) : Prop := sorry

/-- A coloring of regions --/
def Coloring (n : ℕ) : Type := Region → Color

/-- A predicate that determines if a coloring is valid --/
def is_valid_coloring (n : ℕ) (c : Coloring n) : Prop :=
  ∀ r1 r2 : Region, adjacent n r1 r2 → c r1 ≠ c r2

/-- The main theorem: there exists a valid two-coloring for any number of circles --/
theorem two_color_theorem (n : ℕ) (h : n ≥ 1) :
  ∃ c : Coloring n, is_valid_coloring n c :=
sorry

end two_color_theorem_l2507_250723


namespace complex_product_theorem_l2507_250748

theorem complex_product_theorem : 
  let z1 : ℂ := -1 + 2*I
  let z2 : ℂ := 2 + I
  z1 * z2 = -4 + 3*I := by
  sorry

end complex_product_theorem_l2507_250748


namespace rectangle_width_l2507_250708

theorem rectangle_width (w : ℝ) (h1 : w > 0) (h2 : 4 * w * w = 100) : w = 5 := by
  sorry

end rectangle_width_l2507_250708


namespace card_probability_theorem_probability_after_10_shuffles_value_l2507_250764

/-- The probability that card 6 is higher than card 3 after n shuffles -/
def p (n : ℕ) : ℚ :=
  (3^n - 2^n) / (2 * 3^n)

/-- The recurrence relation for the probability -/
def recurrence (p_prev : ℚ) : ℚ :=
  (4 * p_prev + 1) / 6

theorem card_probability_theorem (n : ℕ) :
  p n = recurrence (p (n - 1)) ∧ p 0 = 0 :=
by sorry

/-- The probability that card 6 is higher than card 3 after 10 shuffles -/
def probability_after_10_shuffles : ℚ := p 10

theorem probability_after_10_shuffles_value :
  probability_after_10_shuffles = (3^10 - 2^10) / (2 * 3^10) :=
by sorry

end card_probability_theorem_probability_after_10_shuffles_value_l2507_250764


namespace grades_theorem_l2507_250796

structure Student :=
  (name : String)
  (gotA : Prop)

def Emily : Student := ⟨"Emily", true⟩
def Fran : Student := ⟨"Fran", true⟩
def George : Student := ⟨"George", true⟩
def Hailey : Student := ⟨"Hailey", false⟩

theorem grades_theorem :
  (Emily.gotA → Fran.gotA) ∧
  (Fran.gotA → George.gotA) ∧
  (George.gotA → ¬Hailey.gotA) ∧
  (Emily.gotA ∧ Fran.gotA ∧ George.gotA ∧ ¬Hailey.gotA) ∧
  (∃! (s : Finset Student), s.card = 3 ∧ ∀ student ∈ s, student.gotA) →
  ∃! (s : Finset Student),
    s.card = 3 ∧
    Emily ∈ s ∧ Fran ∈ s ∧ George ∈ s ∧ Hailey ∉ s ∧
    (∀ student ∈ s, student.gotA) :=
by sorry

end grades_theorem_l2507_250796


namespace polynomial_remainder_l2507_250759

theorem polynomial_remainder (x : ℝ) : 
  (x^3 - 3*x^2 + 5) % (x - 1) = 3 := by
sorry

end polynomial_remainder_l2507_250759


namespace total_people_after_hour_l2507_250743

/-- Represents the number of people in each line at the fair -/
structure FairLines where
  ferrisWheel : ℕ
  bumperCars : ℕ
  rollerCoaster : ℕ

/-- Calculates the total number of people across all lines after an hour -/
def totalPeopleAfterHour (initial : FairLines) (x y Z : ℕ) : ℕ :=
  Z + initial.rollerCoaster * 2

/-- Theorem stating the total number of people after an hour -/
theorem total_people_after_hour 
  (initial : FairLines)
  (x y Z : ℕ)
  (h1 : initial.ferrisWheel = 50)
  (h2 : initial.bumperCars = 50)
  (h3 : initial.rollerCoaster = 50)
  (h4 : Z = (50 - x) + (50 + y)) :
  totalPeopleAfterHour initial x y Z = Z + 100 := by
  sorry

#check total_people_after_hour

end total_people_after_hour_l2507_250743


namespace correct_operation_l2507_250798

theorem correct_operation (x y : ℝ) : 2 * x^2 * (3 * x^2 - 5 * y) = 6 * x^4 - 10 * x^2 * y := by
  sorry

end correct_operation_l2507_250798


namespace no_fast_connectivity_algorithm_l2507_250713

/-- A graph with 64 vertices -/
def Graph := Fin 64 → Fin 64 → Bool

/-- Number of queries required -/
def required_queries : ℕ := 2016

/-- An algorithm that determines graph connectivity -/
def ConnectivityAlgorithm := Graph → Bool

/-- The number of queries an algorithm makes -/
def num_queries (alg : ConnectivityAlgorithm) : ℕ := sorry

/-- A graph is connected -/
def is_connected (g : Graph) : Prop := sorry

/-- Theorem: No algorithm can determine connectivity in fewer than 2016 queries -/
theorem no_fast_connectivity_algorithm :
  ¬∃ (alg : ConnectivityAlgorithm),
    (∀ g : Graph, alg g = is_connected g) ∧
    (num_queries alg < required_queries) :=
sorry

end no_fast_connectivity_algorithm_l2507_250713


namespace min_value_x_plus_2y_l2507_250719

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 8 / x + 1 / y = 1) :
  x + 2 * y ≥ 18 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 8 / x₀ + 1 / y₀ = 1 ∧ x₀ + 2 * y₀ = 18 :=
sorry

end min_value_x_plus_2y_l2507_250719


namespace puzzle_solution_l2507_250784

theorem puzzle_solution (p q r s t : ℕ+) 
  (eq1 : p * q + p + q = 322)
  (eq2 : q * r + q + r = 186)
  (eq3 : r * s + r + s = 154)
  (eq4 : s * t + s + t = 272)
  (product : p * q * r * s * t = 3628800) : -- 3628800 is 10!
  p - t = 6 := by
  sorry

end puzzle_solution_l2507_250784


namespace parallelogram_height_l2507_250797

/-- The height of a parallelogram with given area and base -/
theorem parallelogram_height (area : ℝ) (base : ℝ) (h_area : area = 288) (h_base : base = 18) :
  area / base = 16 := by
  sorry

end parallelogram_height_l2507_250797


namespace perfect_square_polynomial_l2507_250794

theorem perfect_square_polynomial (a b : ℝ) : 
  (∃ p q r : ℝ, ∀ x : ℝ, x^4 - x^3 + x^2 + a*x + b = (p*x^2 + q*x + r)^2) → 
  b = 9/64 := by
  sorry

end perfect_square_polynomial_l2507_250794


namespace marie_sold_925_reading_materials_l2507_250722

/-- The total number of reading materials Marie sold -/
def total_reading_materials (magazines newspapers books pamphlets : ℕ) : ℕ :=
  magazines + newspapers + books + pamphlets

/-- Theorem stating that Marie sold 925 reading materials -/
theorem marie_sold_925_reading_materials :
  total_reading_materials 425 275 150 75 = 925 := by
  sorry

end marie_sold_925_reading_materials_l2507_250722


namespace circle_radius_given_area_and_circumference_sum_l2507_250783

theorem circle_radius_given_area_and_circumference_sum (x y : ℝ) :
  x ≥ 0 →
  y > 0 →
  x = π * (y / (2 * π))^2 →
  x + y = 90 * π →
  y / (2 * π) = 10 := by
sorry

end circle_radius_given_area_and_circumference_sum_l2507_250783


namespace quadratic_inequality_properties_l2507_250787

/-- Given a quadratic inequality ax^2 - bx + c > 0 with solution set (-1/2, 2), 
    prove properties about its coefficients -/
theorem quadratic_inequality_properties (a b c : ℝ) 
  (h : ∀ x, -1/2 < x ∧ x < 2 ↔ a * x^2 - b * x + c > 0) : 
  b < 0 ∧ c > 0 ∧ a - b + c > 0 ∧ a ≤ 0 ∧ a + b + c ≤ 0 := by
  sorry

end quadratic_inequality_properties_l2507_250787


namespace sum_seven_times_difference_l2507_250765

theorem sum_seven_times_difference (x y : ℝ) : 
  x > 0 → y > 0 → x > y → x - y = 3 → x + y = 7 * (x - y) → x + y = 21 := by
  sorry

end sum_seven_times_difference_l2507_250765


namespace second_solution_concentration_l2507_250714

/-- Represents an alcohol solution --/
structure AlcoholSolution where
  volume : ℝ
  concentration : ℝ

/-- Represents a mixture of two alcohol solutions --/
structure AlcoholMixture where
  solution1 : AlcoholSolution
  solution2 : AlcoholSolution
  final : AlcoholSolution

/-- The alcohol mixture satisfies the given conditions --/
def satisfies_conditions (mixture : AlcoholMixture) : Prop :=
  mixture.final.volume = 200 ∧
  mixture.final.concentration = 0.15 ∧
  mixture.solution1.volume = 75 ∧
  mixture.solution1.concentration = 0.20 ∧
  mixture.solution2.volume = mixture.final.volume - mixture.solution1.volume

theorem second_solution_concentration
  (mixture : AlcoholMixture)
  (h : satisfies_conditions mixture) :
  mixture.solution2.concentration = 0.12 := by
  sorry

end second_solution_concentration_l2507_250714


namespace flight_duration_sum_l2507_250747

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Calculates the difference between two times in minutes -/
def timeDifferenceInMinutes (t1 t2 : Time) : ℕ :=
  (t2.hours - t1.hours) * 60 + (t2.minutes - t1.minutes)

/-- Theorem: Flight duration calculation -/
theorem flight_duration_sum (departureTime : Time) (arrivalTime : Time) 
  (h m : ℕ) (hm : 0 < m ∧ m < 60) :
  departureTime.hours = 9 ∧ departureTime.minutes = 17 →
  arrivalTime.hours = 13 ∧ arrivalTime.minutes = 53 →
  timeDifferenceInMinutes departureTime arrivalTime = h * 60 + m →
  h + m = 41 := by
  sorry

#check flight_duration_sum

end flight_duration_sum_l2507_250747


namespace min_absolute_value_complex_l2507_250776

open Complex

theorem min_absolute_value_complex (z : ℂ) :
  (abs (z + I) + abs (z - 2 - I) = 2 * Real.sqrt 2) →
  (∃ (w : ℂ), abs w ≤ abs z ∧ abs (w + I) + abs (w - 2 - I) = 2 * Real.sqrt 2) →
  abs z ≥ Real.sqrt 2 / 2 :=
by sorry

end min_absolute_value_complex_l2507_250776


namespace existence_of_close_points_l2507_250760

theorem existence_of_close_points :
  ∃ (x y : ℝ), y = x^3 ∧ |y - (x^3 + |x| + 1)| ≤ 1/100 := by
  sorry

end existence_of_close_points_l2507_250760


namespace kaleb_sold_games_l2507_250775

theorem kaleb_sold_games (initial_games : ℕ) (games_per_box : ℕ) (boxes_used : ℕ) 
  (h1 : initial_games = 76)
  (h2 : games_per_box = 5)
  (h3 : boxes_used = 6) :
  initial_games - (games_per_box * boxes_used) = 46 := by
  sorry

end kaleb_sold_games_l2507_250775


namespace seven_consecutive_integers_product_divisible_by_ten_l2507_250729

theorem seven_consecutive_integers_product_divisible_by_ten (n : ℕ+) :
  ∃ k : ℕ, (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) * (n + 6) : ℕ) = 10 * k := by
  sorry

end seven_consecutive_integers_product_divisible_by_ten_l2507_250729


namespace logarithmic_equation_solution_l2507_250740

theorem logarithmic_equation_solution (x : ℝ) :
  (x > 0) →
  (5 * (Real.log x / Real.log (x / 9)) + 
   (Real.log (x^3) / Real.log (9 / x)) + 
   8 * (Real.log (x^2) / Real.log (9 * x^2)) = 2) ↔ 
  (x = 3 ∨ x = Real.sqrt 3) :=
by sorry

end logarithmic_equation_solution_l2507_250740


namespace pencils_lost_l2507_250757

theorem pencils_lost (initial_pencils : ℕ) (current_pencils : ℕ) (lost_pencils : ℕ) :
  initial_pencils = 30 →
  current_pencils = 16 →
  current_pencils = initial_pencils - lost_pencils - (initial_pencils - lost_pencils) / 3 →
  lost_pencils = 6 := by
  sorry

end pencils_lost_l2507_250757


namespace correct_division_l2507_250731

theorem correct_division (n : ℚ) : n / 22 = 2 → n / 20 = 2.2 := by
  sorry

end correct_division_l2507_250731


namespace solution_set_for_neg_eight_solution_range_for_a_l2507_250756

-- Define the inequality function
def inequality (x a : ℝ) : Prop :=
  |x - 3| + |x + 2| ≤ |a + 1|

-- Theorem 1: Solution set when a = -8
theorem solution_set_for_neg_eight :
  Set.Icc (-3 : ℝ) 4 = {x : ℝ | inequality x (-8)} :=
sorry

-- Theorem 2: Range of a for which the inequality has solutions
theorem solution_range_for_a :
  {a : ℝ | ∃ x, inequality x a} = Set.Iic (-6) ∪ Set.Ici 4 :=
sorry

end solution_set_for_neg_eight_solution_range_for_a_l2507_250756


namespace system_solution_l2507_250711

theorem system_solution (x y : ℝ) : 
  x^5 + y^5 = 1 ∧ x^6 + y^6 = 1 ↔ (x = 1 ∧ y = 0) ∨ (x = 0 ∧ y = 1) := by
  sorry

end system_solution_l2507_250711


namespace cos_double_angle_special_case_l2507_250790

theorem cos_double_angle_special_case (θ : Real) 
  (h : Real.sin (Real.pi / 2 + θ) = 1 / 3) : 
  Real.cos (2 * θ) = -7 / 9 := by
  sorry

end cos_double_angle_special_case_l2507_250790


namespace square_value_l2507_250716

/-- Given that square times 3a equals -3a^2b, prove that square equals -ab -/
theorem square_value (a b : ℝ) (square : ℝ) (h : square * 3 * a = -3 * a^2 * b) :
  square = -a * b := by sorry

end square_value_l2507_250716
