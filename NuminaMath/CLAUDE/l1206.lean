import Mathlib

namespace intersection_P_Q_l1206_120643

-- Define the sets P and Q
def P : Set ℝ := {x | x > 1}
def Q : Set ℝ := {x | x < 2}

-- State the theorem
theorem intersection_P_Q : P ∩ Q = Set.Ioo 1 2 := by
  sorry

end intersection_P_Q_l1206_120643


namespace min_x_minus_y_l1206_120612

theorem min_x_minus_y (x y : ℝ) (h1 : x > 0) (h2 : 0 > y) 
  (h3 : 1 / (x + 2) + 1 / (1 - y) = 1 / 6) : x - y ≥ 21 := by
  sorry

end min_x_minus_y_l1206_120612


namespace share_of_c_l1206_120602

/-- 
Given a total amount to be divided among three people A, B, and C,
where A gets 2/3 of what B gets, and B gets 1/4 of what C gets,
prove that the share of C is 360 when the total amount is 510.
-/
theorem share_of_c (total : ℚ) (share_a share_b share_c : ℚ) : 
  total = 510 →
  share_a = (2/3) * share_b →
  share_b = (1/4) * share_c →
  share_a + share_b + share_c = total →
  share_c = 360 := by
sorry

end share_of_c_l1206_120602


namespace three_gorges_electricity_production_l1206_120666

/-- Scientific notation representation -/
structure ScientificNotation where
  a : ℝ
  n : ℤ
  h1 : 1 ≤ |a| 
  h2 : |a| < 10

/-- The number to be represented (798.5 billion) -/
def number : ℝ := 798.5e9

/-- Theorem stating that 798.5 billion can be represented as 7.985 × 10^2 billion in scientific notation -/
theorem three_gorges_electricity_production :
  ∃ (sn : ScientificNotation), sn.a * (10 : ℝ)^sn.n = number ∧ sn.a = 7.985 ∧ sn.n = 2 :=
sorry

end three_gorges_electricity_production_l1206_120666


namespace philips_bananas_l1206_120663

theorem philips_bananas (num_groups : ℕ) (bananas_per_group : ℕ) 
  (h1 : num_groups = 11) (h2 : bananas_per_group = 37) :
  num_groups * bananas_per_group = 407 := by
  sorry

end philips_bananas_l1206_120663


namespace quiz_score_theorem_l1206_120653

theorem quiz_score_theorem :
  ∀ (correct : ℕ),
  correct ≤ 15 →
  6 * correct - 2 * (15 - correct) ≥ 75 →
  correct ≥ 14 :=
by
  sorry

end quiz_score_theorem_l1206_120653


namespace hyperbola_distance_theorem_l1206_120614

-- Define the hyperbola
def is_on_hyperbola (x y : ℝ) : Prop := x^2 / 25 - y^2 / 9 = 1

-- Define the foci
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

-- Define the distance function
def distance (p₁ p₂ : ℝ × ℝ) : ℝ := sorry

-- The main theorem
theorem hyperbola_distance_theorem (x y : ℝ) (P : ℝ × ℝ) :
  is_on_hyperbola x y →
  P = (x, y) →
  distance P F₁ = 12 →
  distance P F₂ = 2 ∨ distance P F₂ = 22 := by
  sorry

end hyperbola_distance_theorem_l1206_120614


namespace investment_value_proof_l1206_120628

theorem investment_value_proof (x : ℝ) : 
  x > 0 ∧ 
  0.07 * x + 0.23 * 1500 = 0.19 * (x + 1500) →
  x = 500 := by
sorry

end investment_value_proof_l1206_120628


namespace expression_simplification_l1206_120683

theorem expression_simplification (a b c : ℚ) 
  (ha : a = 1/3) (hb : b = 1/2) (hc : c = 1) :
  (2*a^2 - b) - (a^2 - 4*b) - (b + c) = 1/9 := by
  sorry

end expression_simplification_l1206_120683


namespace prosecutor_conclusion_l1206_120672

-- Define the types for guilt
inductive Guilt
| Guilty
| NotGuilty

-- Define the prosecutor's statements
def statement1 (X Y : Guilt) : Prop :=
  X = Guilt.NotGuilty ∨ Y = Guilt.Guilty

def statement2 (X : Guilt) : Prop :=
  X = Guilt.Guilty

-- Theorem to prove
theorem prosecutor_conclusion (X Y : Guilt) :
  statement1 X Y ∧ statement2 X →
  X = Guilt.Guilty ∧ Y = Guilt.Guilty :=
by
  sorry


end prosecutor_conclusion_l1206_120672


namespace investment_growth_approx_l1206_120678

/-- Approximates the future value of an investment with compound interest -/
def future_value (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Theorem: An investment of $1500 at 8% annual interest grows to approximately $13500 in 28 years -/
theorem investment_growth_approx :
  ∃ ε > 0, abs (future_value 1500 0.08 28 - 13500) < ε :=
by
  sorry

end investment_growth_approx_l1206_120678


namespace minimal_withdrawals_l1206_120620

/-- Represents a withdrawal strategy -/
structure WithdrawalStrategy where
  red : ℕ
  blue : ℕ
  green : ℕ
  count : ℕ

/-- Represents the package of marbles -/
structure MarblePackage where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- Checks if a withdrawal strategy is valid according to the constraints -/
def is_valid_strategy (s : WithdrawalStrategy) : Prop :=
  s.red ≤ 1 ∧ s.blue ≤ 2 ∧ s.red + s.blue + s.green ≤ 5

/-- Checks if a list of withdrawal strategies empties the package -/
def empties_package (p : MarblePackage) (strategies : List WithdrawalStrategy) : Prop :=
  strategies.foldl (fun acc s => 
    { red := acc.red - s.red * s.count
    , blue := acc.blue - s.blue * s.count
    , green := acc.green - s.green * s.count
    }) p = ⟨0, 0, 0⟩

/-- The main theorem stating the minimal number of withdrawals -/
theorem minimal_withdrawals (p : MarblePackage) 
  (h_red : p.red = 200) (h_blue : p.blue = 300) (h_green : p.green = 400) :
  ∃ (strategies : List WithdrawalStrategy),
    (∀ s ∈ strategies, is_valid_strategy s) ∧
    empties_package p strategies ∧
    (strategies.foldl (fun acc s => acc + s.count) 0 = 200) ∧
    (∀ (other_strategies : List WithdrawalStrategy),
      (∀ s ∈ other_strategies, is_valid_strategy s) →
      empties_package p other_strategies →
      strategies.foldl (fun acc s => acc + s.count) 0 ≤ 
      other_strategies.foldl (fun acc s => acc + s.count) 0) :=
sorry

end minimal_withdrawals_l1206_120620


namespace ball_throw_circle_l1206_120616

/-- Given a circular arrangement of 15 elements, prove that starting from
    element 1 and moving with a step of 5 (modulo 15), it takes exactly 3
    steps to return to element 1. -/
theorem ball_throw_circle (n : ℕ) (h : n = 15) :
  let f : ℕ → ℕ := λ x => (x + 5) % n
  ∃ k : ℕ, k > 0 ∧ (f^[k] 1 = 1) ∧ ∀ m : ℕ, 0 < m → m < k → f^[m] 1 ≠ 1 ∧ k = 3 :=
by sorry

end ball_throw_circle_l1206_120616


namespace singer_tip_percentage_l1206_120649

/-- Proves that the tip percentage is 20% given the conditions of the problem -/
theorem singer_tip_percentage (hours : ℕ) (hourly_rate : ℚ) (total_paid : ℚ) :
  hours = 3 →
  hourly_rate = 15 →
  total_paid = 54 →
  (total_paid - hours * hourly_rate) / (hours * hourly_rate) * 100 = 20 := by
  sorry

end singer_tip_percentage_l1206_120649


namespace juice_left_in_cup_l1206_120629

theorem juice_left_in_cup (consumed : Rat) (h : consumed = 4/6) :
  1 - consumed = 2/6 ∨ 1 - consumed = 1/3 := by
  sorry

end juice_left_in_cup_l1206_120629


namespace sum_g_79_l1206_120679

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x^2 + 3 * x - 1
def g (y : ℝ) : ℝ := y^2 - 2 * y + 2

-- Define the equation f(x) = 79
def f_eq_79 (x : ℝ) : Prop := f x = 79

-- Theorem statement
theorem sum_g_79 (x₁ x₂ : ℝ) (h₁ : f_eq_79 x₁) (h₂ : f_eq_79 x₂) (h₃ : x₁ ≠ x₂) :
  ∃ (s : ℝ), s = g (f x₁) + g (f x₂) ∧ 
  (∀ (y : ℝ), g y = s ↔ y = 79) :=
sorry

end sum_g_79_l1206_120679


namespace rectangle_iff_equal_diagonals_l1206_120687

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define the concept of a rectangle
def isRectangle (q : Quadrilateral) : Prop := sorry

-- Define the concept of diagonal length
def diagonalLength (q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem rectangle_iff_equal_diagonals (q : Quadrilateral) :
  isRectangle q ↔ diagonalLength q = diagonalLength q := by sorry

end rectangle_iff_equal_diagonals_l1206_120687


namespace eggs_at_town_hall_l1206_120655

/-- Given the number of eggs found at different locations during an Easter egg hunt, 
    this theorem proves how many eggs were found at the town hall. -/
theorem eggs_at_town_hall 
  (total_eggs : ℕ)
  (club_house_eggs : ℕ)
  (park_eggs : ℕ)
  (h1 : total_eggs = 80)
  (h2 : club_house_eggs = 40)
  (h3 : park_eggs = 25) :
  total_eggs - (club_house_eggs + park_eggs) = 15 := by
  sorry

#check eggs_at_town_hall

end eggs_at_town_hall_l1206_120655


namespace inequality_solution_set_l1206_120657

theorem inequality_solution_set (a : ℕ) : 
  (∀ x, (a - 2) * x > a - 2 ↔ x < 1) → (a = 0 ∨ a = 1) := by
  sorry

end inequality_solution_set_l1206_120657


namespace probability_four_students_same_group_l1206_120697

theorem probability_four_students_same_group 
  (total_students : ℕ) 
  (num_groups : ℕ) 
  (h1 : total_students = 800) 
  (h2 : num_groups = 4) 
  (h3 : total_students % num_groups = 0) :
  (1 : ℚ) / (num_groups^3) = 1/64 :=
sorry

end probability_four_students_same_group_l1206_120697


namespace wine_problem_equations_l1206_120645

/-- Represents the number of guests intoxicated by one bottle of good wine -/
def good_wine_intoxication : ℚ := 3

/-- Represents the number of bottles of weak wine needed to intoxicate one guest -/
def weak_wine_intoxication : ℚ := 3

/-- Represents the total number of intoxicated guests -/
def total_intoxicated_guests : ℚ := 33

/-- Represents the total number of bottles of wine consumed -/
def total_bottles : ℚ := 19

/-- Represents the number of bottles of good wine -/
def x : ℚ := sorry

/-- Represents the number of bottles of weak wine -/
def y : ℚ := sorry

theorem wine_problem_equations :
  (x + y = total_bottles) ∧
  (good_wine_intoxication * x + (1 / weak_wine_intoxication) * y = total_intoxicated_guests) :=
by sorry

end wine_problem_equations_l1206_120645


namespace part_one_part_two_l1206_120696

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 2| - |2*x - a|

-- Part I
theorem part_one :
  {x : ℝ | f 3 x > 0} = {x : ℝ | 1/3 < x ∧ x < 5} :=
sorry

-- Part II
theorem part_two :
  ∀ a : ℝ, (∀ x : ℝ, x ≥ 0 → f a x < 3) → a < 2 :=
sorry

end part_one_part_two_l1206_120696


namespace cat_whiskers_correct_l1206_120611

structure Cat where
  name : String
  whiskers : ℕ

def princess_puff : Cat := { name := "Princess Puff", whiskers := 14 }

def catman_do : Cat := { 
  name := "Catman Do", 
  whiskers := 2 * princess_puff.whiskers - 6 
}

def sir_whiskerson : Cat := { 
  name := "Sir Whiskerson", 
  whiskers := princess_puff.whiskers + catman_do.whiskers + 8 
}

def lady_flufflepuff : Cat := { 
  name := "Lady Flufflepuff", 
  whiskers := sir_whiskerson.whiskers / 2 + 4 
}

def mr_mittens : Cat := { 
  name := "Mr. Mittens", 
  whiskers := Int.natAbs (catman_do.whiskers - lady_flufflepuff.whiskers)
}

theorem cat_whiskers_correct : 
  princess_puff.whiskers = 14 ∧ 
  catman_do.whiskers = 22 ∧ 
  sir_whiskerson.whiskers = 44 ∧ 
  lady_flufflepuff.whiskers = 26 ∧ 
  mr_mittens.whiskers = 4 := by
  sorry

end cat_whiskers_correct_l1206_120611


namespace no_integer_satisfies_inequality_l1206_120648

theorem no_integer_satisfies_inequality : 
  ¬ ∃ (n : ℤ), n > 1 ∧ (⌊Real.sqrt (n - 2) + 2 * Real.sqrt (n + 2)⌋ : ℤ) < ⌊Real.sqrt (9 * n + 6)⌋ := by
  sorry

end no_integer_satisfies_inequality_l1206_120648


namespace intersection_A_complement_B_l1206_120665

open Set Real

noncomputable def A : Set ℝ := {x | x^2 < 1}
noncomputable def B : Set ℝ := {x | x^2 - 2*x > 0}

theorem intersection_A_complement_B :
  A ∩ (𝒰 \ B) = Icc 0 1 := by sorry

end intersection_A_complement_B_l1206_120665


namespace polygon_product_symmetric_l1206_120677

/-- Represents a convex polygon in a plane -/
structure ConvexPolygon where
  -- Add necessary fields here
  
/-- Calculates the sum of products of side lengths and distances for two polygons -/
def polygonProduct (P Q : ConvexPolygon) : ℝ :=
  sorry

/-- Theorem stating that the polygon product is symmetric -/
theorem polygon_product_symmetric (P Q : ConvexPolygon) :
  polygonProduct P Q = polygonProduct Q P := by
  sorry

end polygon_product_symmetric_l1206_120677


namespace mayo_savings_l1206_120637

/-- Proves the savings when buying mayo in bulk -/
theorem mayo_savings (costco_price : ℝ) (store_price : ℝ) (gallon_oz : ℝ) (bottle_oz : ℝ) :
  costco_price = 8 →
  store_price = 3 →
  gallon_oz = 128 →
  bottle_oz = 16 →
  (gallon_oz / bottle_oz) * store_price - costco_price = 16 := by
sorry

end mayo_savings_l1206_120637


namespace product_of_sums_inequality_l1206_120642

theorem product_of_sums_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c := by
  sorry

end product_of_sums_inequality_l1206_120642


namespace total_pupils_l1206_120646

theorem total_pupils (girls : ℕ) (boys : ℕ) 
  (h1 : girls = 542) (h2 : boys = 387) : 
  girls + boys = 929 := by
  sorry

end total_pupils_l1206_120646


namespace circle_intersects_y_axis_l1206_120652

theorem circle_intersects_y_axis (D E F : ℝ) :
  (∃ y₁ y₂ : ℝ, y₁ < 0 ∧ y₂ > 0 ∧ 
    y₁^2 + E*y₁ + F = 0 ∧ 
    y₂^2 + E*y₂ + F = 0) →
  F < 0 :=
by sorry

end circle_intersects_y_axis_l1206_120652


namespace quadratic_value_at_three_l1206_120694

/-- A quadratic function with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  min_value : ℝ
  min_x : ℝ
  y_at_zero : ℝ
  h : min_value = -4
  h' : min_x = -2
  h'' : y_at_zero = 8

/-- The value of y when x = 3 for the given quadratic function -/
def y_at_three (f : QuadraticFunction) : ℝ :=
  f.a * 3^2 + f.b * 3 + f.c

/-- Theorem stating that y = 71 when x = 3 for the given quadratic function -/
theorem quadratic_value_at_three (f : QuadraticFunction) : y_at_three f = 71 := by
  sorry

end quadratic_value_at_three_l1206_120694


namespace nadia_flower_cost_l1206_120606

/-- The total cost of flowers bought by Nadia -/
def total_cost (num_roses : ℕ) (rose_price : ℚ) : ℚ :=
  let num_lilies : ℚ := (3 / 4) * num_roses
  let lily_price : ℚ := 2 * rose_price
  num_roses * rose_price + num_lilies * lily_price

/-- Theorem stating the total cost of flowers for Nadia's purchase -/
theorem nadia_flower_cost : total_cost 20 5 = 250 := by
  sorry

end nadia_flower_cost_l1206_120606


namespace complex_sum_l1206_120698

theorem complex_sum (z : ℂ) (h : z^2 + z + 1 = 0) : 
  z^101 + z^102 + z^103 + z^104 + z^105 = -2 := by
  sorry

end complex_sum_l1206_120698


namespace compare_negative_mixed_numbers_l1206_120608

theorem compare_negative_mixed_numbers :
  -6.5 > -(6 + 3/5) := by sorry

end compare_negative_mixed_numbers_l1206_120608


namespace max_projection_area_is_one_l1206_120641

/-- A tetrahedron with two adjacent isosceles right triangle faces -/
structure Tetrahedron where
  /-- The length of the hypotenuse of the isosceles right triangle faces -/
  hypotenuse : ℝ
  /-- The dihedral angle between the two adjacent isosceles right triangle faces -/
  dihedral_angle : ℝ

/-- The maximum area of the projection of a rotating tetrahedron -/
def max_projection_area (t : Tetrahedron) : ℝ := 1

/-- Theorem stating that the maximum area of the projection is 1 -/
theorem max_projection_area_is_one (t : Tetrahedron) 
  (h1 : t.hypotenuse = 2)
  (h2 : t.dihedral_angle = π / 3) : 
  max_projection_area t = 1 := by
  sorry

end max_projection_area_is_one_l1206_120641


namespace arithmetic_calculations_l1206_120667

theorem arithmetic_calculations :
  (15 + (-23) - (-10) = 2) ∧
  (-1^2 - (-2)^3 / 4 * (1/4) = -1/2) := by
  sorry

end arithmetic_calculations_l1206_120667


namespace apple_buying_difference_l1206_120682

theorem apple_buying_difference :
  ∀ (w : ℕ),
  (2 * 30 + 3 * w = 210) →
  (30 < w) →
  (w - 30 = 20) :=
by
  sorry

end apple_buying_difference_l1206_120682


namespace courtyard_length_l1206_120668

/-- Proves that the length of a rectangular courtyard is 18 meters -/
theorem courtyard_length (width : ℝ) (brick_length : ℝ) (brick_width : ℝ) (total_bricks : ℕ) :
  width = 12 →
  brick_length = 0.12 →
  brick_width = 0.06 →
  total_bricks = 30000 →
  (width * (width * total_bricks * brick_length * brick_width)⁻¹) = 18 :=
by sorry

end courtyard_length_l1206_120668


namespace golden_ratio_pentagon_l1206_120615

theorem golden_ratio_pentagon (a : ℝ) : 
  a = 2 * Real.cos (72 * π / 180) → 
  (a * Real.cos (18 * π / 180)) / Real.sqrt (2 - a) = 1 / 2 := by
sorry

end golden_ratio_pentagon_l1206_120615


namespace third_home_donation_l1206_120604

/-- Represents the donation amounts in cents to avoid floating-point issues -/
def total_donation : ℕ := 70000
def first_home_donation : ℕ := 24500
def second_home_donation : ℕ := 22500

/-- The donation to the third home is the difference between the total donation
    and the sum of donations to the first two homes -/
theorem third_home_donation :
  total_donation - first_home_donation - second_home_donation = 23000 := by
  sorry

end third_home_donation_l1206_120604


namespace similar_triangle_shortest_side_l1206_120688

theorem similar_triangle_shortest_side 
  (a b c : ℝ) 
  (h₁ : a^2 + b^2 = c^2) 
  (h₂ : a = 21) 
  (h₃ : c = 29) 
  (h₄ : a ≤ b) 
  (k : ℝ) 
  (h₅ : k * c = 87) : 
  k * a = 60 := by
sorry

end similar_triangle_shortest_side_l1206_120688


namespace largest_possible_b_l1206_120676

theorem largest_possible_b (a b c : ℕ) : 
  (a * b * c = 360) →
  (1 < c) →
  (c < b) →
  (b < a) →
  (∀ a' b' c' : ℕ, (a' * b' * c' = 360) → (1 < c') → (c' < b') → (b' < a') → b' ≤ b) →
  b = 10 :=
by sorry

end largest_possible_b_l1206_120676


namespace polynomial_roots_l1206_120617

theorem polynomial_roots : ∃ (x₁ x₂ x₃ : ℝ), 
  (x₁ = -2 ∧ x₂ = 2 + Real.sqrt 2 ∧ x₃ = 2 - Real.sqrt 2) ∧
  (∀ x : ℝ, x^4 - 4*x^3 + 5*x^2 - 2*x - 8 = 0 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃)) := by
  sorry

end polynomial_roots_l1206_120617


namespace delaware_cell_phones_l1206_120674

/-- The number of cell phones in Delaware -/
def cell_phones_in_delaware (population : ℕ) (phones_per_thousand : ℕ) : ℕ :=
  (population / 1000) * phones_per_thousand

/-- Theorem stating the number of cell phones in Delaware -/
theorem delaware_cell_phones :
  cell_phones_in_delaware 974000 673 = 655502 := by
  sorry

end delaware_cell_phones_l1206_120674


namespace plane_binary_trees_eq_triangulations_l1206_120660

/-- A plane binary tree -/
structure PlaneBinaryTree where
  vertices : Set Nat
  edges : Set (Nat × Nat)
  root : Nat
  leaves : Set Nat

/-- A triangulation of a polygon -/
structure Triangulation where
  vertices : Set Nat
  diagonals : Set (Nat × Nat)

/-- The number of different plane binary trees with one root and n leaves -/
def num_plane_binary_trees (n : Nat) : Nat :=
  sorry

/-- The number of triangulations of an (n+1)-gon -/
def num_triangulations (n : Nat) : Nat :=
  sorry

/-- Theorem stating the equality between the number of plane binary trees and triangulations -/
theorem plane_binary_trees_eq_triangulations (n : Nat) :
  num_plane_binary_trees n = num_triangulations n :=
  sorry

end plane_binary_trees_eq_triangulations_l1206_120660


namespace jacks_savings_after_eight_weeks_l1206_120686

/-- Calculates the amount in Jack's savings account after a given number of weeks -/
def savings_after_weeks (initial_amount : ℝ) (weekly_allowance : ℝ) (weekly_expense : ℝ) (weeks : ℕ) : ℝ :=
  initial_amount + (weekly_allowance - weekly_expense) * weeks

/-- Proves that Jack's savings after 8 weeks equals $99 -/
theorem jacks_savings_after_eight_weeks :
  savings_after_weeks 43 10 3 8 = 99 := by
  sorry

#eval savings_after_weeks 43 10 3 8

end jacks_savings_after_eight_weeks_l1206_120686


namespace inequality_solution_l1206_120670

theorem inequality_solution (x : ℝ) : 
  (x / (x - 1) ≥ 2 * x) ↔ (1 < x ∧ x ≤ 3/2) ∨ (x ≤ 0) :=
by sorry

end inequality_solution_l1206_120670


namespace curve_equation_relationship_l1206_120662

-- Define the curve C as a set of points in 2D space
def C : Set (ℝ × ℝ) := sorry

-- Define the function f
def f : ℝ → ℝ → ℝ := sorry

-- State the theorem
theorem curve_equation_relationship :
  (∀ x y, f x y = 0 → (x, y) ∈ C) →
  (∀ x y, (x, y) ∉ C → f x y ≠ 0) := by
  sorry

end curve_equation_relationship_l1206_120662


namespace cubic_equation_only_trivial_solution_l1206_120659

theorem cubic_equation_only_trivial_solution (x y z : ℤ) :
  x^3 - 2*y^3 - 4*z^3 = 0 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end cubic_equation_only_trivial_solution_l1206_120659


namespace equilateral_triangle_perimeter_l1206_120632

/-- The perimeter of an equilateral triangle with side length 13/12 meters is 3.25 meters. -/
theorem equilateral_triangle_perimeter :
  let side_length : ℚ := 13 / 12
  let perimeter : ℚ := 3 * side_length
  perimeter = 13 / 4 := by sorry

end equilateral_triangle_perimeter_l1206_120632


namespace simplify_expression_l1206_120610

theorem simplify_expression : 18 * (7/8) * (1/12)^2 = 7/768 := by
  sorry

end simplify_expression_l1206_120610


namespace trapezoid_area_is_6_or_10_l1206_120625

/-- Represents a trapezoid with four side lengths -/
structure Trapezoid where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ

/-- Calculates the area of a trapezoid -/
def area (t : Trapezoid) : ℝ := sorry

/-- Theorem stating that a trapezoid with side lengths 1, 4, 4, and 5 has an area of either 6 or 10 -/
theorem trapezoid_area_is_6_or_10 (t : Trapezoid) 
    (h1 : t.side1 = 1 ∨ t.side2 = 1 ∨ t.side3 = 1 ∨ t.side4 = 1)
    (h2 : t.side1 = 4 ∨ t.side2 = 4 ∨ t.side3 = 4 ∨ t.side4 = 4)
    (h3 : t.side1 = 4 ∨ t.side2 = 4 ∨ t.side3 = 4 ∨ t.side4 = 4)
    (h4 : t.side1 = 5 ∨ t.side2 = 5 ∨ t.side3 = 5 ∨ t.side4 = 5)
    (h5 : t.side1 ≠ t.side2 ∨ t.side2 ≠ t.side3 ∨ t.side3 ≠ t.side4) : 
  area t = 6 ∨ area t = 10 := by sorry

end trapezoid_area_is_6_or_10_l1206_120625


namespace parallel_vectors_y_value_l1206_120631

/-- Two planar vectors are parallel if their components are proportional -/
def are_parallel (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

theorem parallel_vectors_y_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (2, y)
  are_parallel a b → y = 4 := by
  sorry

end parallel_vectors_y_value_l1206_120631


namespace inverse_proportion_y_relationship_l1206_120671

/-- Given two points on an inverse proportion function, prove that y₁ < y₂ -/
theorem inverse_proportion_y_relationship (k : ℝ) (y₁ y₂ : ℝ) :
  (2 : ℝ) > 0 ∧ (3 : ℝ) > 0 ∧
  y₁ = (-k^2 - 1) / 2 ∧
  y₂ = (-k^2 - 1) / 3 →
  y₁ < y₂ :=
by sorry

end inverse_proportion_y_relationship_l1206_120671


namespace alexis_bought_21_pants_l1206_120609

/-- Given information about Isabella and Alexis's shopping -/
structure ShoppingInfo where
  isabella_total : ℕ
  alexis_dresses : ℕ
  alexis_multiplier : ℕ

/-- Calculates the number of pants Alexis bought -/
def alexis_pants (info : ShoppingInfo) : ℕ :=
  info.alexis_multiplier * (info.isabella_total - (info.alexis_dresses / info.alexis_multiplier))

/-- Theorem stating that Alexis bought 21 pants given the shopping information -/
theorem alexis_bought_21_pants (info : ShoppingInfo) 
  (h1 : info.isabella_total = 13)
  (h2 : info.alexis_dresses = 18)
  (h3 : info.alexis_multiplier = 3) : 
  alexis_pants info = 21 := by
  sorry

#eval alexis_pants ⟨13, 18, 3⟩

end alexis_bought_21_pants_l1206_120609


namespace unit_digit_product_l1206_120658

theorem unit_digit_product : ∃ n : ℕ, (5 + 1) * (5^3 + 1) * (5^6 + 1) * (5^12 + 1) ≡ 6 [ZMOD 10] := by
  sorry

end unit_digit_product_l1206_120658


namespace quadratic_radical_problem_l1206_120638

-- Define what it means for two quadratic radicals to be of the same type
def same_type (x y : ℝ) : Prop :=
  ∃ (c₁ c₂ : ℝ) (p₁ p₂ : ℕ), c₁ > 0 ∧ c₂ > 0 ∧ 
  Nat.Prime p₁ ∧ Nat.Prime p₂ ∧
  Real.sqrt x = c₁ * Real.sqrt (p₁ : ℝ) ∧
  Real.sqrt y = c₂ * Real.sqrt (p₂ : ℝ) ∧
  c₁ = c₂

-- State the theorem
theorem quadratic_radical_problem (a : ℝ) :
  same_type (3*a - 4) 8 → a = 2 := by
  sorry

end quadratic_radical_problem_l1206_120638


namespace polygon_chain_sides_l1206_120656

/-- Represents a regular polygon with a given number of sides. -/
structure RegularPolygon where
  sides : ℕ
  sides_positive : sides > 0

/-- Represents a chain of connected regular polygons. -/
structure PolygonChain where
  polygons : List RegularPolygon
  connected : polygons.length > 1

/-- Calculates the number of exposed sides in a chain of connected polygons. -/
def exposedSides (chain : PolygonChain) : ℕ :=
  let n := chain.polygons.length
  let total_sides := (chain.polygons.map RegularPolygon.sides).sum
  let shared_sides := 2 * (n - 1) - 2
  total_sides - shared_sides

/-- The theorem to be proved. -/
theorem polygon_chain_sides :
  ∀ (chain : PolygonChain),
    chain.polygons.map RegularPolygon.sides = [3, 4, 5, 6, 7, 8, 9] →
    exposedSides chain = 30 := by
  sorry

end polygon_chain_sides_l1206_120656


namespace factorization_of_2x_squared_minus_8_l1206_120647

theorem factorization_of_2x_squared_minus_8 (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) := by
  sorry

end factorization_of_2x_squared_minus_8_l1206_120647


namespace initial_working_hours_l1206_120681

/-- Given the following conditions:
  - 75 men initially working
  - Initial depth dug: 50 meters
  - New depth to dig: 70 meters
  - New working hours: 6 hours/day
  - 65 extra men added
Prove that the initial working hours H satisfy the equation:
  75 * H * 50 = (75 + 65) * 6 * 70
-/
theorem initial_working_hours (H : ℝ) : 75 * H * 50 = (75 + 65) * 6 * 70 := by
  sorry

end initial_working_hours_l1206_120681


namespace largest_number_l1206_120635

def a : ℝ := 8.12334
def b : ℝ := 8.123333333 -- Approximation of 8.123̅3
def c : ℝ := 8.123333333 -- Approximation of 8.12̅33
def d : ℝ := 8.123323323 -- Approximation of 8.1̅233
def e : ℝ := 8.123312331 -- Approximation of 8.̅1233

theorem largest_number : 
  (b = c) ∧ (b ≥ a) ∧ (b ≥ d) ∧ (b ≥ e) := by sorry

end largest_number_l1206_120635


namespace bob_probability_after_two_turns_l1206_120639

/-- Represents the player who has the ball -/
inductive Player : Type
| Alice : Player
| Bob : Player

/-- The probability of keeping the ball for each player -/
def keep_prob (p : Player) : ℚ :=
  match p with
  | Player.Alice => 2/3
  | Player.Bob => 3/4

/-- The probability of tossing the ball for each player -/
def toss_prob (p : Player) : ℚ :=
  1 - keep_prob p

/-- The probability that Bob has the ball after two turns, given he starts with it -/
def bob_has_ball_after_two_turns : ℚ :=
  keep_prob Player.Bob * keep_prob Player.Bob +
  keep_prob Player.Bob * toss_prob Player.Bob * keep_prob Player.Alice +
  toss_prob Player.Bob * toss_prob Player.Alice

theorem bob_probability_after_two_turns :
  bob_has_ball_after_two_turns = 37/48 := by
  sorry

end bob_probability_after_two_turns_l1206_120639


namespace fixed_point_of_function_l1206_120680

/-- The function f(x) = kx - k - a^(x-1) always passes through the point (1, -1) -/
theorem fixed_point_of_function (k : ℝ) (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f := fun x => k * x - k - a^(x - 1)
  f 1 = -1 := by sorry

end fixed_point_of_function_l1206_120680


namespace sum_of_three_consecutive_integers_l1206_120622

theorem sum_of_three_consecutive_integers :
  ∃ n : ℤ, n - 1 + n + (n + 1) = 21 ∧
  (n - 1 + n + (n + 1) = 17 ∨
   n - 1 + n + (n + 1) = 11 ∨
   n - 1 + n + (n + 1) = 25 ∨
   n - 1 + n + (n + 1) = 21 ∨
   n - 1 + n + (n + 1) = 8) :=
by sorry

end sum_of_three_consecutive_integers_l1206_120622


namespace fifth_largest_divisor_l1206_120613

def n : ℕ := 1936000000

def is_fifth_largest_divisor (d : ℕ) : Prop :=
  d ∣ n ∧ (∃ (a b c e : ℕ), a ∣ n ∧ b ∣ n ∧ c ∣ n ∧ e ∣ n ∧
    a > b ∧ b > c ∧ c > e ∧ e > d ∧
    ∀ (x : ℕ), x ∣ n → x ≤ d ∨ x = e ∨ x = c ∨ x = b ∨ x = a ∨ x = n)

theorem fifth_largest_divisor :
  is_fifth_largest_divisor 121000000 := by sorry

end fifth_largest_divisor_l1206_120613


namespace same_solutions_quadratic_l1206_120661

theorem same_solutions_quadratic (b c : ℝ) : 
  (∀ x : ℝ, |x - 5| = 2 ↔ x^2 + b*x + c = 0) → 
  b = -10 ∧ c = 21 := by
sorry

end same_solutions_quadratic_l1206_120661


namespace race_time_difference_l1206_120607

/-- Race parameters and runner speeds -/
def race_distance : ℕ := 12
def malcolm_speed : ℕ := 7
def joshua_speed : ℕ := 8

/-- Theorem stating the time difference between Malcolm and Joshua finishing the race -/
theorem race_time_difference : 
  joshua_speed * race_distance - malcolm_speed * race_distance = 12 := by
  sorry

end race_time_difference_l1206_120607


namespace max_digit_sum_two_digit_primes_l1206_120699

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

theorem max_digit_sum_two_digit_primes :
  ∃ (p : ℕ), is_two_digit p ∧ is_prime p ∧
    digit_sum p = 17 ∧
    ∀ (q : ℕ), is_two_digit q → is_prime q → digit_sum q ≤ 17 :=
sorry

end max_digit_sum_two_digit_primes_l1206_120699


namespace complex_sum_problem_l1206_120664

theorem complex_sum_problem (x y u v w z : ℝ) 
  (h1 : y = 2)
  (h2 : w = -x - u)
  (h3 : Complex.mk x y + Complex.mk u v + Complex.mk w z = Complex.I * (-2)) :
  v + z = -4 := by
  sorry

end complex_sum_problem_l1206_120664


namespace quadratic_equation_real_roots_l1206_120619

theorem quadratic_equation_real_roots (m : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + (3*m - 1)*x₁ + (2*m^2 - m) = 0 ∧
                x₂^2 + (3*m - 1)*x₂ + (2*m^2 - m) = 0 := by
  sorry

end quadratic_equation_real_roots_l1206_120619


namespace max_reciprocal_sum_l1206_120691

theorem max_reciprocal_sum (m n : ℝ) (h1 : m * n > 0) (h2 : m + n = -1) :
  1 / m + 1 / n ≤ 4 := by
sorry

end max_reciprocal_sum_l1206_120691


namespace prime_square_difference_one_l1206_120623

theorem prime_square_difference_one (p q : ℕ) : 
  Prime p → Prime q → p^2 - 2*q^2 = 1 → (p = 3 ∧ q = 2) :=
by sorry

end prime_square_difference_one_l1206_120623


namespace dan_gave_41_cards_l1206_120605

/-- Given the initial number of cards, the number of cards bought, and the final number of cards,
    calculate the number of cards given by Dan. -/
def cards_given_by_dan (initial_cards : ℕ) (bought_cards : ℕ) (final_cards : ℕ) : ℕ :=
  final_cards - initial_cards - bought_cards

/-- Theorem stating that Dan gave Sally 41 cards -/
theorem dan_gave_41_cards :
  cards_given_by_dan 27 20 88 = 41 := by
  sorry

end dan_gave_41_cards_l1206_120605


namespace expansion_coefficient_sum_l1206_120684

theorem expansion_coefficient_sum (a : ℝ) : 
  ((-a)^4 * (Nat.choose 8 4 : ℝ) = 1120) → 
  ((1 - a)^8 = 1 ∨ (1 - a)^8 = 6561) := by
  sorry

end expansion_coefficient_sum_l1206_120684


namespace theorem_1_theorem_2_l1206_120692

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the functional equation
def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x + f y = 2 * f ((x + y) / 2) * f ((x - y) / 2)

-- Theorem 1: If f(1) = 1/2, then f(2) = -1/2
theorem theorem_1 (h : functional_equation f) (h1 : f 1 = 1/2) : f 2 = -1/2 := by
  sorry

-- Theorem 2: If f(1) = 0, then f(11/2) + f(15/2) + f(19/2) + ... + f(2019/2) + f(2023/2) = 0
theorem theorem_2 (h : functional_equation f) (h1 : f 1 = 0) :
  f (11/2) + f (15/2) + f (19/2) + f (2019/2) + f (2023/2) = 0 := by
  sorry

end theorem_1_theorem_2_l1206_120692


namespace gcd_abcd_plus_dcba_l1206_120600

def abcd_plus_dcba (a : ℕ) : ℕ := 2222 * a + 12667

theorem gcd_abcd_plus_dcba : 
  Nat.gcd (abcd_plus_dcba 0) (Nat.gcd (abcd_plus_dcba 1) (Nat.gcd (abcd_plus_dcba 2) (abcd_plus_dcba 3))) = 2222 := by
  sorry

end gcd_abcd_plus_dcba_l1206_120600


namespace grandmas_salad_ratio_l1206_120621

/-- Given the conditions of Grandma's salad, prove the ratio of pickles to cherry tomatoes -/
theorem grandmas_salad_ratio : 
  ∀ (mushrooms cherry_tomatoes pickles bacon_bits red_bacon_bits : ℕ),
    mushrooms = 3 →
    cherry_tomatoes = 2 * mushrooms →
    bacon_bits = 4 * pickles →
    red_bacon_bits * 3 = bacon_bits →
    red_bacon_bits = 32 →
    (pickles : ℚ) / cherry_tomatoes = 4 / 1 :=
by
  sorry

end grandmas_salad_ratio_l1206_120621


namespace total_money_sally_condition_jolly_condition_molly_condition_l1206_120644

/-- The amount of money Sally has -/
def sally_money : ℕ := 100

/-- The amount of money Jolly has -/
def jolly_money : ℕ := 50

/-- The amount of money Molly has -/
def molly_money : ℕ := 70

/-- The theorem stating the total amount of money -/
theorem total_money : sally_money + jolly_money + molly_money = 220 := by
  sorry

/-- Sally would have $80 if she had $20 less -/
theorem sally_condition : sally_money - 20 = 80 := by
  sorry

/-- Jolly would have $70 if she had $20 more -/
theorem jolly_condition : jolly_money + 20 = 70 := by
  sorry

/-- Molly would have $100 if she had $30 more -/
theorem molly_condition : molly_money + 30 = 100 := by
  sorry

end total_money_sally_condition_jolly_condition_molly_condition_l1206_120644


namespace geometric_sequence_properties_l1206_120695

/-- A geometric sequence with sum S_n = k^n + r^m -/
structure GeometricSequence where
  k : ℝ
  r : ℝ
  m : ℤ
  a : ℕ → ℝ
  sum : ℕ → ℝ
  is_geometric : ∀ n, a (n + 1) = a n * (a 2 / a 1)
  sum_formula : ∀ n, sum n = k^n + r^m

/-- The properties of r and m in the geometric sequence -/
theorem geometric_sequence_properties (seq : GeometricSequence) : 
  seq.r = -1 ∧ Odd seq.m :=
by sorry

end geometric_sequence_properties_l1206_120695


namespace like_terms_exponent_difference_l1206_120624

theorem like_terms_exponent_difference (m n : ℕ) : 
  (∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a * X^m * Y = b * X^3 * Y^n) → m - n = 2 :=
by sorry

end like_terms_exponent_difference_l1206_120624


namespace expected_yolks_in_carton_l1206_120675

/-- Represents a carton of eggs with various yolk counts -/
structure EggCarton where
  total_eggs : ℕ
  double_yolk_eggs : ℕ
  triple_yolk_eggs : ℕ
  extra_yolk_probability : ℝ

/-- Calculates the expected number of yolks in a carton of eggs -/
def expected_yolks (carton : EggCarton) : ℝ :=
  let single_yolk_eggs := carton.total_eggs - carton.double_yolk_eggs - carton.triple_yolk_eggs
  let base_yolks := single_yolk_eggs + 2 * carton.double_yolk_eggs + 3 * carton.triple_yolk_eggs
  let extra_yolks := carton.extra_yolk_probability * (carton.double_yolk_eggs + carton.triple_yolk_eggs)
  base_yolks + extra_yolks

/-- Theorem stating the expected number of yolks in the given carton -/
theorem expected_yolks_in_carton :
  let carton : EggCarton := {
    total_eggs := 15,
    double_yolk_eggs := 5,
    triple_yolk_eggs := 3,
    extra_yolk_probability := 0.1
  }
  expected_yolks carton = 26.8 := by sorry

end expected_yolks_in_carton_l1206_120675


namespace arithmetic_sequence_problem_l1206_120654

/-- Given an arithmetic sequence, prove that if the sum of the first four terms is 2l,
    the sum of the last four terms is 67, and the sum of the first n terms is 286,
    then the number of terms n is 26. -/
theorem arithmetic_sequence_problem (l : ℝ) (a d : ℝ) (n : ℕ) :
  (4 * a + 6 * d = 2 * l) →
  (4 * (a + (n - 1) * d) - 6 * d = 67) →
  (n * (2 * a + (n - 1) * d) / 2 = 286) →
  n = 26 := by
sorry

end arithmetic_sequence_problem_l1206_120654


namespace soccer_field_area_l1206_120603

theorem soccer_field_area (w l : ℝ) (h1 : l = 3 * w - 30) (h2 : 2 * (w + l) = 880) :
  w * l = 37906.25 := by
  sorry

end soccer_field_area_l1206_120603


namespace complement_of_union_is_two_four_l1206_120693

-- Define the universal set U
def U : Set ℕ := {x | x > 0 ∧ x < 6}

-- Define sets A and B
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

-- State the theorem
theorem complement_of_union_is_two_four :
  (U \ (A ∪ B)) = {2, 4} := by sorry

end complement_of_union_is_two_four_l1206_120693


namespace square_area_ratio_l1206_120601

theorem square_area_ratio (s t : ℝ) (h : s > 0) (k : t > 0) (h_perimeter : 4 * s = 4 * (4 * t)) :
  s^2 = 16 * t^2 := by
  sorry

end square_area_ratio_l1206_120601


namespace quadratic_distinct_integer_roots_l1206_120673

theorem quadratic_distinct_integer_roots (a : ℤ) : 
  (∃ x y : ℤ, x ≠ y ∧ 2 * x^2 - a * x + 2 * a = 0 ∧ 2 * y^2 - a * y + 2 * a = 0) ↔ 
  (a = -2 ∨ a = 18) :=
sorry

end quadratic_distinct_integer_roots_l1206_120673


namespace equality_condition_l1206_120650

theorem equality_condition (a b c : ℝ) : a^2 + b*c = (a - b)*(a - c) ↔ a = 0 ∨ b + c = 0 := by
  sorry

end equality_condition_l1206_120650


namespace cos_2alpha_plus_5pi_12_l1206_120651

theorem cos_2alpha_plus_5pi_12 (α : Real) (h1 : π < α ∧ α < 2*π) 
  (h2 : Real.sin (α + π/3) = -4/5) : 
  Real.cos (2*α + 5*π/12) = 17*Real.sqrt 2/50 := by
sorry

end cos_2alpha_plus_5pi_12_l1206_120651


namespace ace_of_hearts_probability_l1206_120626

def standard_deck := 52
def ace_of_hearts_per_deck := 1

theorem ace_of_hearts_probability (combined_deck : ℕ) (ace_of_hearts : ℕ) :
  combined_deck = 2 * standard_deck →
  ace_of_hearts = 2 * ace_of_hearts_per_deck →
  (ace_of_hearts : ℚ) / combined_deck = 1 / 52 :=
by sorry

end ace_of_hearts_probability_l1206_120626


namespace intersection_and_non_membership_l1206_120640

-- Define the lines
def line1 (x y : ℚ) : Prop := y = -3 * x
def line2 (x y : ℚ) : Prop := y + 3 = 9 * x
def line3 (x y : ℚ) : Prop := y = 2 * x - 1

-- Define the intersection point
def intersection_point : ℚ × ℚ := (1/4, -3/4)

-- Theorem statement
theorem intersection_and_non_membership :
  let (x, y) := intersection_point
  (line1 x y ∧ line2 x y) ∧ ¬(line3 x y) := by sorry

end intersection_and_non_membership_l1206_120640


namespace cube_order_preserving_l1206_120685

theorem cube_order_preserving (a b : ℝ) : a^3 > b^3 → a > b := by
  sorry

end cube_order_preserving_l1206_120685


namespace gcf_275_180_l1206_120636

theorem gcf_275_180 : Nat.gcd 275 180 = 5 := by
  sorry

end gcf_275_180_l1206_120636


namespace calculation_proof_l1206_120690

theorem calculation_proof : (300000 * 200000) / 100000 = 600000 := by
  sorry

end calculation_proof_l1206_120690


namespace max_intersection_points_fifth_degree_polynomials_l1206_120689

/-- A fifth-degree polynomial function with leading coefficient 1 -/
def FifthDegreePolynomial (a b c d e : ℝ) : ℝ → ℝ := 
  λ x => x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + e

/-- The difference between two fifth-degree polynomials where one has an additional -x^3 term -/
def PolynomialDifference (p q : ℝ → ℝ) : ℝ → ℝ :=
  λ x => p x - q x

theorem max_intersection_points_fifth_degree_polynomials :
  ∀ (a₁ b₁ c₁ d₁ e₁ a₂ b₂ c₂ d₂ e₂ : ℝ),
  let p := FifthDegreePolynomial a₁ b₁ c₁ d₁ e₁
  let q := FifthDegreePolynomial a₂ (b₂ - 1) c₂ d₂ e₂
  let diff := PolynomialDifference p q
  (∀ x : ℝ, diff x = 0 → x = 0) ∧
  (∃ x : ℝ, diff x = 0) :=
by sorry

end max_intersection_points_fifth_degree_polynomials_l1206_120689


namespace binomial_coefficient_19_12_l1206_120634

theorem binomial_coefficient_19_12 : 
  (Nat.choose 20 12 = 125970) → 
  (Nat.choose 19 11 = 75582) → 
  (Nat.choose 18 11 = 31824) → 
  (Nat.choose 19 12 = 50388) := by
sorry

end binomial_coefficient_19_12_l1206_120634


namespace hexagon_ratio_l1206_120630

/-- A hexagon with specific properties -/
structure Hexagon :=
  (total_area : ℝ)
  (bisector : ℝ → ℝ → Prop)
  (lower_part : ℝ → ℝ → Prop)
  (triangle_base : ℝ)

/-- The theorem statement -/
theorem hexagon_ratio (h : Hexagon) (x y : ℝ) : 
  h.total_area = 7 ∧ 
  h.bisector x y ∧ 
  h.lower_part 1 (5/2) ∧ 
  h.triangle_base = 4 →
  x / y = 1 :=
by sorry

end hexagon_ratio_l1206_120630


namespace peaches_in_basket_l1206_120618

/-- Represents the number of peaches in a basket -/
structure Basket :=
  (red : ℕ)
  (green : ℕ)

/-- The total number of peaches in a basket is the sum of red and green peaches -/
def total_peaches (b : Basket) : ℕ := b.red + b.green

/-- Given a basket with 7 red peaches and 3 green peaches, prove that the total number of peaches is 10 -/
theorem peaches_in_basket :
  ∀ b : Basket, b.red = 7 ∧ b.green = 3 → total_peaches b = 10 :=
by
  sorry

#check peaches_in_basket

end peaches_in_basket_l1206_120618


namespace fifteen_distinct_configurations_l1206_120633

/-- Represents a 4x4x4 cube configuration with 63 white cubes and 1 black cube -/
def CubeConfiguration := Fin 4 → Fin 4 → Fin 4 → Bool

/-- Counts the number of distinct cube configurations -/
def countDistinctConfigurations : ℕ :=
  let corner_configs := 1
  let edge_configs := 2
  let face_configs := 1
  let inner_configs := 8
  corner_configs + edge_configs + face_configs + inner_configs

/-- Theorem stating that there are 15 distinct cube configurations -/
theorem fifteen_distinct_configurations :
  countDistinctConfigurations = 15 := by
  sorry

end fifteen_distinct_configurations_l1206_120633


namespace inequality_proof_l1206_120627

theorem inequality_proof (a b t : ℝ) (h1 : 0 < t) (h2 : t < 1) (h3 : a * b > 0) :
  (a^2 / t^3) + (b^2 / (1 - t^3)) ≥ (a + b)^2 := by
  sorry

end inequality_proof_l1206_120627


namespace rect_to_cylindrical_conversion_l1206_120669

/-- Conversion from rectangular to cylindrical coordinates -/
theorem rect_to_cylindrical_conversion 
  (x y z : ℝ) 
  (h_x : x = -2) 
  (h_y : y = -2 * Real.sqrt 3) 
  (h_z : z = -1) :
  ∃ (r θ : ℝ),
    r > 0 ∧
    0 ≤ θ ∧ θ < 2 * Real.pi ∧
    r = 4 ∧
    θ = 4 * Real.pi / 3 ∧
    x = r * Real.cos θ ∧
    y = r * Real.sin θ ∧
    z = -1 :=
by sorry

end rect_to_cylindrical_conversion_l1206_120669
