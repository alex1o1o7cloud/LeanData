import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l2376_237639

theorem inequality_proof (x a : ℝ) (h1 : x < a) (h2 : a < -1) (h3 : x < 0) (h4 : a < 0) :
  x^2 > a*x ∧ a*x > a^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2376_237639


namespace NUMINAMATH_CALUDE_h_transformation_l2376_237697

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define the transformation h
def h (f : ℝ → ℝ) (x : ℝ) : ℝ := 2 * f x + 3

-- Theorem statement
theorem h_transformation (f : ℝ → ℝ) (x : ℝ) : 
  h f x = 2 * f x + 3 := by
  sorry

end NUMINAMATH_CALUDE_h_transformation_l2376_237697


namespace NUMINAMATH_CALUDE_quadruple_inequality_l2376_237695

theorem quadruple_inequality (a p q r : ℕ) 
  (ha : a > 1) (hp : p > 1) (hq : q > 1) (hr : r > 1)
  (hdiv_p : p ∣ a * q * r + 1)
  (hdiv_q : q ∣ a * p * r + 1)
  (hdiv_r : r ∣ a * p * q + 1) :
  a ≥ (p * q * r - 1) / (p * q + q * r + r * p) :=
sorry

end NUMINAMATH_CALUDE_quadruple_inequality_l2376_237695


namespace NUMINAMATH_CALUDE_barney_weight_difference_l2376_237688

-- Define the weight of a regular dinosaur
def regular_dinosaur_weight : ℕ := 800

-- Define the number of regular dinosaurs
def num_regular_dinosaurs : ℕ := 5

-- Define the total weight of Barney and the regular dinosaurs
def total_weight : ℕ := 9500

-- Define Barney's weight
def barney_weight : ℕ := total_weight - (num_regular_dinosaurs * regular_dinosaur_weight)

-- Theorem to prove
theorem barney_weight_difference : 
  barney_weight - (num_regular_dinosaurs * regular_dinosaur_weight) = 1500 := by
  sorry

end NUMINAMATH_CALUDE_barney_weight_difference_l2376_237688


namespace NUMINAMATH_CALUDE_probability_different_rooms_l2376_237612

theorem probability_different_rooms (n : ℕ) (h : n = 2) : 
  (n - 1 : ℚ) / n = 1 / 2 := by
  sorry

#check probability_different_rooms

end NUMINAMATH_CALUDE_probability_different_rooms_l2376_237612


namespace NUMINAMATH_CALUDE_problem_solution_l2376_237686

theorem problem_solution (r s : ℝ) 
  (h1 : 1 < r) 
  (h2 : r < s) 
  (h3 : 1 / r + 1 / s = 1) 
  (h4 : r * s = 15 / 4) : 
  s = (15 + Real.sqrt 15) / 8 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2376_237686


namespace NUMINAMATH_CALUDE_evaluate_expression_l2376_237637

theorem evaluate_expression (a : ℚ) (h : a = 4/3) : (6*a^2 - 11*a + 2)*(3*a - 4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2376_237637


namespace NUMINAMATH_CALUDE_largest_power_dividing_factorial_l2376_237679

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem largest_power_dividing_factorial :
  let n := 2520
  ∃ k : ℕ, k = 418 ∧
    (∀ m : ℕ, n^m ∣ factorial n → m ≤ k) ∧
    n^k ∣ factorial n :=
by sorry

end NUMINAMATH_CALUDE_largest_power_dividing_factorial_l2376_237679


namespace NUMINAMATH_CALUDE_no_real_roots_quadratic_l2376_237632

theorem no_real_roots_quadratic (k : ℝ) : 
  (∀ x : ℝ, x^2 - x + k ≠ 0) → k > 1/4 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_quadratic_l2376_237632


namespace NUMINAMATH_CALUDE_equal_size_meetings_l2376_237635

/-- Given n sets representing daily meetings, prove that all sets have the same size. -/
theorem equal_size_meetings (n : ℕ) (A : Fin n → Finset (Fin n)) 
  (h_n : n ≥ 3)
  (h_size : ∀ i, (A i).card ≥ 3)
  (h_cover : ∀ i j, i < j → ∃! k, i ∈ A k ∧ j ∈ A k) :
  ∃ k, ∀ i, (A i).card = k :=
sorry

end NUMINAMATH_CALUDE_equal_size_meetings_l2376_237635


namespace NUMINAMATH_CALUDE_exists_min_value_in_interval_l2376_237653

-- Define the function
def f (x : ℝ) : ℝ := x^3 + x^2 - x + 1

-- State the theorem
theorem exists_min_value_in_interval :
  ∃ (m : ℝ), ∀ (x : ℝ), -2 ≤ x ∧ x ≤ 1 → m ≤ f x :=
sorry

end NUMINAMATH_CALUDE_exists_min_value_in_interval_l2376_237653


namespace NUMINAMATH_CALUDE_least_integer_with_specific_divisibility_l2376_237617

theorem least_integer_with_specific_divisibility : ∃ (n : ℕ), 
  (∀ (k : ℕ), k ≤ 28 → k ∣ n) ∧ 
  (30 ∣ n) ∧ 
  ¬(29 ∣ n) ∧
  (∀ (m : ℕ), m < n → ¬((∀ (k : ℕ), k ≤ 28 → k ∣ m) ∧ (30 ∣ m) ∧ ¬(29 ∣ m))) ∧
  n = 232792560 := by
sorry

end NUMINAMATH_CALUDE_least_integer_with_specific_divisibility_l2376_237617


namespace NUMINAMATH_CALUDE_pascal_interior_sum_l2376_237627

/-- Sum of interior numbers in the n-th row of Pascal's Triangle -/
def sumInteriorNumbers (n : ℕ) : ℕ := 2^(n-1) - 2

theorem pascal_interior_sum :
  (sumInteriorNumbers 5 = 14) →
  (sumInteriorNumbers 6 = 30) →
  (sumInteriorNumbers 8 = 126) :=
by
  sorry

end NUMINAMATH_CALUDE_pascal_interior_sum_l2376_237627


namespace NUMINAMATH_CALUDE_negative_expression_l2376_237614

theorem negative_expression (a b c : ℝ) 
  (ha : 0 < a ∧ a < 2) 
  (hb : -2 < b ∧ b < 0) 
  (hc : 0 < c ∧ c < 1) : 
  b + 3 * b^2 < 0 :=
by sorry

end NUMINAMATH_CALUDE_negative_expression_l2376_237614


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2376_237616

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 1 > 0 →
  a 1 * a 5 + 2 * a 3 * a 5 + a 3 * a 7 = 16 →
  a 3 + a 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2376_237616


namespace NUMINAMATH_CALUDE_cabinet_installation_ratio_l2376_237625

/-- Proves the ratio of newly installed cabinets to initial cabinets --/
theorem cabinet_installation_ratio : 
  ∀ (x : ℕ), 
  (3 : ℕ) + 3 * x + (5 : ℕ) = (26 : ℕ) → 
  (x : ℚ) / (3 : ℚ) = (2 : ℚ) / (1 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_cabinet_installation_ratio_l2376_237625


namespace NUMINAMATH_CALUDE_team_b_mean_tasks_l2376_237674

/-- Represents the office with two teams -/
structure Office :=
  (total_members : ℕ)
  (team_a_members : ℕ)
  (team_b_members : ℕ)
  (team_a_mean_tasks : ℝ)
  (team_b_mean_tasks : ℝ)

/-- The conditions of the office as described in the problem -/
def office_conditions (o : Office) : Prop :=
  o.total_members = 260 ∧
  o.team_a_members = (13 * o.team_b_members) / 10 ∧
  o.team_a_mean_tasks = 80 ∧
  o.team_b_mean_tasks = (6 * o.team_a_mean_tasks) / 5

/-- The theorem stating that under the given conditions, Team B's mean tasks is 96 -/
theorem team_b_mean_tasks (o : Office) (h : office_conditions o) : 
  o.team_b_mean_tasks = 96 := by
  sorry


end NUMINAMATH_CALUDE_team_b_mean_tasks_l2376_237674


namespace NUMINAMATH_CALUDE_prob_same_color_is_correct_l2376_237655

def red_marbles : ℕ := 5
def white_marbles : ℕ := 6
def blue_marbles : ℕ := 7
def green_marbles : ℕ := 3
def total_marbles : ℕ := red_marbles + white_marbles + blue_marbles + green_marbles

def prob_same_color : ℚ :=
  (red_marbles * (red_marbles - 1) * (red_marbles - 2) +
   white_marbles * (white_marbles - 1) * (white_marbles - 2) +
   blue_marbles * (blue_marbles - 1) * (blue_marbles - 2) +
   green_marbles * (green_marbles - 1) * (green_marbles - 2)) /
  (total_marbles * (total_marbles - 1) * (total_marbles - 2))

theorem prob_same_color_is_correct : prob_same_color = 66 / 1330 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_color_is_correct_l2376_237655


namespace NUMINAMATH_CALUDE_caroline_citrus_drinks_l2376_237684

/-- The number of citrus drinks Caroline can make from a given number of oranges -/
def citrus_drinks (oranges : ℕ) : ℕ :=
  8 * oranges / 3

/-- Theorem stating that Caroline can make 56 citrus drinks from 21 oranges -/
theorem caroline_citrus_drinks :
  citrus_drinks 21 = 56 := by
  sorry

end NUMINAMATH_CALUDE_caroline_citrus_drinks_l2376_237684


namespace NUMINAMATH_CALUDE_recipe_total_cups_l2376_237660

/-- Given a recipe with a butter:flour:sugar ratio of 1:6:4, prove that when 8 cups of sugar are used, the total cups of ingredients is 22. -/
theorem recipe_total_cups (butter flour sugar total : ℚ) : 
  butter / sugar = 1 / 4 →
  flour / sugar = 6 / 4 →
  sugar = 8 →
  total = butter + flour + sugar →
  total = 22 := by
sorry

end NUMINAMATH_CALUDE_recipe_total_cups_l2376_237660


namespace NUMINAMATH_CALUDE_deal_or_no_deal_probability_l2376_237685

theorem deal_or_no_deal_probability (total : Nat) (desired : Nat) (chosen : Nat) 
  (h1 : total = 26)
  (h2 : desired = 9)
  (h3 : chosen = 1) :
  ∃ (removed : Nat), 
    (1 : ℚ) * desired / (total - removed - chosen) ≥ (1 : ℚ) / 2 ∧ 
    ∀ (r : Nat), r < removed → (1 : ℚ) * desired / (total - r - chosen) < (1 : ℚ) / 2 :=
by sorry

end NUMINAMATH_CALUDE_deal_or_no_deal_probability_l2376_237685


namespace NUMINAMATH_CALUDE_sin_cos_values_l2376_237607

theorem sin_cos_values (α : Real) (h : Real.sin α + 3 * Real.cos α = 0) :
  (Real.sin α = 3 * (Real.sqrt 10) / 10 ∧ Real.cos α = -(Real.sqrt 10) / 10) ∨
  (Real.sin α = -(3 * (Real.sqrt 10) / 10) ∧ Real.cos α = (Real.sqrt 10) / 10) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_values_l2376_237607


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l2376_237643

theorem complex_number_quadrant (z : ℂ) (m : ℝ) 
  (h1 : z * Complex.I = Complex.I + m)
  (h2 : z.im = 1) : 
  0 < z.re :=
sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l2376_237643


namespace NUMINAMATH_CALUDE_triangle_median_altitude_equations_l2376_237646

/-- Triangle ABC with vertices A(-5, 0), B(4, -4), and C(0, 2) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

def triangle_ABC : Triangle :=
  { A := (-5, 0)
    B := (4, -4)
    C := (0, 2) }

/-- The equation of the line on which the median to side BC lies -/
def median_equation (t : Triangle) : LineEquation :=
  { a := 1
    b := 7
    c := 5 }

/-- The equation of the line on which the altitude from A to side BC lies -/
def altitude_equation (t : Triangle) : LineEquation :=
  { a := 2
    b := -3
    c := 10 }

theorem triangle_median_altitude_equations :
  (median_equation triangle_ABC).a = 1 ∧
  (median_equation triangle_ABC).b = 7 ∧
  (median_equation triangle_ABC).c = 5 ∧
  (altitude_equation triangle_ABC).a = 2 ∧
  (altitude_equation triangle_ABC).b = -3 ∧
  (altitude_equation triangle_ABC).c = 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_median_altitude_equations_l2376_237646


namespace NUMINAMATH_CALUDE_coefficient_c_negative_l2376_237648

theorem coefficient_c_negative 
  (a b c : ℝ) 
  (sum_neg : a + b + c < 0) 
  (no_real_roots : ∀ x : ℝ, a * x^2 + b * x + c ≠ 0) : 
  c < 0 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_c_negative_l2376_237648


namespace NUMINAMATH_CALUDE_average_weight_increase_l2376_237682

theorem average_weight_increase 
  (n : ℕ) 
  (old_weight new_weight : ℝ) 
  (h1 : n = 8)
  (h2 : old_weight = 35)
  (h3 : new_weight = 55) :
  (new_weight - old_weight) / n = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_increase_l2376_237682


namespace NUMINAMATH_CALUDE_kitchen_area_is_265_l2376_237640

def total_area : ℕ := 1110
def num_bedrooms : ℕ := 4
def bedroom_length : ℕ := 11
def num_bathrooms : ℕ := 2
def bathroom_length : ℕ := 6
def bathroom_width : ℕ := 8

def bedroom_area : ℕ := bedroom_length * bedroom_length
def bathroom_area : ℕ := bathroom_length * bathroom_width
def total_bedroom_area : ℕ := num_bedrooms * bedroom_area
def total_bathroom_area : ℕ := num_bathrooms * bathroom_area
def remaining_area : ℕ := total_area - (total_bedroom_area + total_bathroom_area)

theorem kitchen_area_is_265 : remaining_area / 2 = 265 := by
  sorry

end NUMINAMATH_CALUDE_kitchen_area_is_265_l2376_237640


namespace NUMINAMATH_CALUDE_emily_sixth_quiz_score_l2376_237626

def emily_scores : List ℝ := [94, 97, 88, 91, 102]

theorem emily_sixth_quiz_score :
  let n : ℕ := emily_scores.length
  let sum : ℝ := emily_scores.sum
  let target_mean : ℝ := 95
  let target_sum : ℝ := target_mean * (n + 1)
  let sixth_score : ℝ := target_sum - sum
  sixth_score = 98 ∧ (sum + sixth_score) / (n + 1) = target_mean :=
by sorry

end NUMINAMATH_CALUDE_emily_sixth_quiz_score_l2376_237626


namespace NUMINAMATH_CALUDE_train_speed_l2376_237636

/-- The speed of a train given its length, time to cross a man, and the man's speed -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (man_speed_kmh : ℝ) :
  train_length = 800 →
  crossing_time = 47.99616030717543 →
  man_speed_kmh = 5 →
  ∃ (train_speed : ℝ), abs (train_speed - 64.9848) < 0.0001 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_l2376_237636


namespace NUMINAMATH_CALUDE_jellybean_probability_l2376_237694

/-- The total number of jellybeans in the jar -/
def total_jellybeans : ℕ := 15

/-- The number of red jellybeans in the jar -/
def red_jellybeans : ℕ := 6

/-- The number of blue jellybeans in the jar -/
def blue_jellybeans : ℕ := 3

/-- The number of white jellybeans in the jar -/
def white_jellybeans : ℕ := 6

/-- The number of jellybeans picked -/
def picked_jellybeans : ℕ := 4

/-- The probability of picking at least 3 red jellybeans out of 4 -/
def prob_at_least_three_red : ℚ := 13 / 91

theorem jellybean_probability :
  let total_outcomes := Nat.choose total_jellybeans picked_jellybeans
  let favorable_outcomes := Nat.choose red_jellybeans 3 * Nat.choose (total_jellybeans - red_jellybeans) 1 +
                            Nat.choose red_jellybeans 4
  (favorable_outcomes : ℚ) / total_outcomes = prob_at_least_three_red :=
by sorry

end NUMINAMATH_CALUDE_jellybean_probability_l2376_237694


namespace NUMINAMATH_CALUDE_cars_meet_time_l2376_237669

/-- Represents a rectangle ABCD -/
structure Rectangle where
  BC : ℝ
  CD : ℝ

/-- Represents a car with a constant speed -/
structure Car where
  speed : ℝ

/-- Time for cars to meet on diagonal BD -/
def meetingTime (rect : Rectangle) (car1 car2 : Car) : ℝ :=
  40 -- in minutes

/-- Theorem stating that cars meet after 40 minutes -/
theorem cars_meet_time (rect : Rectangle) (car1 car2 : Car) :
  meetingTime rect car1 car2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_cars_meet_time_l2376_237669


namespace NUMINAMATH_CALUDE_min_max_sum_l2376_237681

theorem min_max_sum (a b c d e f g : ℝ) 
  (non_neg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 ∧ f ≥ 0 ∧ g ≥ 0) 
  (sum_one : a + b + c + d + e + f + g = 1) : 
  max (a + b + c) (max (b + c + d) (max (c + d + e) (max (d + e + f) (e + f + g)))) ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_min_max_sum_l2376_237681


namespace NUMINAMATH_CALUDE_quadratic_points_relationship_l2376_237628

theorem quadratic_points_relationship :
  let f (x : ℝ) := (x - 2)^2 - 1
  let y₁ := f 4
  let y₂ := f (Real.sqrt 2)
  let y₃ := f (-2)
  y₃ > y₁ ∧ y₁ > y₂ := by
sorry

end NUMINAMATH_CALUDE_quadratic_points_relationship_l2376_237628


namespace NUMINAMATH_CALUDE_max_value_expression_l2376_237611

theorem max_value_expression (x y : ℝ) :
  (2 * x + 3 * y + 4) / Real.sqrt (x^4 + y^2 + 1) ≤ Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l2376_237611


namespace NUMINAMATH_CALUDE_distance_point_to_line_polar_example_l2376_237649

/-- The distance from a point to a line in polar coordinates -/
def distance_point_to_line_polar (ρ₀ : ℝ) (θ₀ : ℝ) (f : ℝ → ℝ → ℝ) : ℝ :=
  sorry

theorem distance_point_to_line_polar_example :
  distance_point_to_line_polar 2 (π/3) (fun ρ θ ↦ ρ * Real.cos (θ + π/3) - 2) = 3 :=
sorry

end NUMINAMATH_CALUDE_distance_point_to_line_polar_example_l2376_237649


namespace NUMINAMATH_CALUDE_house_transaction_net_change_l2376_237693

/-- Represents the financial state of a person -/
structure FinancialState where
  cash : Int
  ownsHouse : Bool

/-- Represents a house transaction -/
structure Transaction where
  seller : String
  buyer : String
  price : Int

/-- Calculate the net change in wealth after transactions -/
def netChangeInWealth (initial : FinancialState) (final : FinancialState) (initialHouseValue : Int) : Int :=
  final.cash - initial.cash + (if final.ownsHouse then initialHouseValue else 0) - (if initial.ownsHouse then initialHouseValue else 0)

theorem house_transaction_net_change :
  let initialHouseValue := 15000
  let initialA := FinancialState.mk 15000 true
  let initialB := FinancialState.mk 20000 false
  let transaction1 := Transaction.mk "A" "B" 18000
  let transaction2 := Transaction.mk "B" "A" 12000
  let finalA := FinancialState.mk 21000 true
  let finalB := FinancialState.mk 14000 false
  (netChangeInWealth initialA finalA initialHouseValue = 6000) ∧
  (netChangeInWealth initialB finalB initialHouseValue = -6000) := by
  sorry

end NUMINAMATH_CALUDE_house_transaction_net_change_l2376_237693


namespace NUMINAMATH_CALUDE_pop_expenditure_l2376_237645

theorem pop_expenditure (total : ℝ) (snap crackle pop : ℝ) : 
  total = 150 ∧ 
  snap = 2 * crackle ∧ 
  crackle = 3 * pop ∧ 
  total = snap + crackle + pop →
  pop = 15 := by
sorry

end NUMINAMATH_CALUDE_pop_expenditure_l2376_237645


namespace NUMINAMATH_CALUDE_average_score_theorem_l2376_237641

def perfect_score : ℕ := 30
def deduction_per_mistake : ℕ := 2

def madeline_mistakes : ℕ := 2
def leo_mistakes : ℕ := 2 * madeline_mistakes
def brent_mistakes : ℕ := leo_mistakes + 1
def nicholas_mistakes : ℕ := 3 * madeline_mistakes

def brent_score : ℕ := 25
def nicholas_score : ℕ := brent_score - 5

def student_score (mistakes : ℕ) : ℕ := perfect_score - mistakes * deduction_per_mistake

theorem average_score_theorem : 
  (student_score madeline_mistakes + student_score leo_mistakes + brent_score + nicholas_score) / 4 = 83 / 4 := by
  sorry

end NUMINAMATH_CALUDE_average_score_theorem_l2376_237641


namespace NUMINAMATH_CALUDE_complement_of_M_l2376_237624

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {1, 3, 5}

theorem complement_of_M :
  (U \ M) = {2, 4, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l2376_237624


namespace NUMINAMATH_CALUDE_total_nuts_weight_l2376_237680

def almonds : Real := 0.14
def pecans : Real := 0.38

theorem total_nuts_weight : almonds + pecans = 0.52 := by sorry

end NUMINAMATH_CALUDE_total_nuts_weight_l2376_237680


namespace NUMINAMATH_CALUDE_locus_of_P_l2376_237690

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := x - 2*y - 1 = 0
def l₂ (x y : ℝ) : Prop := 2*x - y - 2 = 0

-- Define point Q
def Q : ℝ × ℝ := (2, -1)

-- Define the condition for a point P to be on the locus
def on_locus (P : ℝ × ℝ) : Prop :=
  P.1 - P.2 - 1 = 0 ∧ P ≠ (3, 4) ∧ P ≠ (-2, -3) ∧ P ≠ (1, 0)

-- State the theorem
theorem locus_of_P (P A B : ℝ × ℝ) :
  l₁ A.1 A.2 →
  l₂ B.1 B.2 →
  (∃ (t : ℝ), P = (1 - t) • A + t • B) →
  P ≠ Q →
  (P.1 - A.1) / (B.1 - P.1) = (Q.1 - A.1) / (B.1 - Q.1) →
  (P.2 - A.2) / (B.2 - P.2) = (Q.2 - A.2) / (B.2 - Q.2) →
  on_locus P :=
sorry

end NUMINAMATH_CALUDE_locus_of_P_l2376_237690


namespace NUMINAMATH_CALUDE_expression_value_l2376_237668

theorem expression_value (x y z w : ℝ) 
  (eq1 : 4 * x * z + y * w = 4) 
  (eq2 : x * w + y * z = 8) : 
  (2 * x + y) * (2 * z + w) = 20 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l2376_237668


namespace NUMINAMATH_CALUDE_six_digit_numbers_with_zero_six_digit_numbers_with_zero_count_l2376_237650

theorem six_digit_numbers_with_zero (total_six_digit : Nat) (six_digit_no_zero : Nat) : Nat :=
  by
  have h1 : total_six_digit = 900000 := by sorry
  have h2 : six_digit_no_zero = 531441 := by sorry
  have h3 : total_six_digit ≥ six_digit_no_zero := by sorry
  exact total_six_digit - six_digit_no_zero

theorem six_digit_numbers_with_zero_count :
    six_digit_numbers_with_zero 900000 531441 = 368559 :=
  by sorry

end NUMINAMATH_CALUDE_six_digit_numbers_with_zero_six_digit_numbers_with_zero_count_l2376_237650


namespace NUMINAMATH_CALUDE_lcm_of_9_12_15_l2376_237619

theorem lcm_of_9_12_15 : Nat.lcm (Nat.lcm 9 12) 15 = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_9_12_15_l2376_237619


namespace NUMINAMATH_CALUDE_simplify_sqrt_difference_l2376_237609

theorem simplify_sqrt_difference : 
  (Real.sqrt 648 / Real.sqrt 81) - (Real.sqrt 294 / Real.sqrt 49) = 2 * Real.sqrt 2 - Real.sqrt 42 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_difference_l2376_237609


namespace NUMINAMATH_CALUDE_smallest_valid_k_l2376_237652

def sum_to(m : ℕ) : ℕ := m * (m + 1) / 2

def is_valid_k(k : ℕ) : Prop :=
  ∃ n : ℕ, n > k ∧ sum_to k = sum_to n - sum_to k

theorem smallest_valid_k :
  (∀ k : ℕ, k > 6 ∧ k < 9 → ¬is_valid_k k) ∧
  is_valid_k 9 :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_k_l2376_237652


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l2376_237689

theorem complex_number_quadrant : ∃ (z : ℂ), z = 2 / (1 - Complex.I) ∧ z.re > 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l2376_237689


namespace NUMINAMATH_CALUDE_family_average_age_l2376_237601

theorem family_average_age
  (num_members : ℕ)
  (youngest_age : ℕ)
  (birth_average_age : ℚ)
  (h1 : num_members = 5)
  (h2 : youngest_age = 10)
  (h3 : birth_average_age = 12.5) :
  (birth_average_age * (num_members - 1) + youngest_age * num_members) / num_members = 20 :=
by sorry

end NUMINAMATH_CALUDE_family_average_age_l2376_237601


namespace NUMINAMATH_CALUDE_williams_tips_l2376_237642

/-- Williams works at a resort for 7 months. Let A be the average monthly tips for 6 of these months.
In August, he made 8 times the average of the other months. -/
theorem williams_tips (A : ℚ) : 
  let august_tips := 8 * A
  let total_tips := 15 * A
  august_tips / total_tips = 8 / 15 := by sorry

end NUMINAMATH_CALUDE_williams_tips_l2376_237642


namespace NUMINAMATH_CALUDE_tangent_line_at_point_p_l2376_237610

/-- The circle equation -/
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 4*x + m*y = 0

/-- Point P is on the circle -/
def point_on_circle (m : ℝ) : Prop :=
  circle_equation 1 1 m

/-- The tangent line equation -/
def tangent_line_equation (x y : ℝ) : Prop :=
  x - 2*y + 1 = 0

/-- Theorem: The equation of the tangent line at point P(1,1) on the given circle is x - 2y + 1 = 0 -/
theorem tangent_line_at_point_p :
  ∃ m : ℝ, point_on_circle m →
  ∀ x y : ℝ, (x = 1 ∧ y = 1) →
  tangent_line_equation x y :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_point_p_l2376_237610


namespace NUMINAMATH_CALUDE_cube_difference_multiple_implies_sum_squares_multiple_of_sum_l2376_237634

theorem cube_difference_multiple_implies_sum_squares_multiple_of_sum
  (a b c : ℕ+)
  (ha : a < 2017)
  (hb : b < 2017)
  (hc : c < 2017)
  (hab : a ≠ b)
  (hbc : b ≠ c)
  (hca : c ≠ a)
  (hab_multiple : ∃ k : ℤ, (a ^ 3 : ℤ) - (b ^ 3 : ℤ) = k * 2017)
  (hbc_multiple : ∃ k : ℤ, (b ^ 3 : ℤ) - (c ^ 3 : ℤ) = k * 2017)
  (hca_multiple : ∃ k : ℤ, (c ^ 3 : ℤ) - (a ^ 3 : ℤ) = k * 2017) :
  ∃ m : ℕ, (a ^ 2 + b ^ 2 + c ^ 2 : ℕ) = m * (a + b + c) :=
by sorry

end NUMINAMATH_CALUDE_cube_difference_multiple_implies_sum_squares_multiple_of_sum_l2376_237634


namespace NUMINAMATH_CALUDE_cookie_difference_l2376_237622

theorem cookie_difference (paul_cookies : ℕ) (total_cookies : ℕ) (paula_cookies : ℕ) : 
  paul_cookies = 45 → 
  total_cookies = 87 → 
  paula_cookies < paul_cookies →
  paul_cookies + paula_cookies = total_cookies →
  paul_cookies - paula_cookies = 3 := by
sorry

end NUMINAMATH_CALUDE_cookie_difference_l2376_237622


namespace NUMINAMATH_CALUDE_thirty_fifth_digit_of_sum_one_ninth_one_fifth_l2376_237698

/-- The decimal representation of a rational number -/
def decimal_rep (q : ℚ) : ℕ → ℕ := sorry

/-- The sum of decimal representations of two rational numbers -/
def sum_decimal_rep (q₁ q₂ : ℚ) : ℕ → ℕ := sorry

/-- Theorem: The 35th digit after the decimal point of the sum of 1/9 and 1/5 is 3 -/
theorem thirty_fifth_digit_of_sum_one_ninth_one_fifth : 
  sum_decimal_rep (1/9) (1/5) 35 = 3 := by sorry

end NUMINAMATH_CALUDE_thirty_fifth_digit_of_sum_one_ninth_one_fifth_l2376_237698


namespace NUMINAMATH_CALUDE_remainder_problem_l2376_237633

theorem remainder_problem (k : ℕ) 
  (h1 : k > 0) 
  (h2 : k < 41) 
  (h3 : k % 5 = 2) 
  (h4 : k % 6 = 5) : 
  k % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2376_237633


namespace NUMINAMATH_CALUDE_julia_money_left_l2376_237656

theorem julia_money_left (initial_amount : ℚ) : initial_amount = 40 →
  let after_game := initial_amount / 2
  let after_purchases := after_game - (after_game / 4)
  after_purchases = 15 := by
sorry

end NUMINAMATH_CALUDE_julia_money_left_l2376_237656


namespace NUMINAMATH_CALUDE_gcd_factorial_seven_eight_l2376_237605

theorem gcd_factorial_seven_eight : Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_seven_eight_l2376_237605


namespace NUMINAMATH_CALUDE_lauren_pencils_l2376_237629

/-- Proves that Lauren received 6 pencils given the conditions of the problem -/
theorem lauren_pencils (initial_pencils : ℕ) (remaining_pencils : ℕ) (matt_extra : ℕ) :
  initial_pencils = 24 →
  remaining_pencils = 9 →
  matt_extra = 3 →
  ∃ (lauren_pencils : ℕ),
    lauren_pencils + (lauren_pencils + matt_extra) = initial_pencils - remaining_pencils ∧
    lauren_pencils = 6 :=
by sorry

end NUMINAMATH_CALUDE_lauren_pencils_l2376_237629


namespace NUMINAMATH_CALUDE_blue_balls_in_jar_l2376_237651

theorem blue_balls_in_jar (total : ℕ) (blue : ℕ) (prob : ℚ) : 
  total = 12 →
  blue ≤ total →
  prob = 1 / 55 →
  (blue.choose 3 : ℚ) / (total.choose 3 : ℚ) = prob →
  blue = 4 := by
  sorry

end NUMINAMATH_CALUDE_blue_balls_in_jar_l2376_237651


namespace NUMINAMATH_CALUDE_candy_store_food_colouring_l2376_237638

/-- The amount of food colouring used by a candy store in one day -/
def total_food_colouring (lollipop_count : ℕ) (hard_candy_count : ℕ) 
  (lollipop_colouring : ℕ) (hard_candy_colouring : ℕ) : ℕ :=
  lollipop_count * lollipop_colouring + hard_candy_count * hard_candy_colouring

/-- Theorem stating the total amount of food colouring used by the candy store -/
theorem candy_store_food_colouring : 
  total_food_colouring 100 5 5 20 = 600 := by
  sorry

end NUMINAMATH_CALUDE_candy_store_food_colouring_l2376_237638


namespace NUMINAMATH_CALUDE_cubic_equation_root_l2376_237677

theorem cubic_equation_root (a b : ℚ) :
  (∃ x : ℂ, x^3 + a*x^2 + b*x + 15 = 0 ∧ x = 2 + Real.sqrt 5) →
  b = 29 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l2376_237677


namespace NUMINAMATH_CALUDE_percentage_girls_like_basketball_l2376_237671

/-- Given a class with the following properties:
  * There are 25 students in total
  * 60% of students are girls
  * 40% of boys like playing basketball
  * The number of girls who like basketball is double the number of boys who don't like it
  Prove that 80% of girls like playing basketball -/
theorem percentage_girls_like_basketball :
  ∀ (total_students : ℕ) 
    (girls boys boys_like_basketball boys_dont_like_basketball girls_like_basketball : ℕ),
  total_students = 25 →
  girls = (60 : ℕ) * total_students / 100 →
  boys = total_students - girls →
  boys_like_basketball = (40 : ℕ) * boys / 100 →
  boys_dont_like_basketball = boys - boys_like_basketball →
  girls_like_basketball = 2 * boys_dont_like_basketball →
  (girls_like_basketball : ℚ) / girls * 100 = 80 := by
sorry

end NUMINAMATH_CALUDE_percentage_girls_like_basketball_l2376_237671


namespace NUMINAMATH_CALUDE_cubic_integer_roots_imply_b_form_l2376_237665

theorem cubic_integer_roots_imply_b_form (a b : ℤ) 
  (h : ∃ (u v w : ℤ), u^3 - a*u^2 - b = 0 ∧ v^3 - a*v^2 - b = 0 ∧ w^3 - a*w^2 - b = 0) :
  ∃ (d k : ℤ), b = d * k^2 ∧ ∃ (m : ℤ), a = d * m :=
by sorry

end NUMINAMATH_CALUDE_cubic_integer_roots_imply_b_form_l2376_237665


namespace NUMINAMATH_CALUDE_inequality_proof_l2376_237658

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_one : a + b + c = 1) :
  (a^2 + b^2 + c^2) * (a / (b + c) + b / (a + c) + c / (a + b)) ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2376_237658


namespace NUMINAMATH_CALUDE_tenth_term_of_specific_geometric_sequence_l2376_237667

/-- Given a geometric sequence with first term a and common ratio r,
    the nth term is given by a * r^(n-1) -/
def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * r^(n-1)

/-- The tenth term of a geometric sequence with first term 5 and common ratio 3/4 -/
theorem tenth_term_of_specific_geometric_sequence :
  geometric_sequence 5 (3/4) 10 = 98415/262144 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_specific_geometric_sequence_l2376_237667


namespace NUMINAMATH_CALUDE_ladder_distance_l2376_237692

theorem ladder_distance (ladder_length height : ℝ) 
  (h1 : ladder_length = 13)
  (h2 : height = 12) :
  ∃ (base : ℝ), base^2 + height^2 = ladder_length^2 ∧ base = 5 := by
  sorry

end NUMINAMATH_CALUDE_ladder_distance_l2376_237692


namespace NUMINAMATH_CALUDE_new_plan_cost_l2376_237600

def old_plan_cost : ℝ := 150
def increase_percentage : ℝ := 0.3

theorem new_plan_cost : 
  old_plan_cost * (1 + increase_percentage) = 195 := by sorry

end NUMINAMATH_CALUDE_new_plan_cost_l2376_237600


namespace NUMINAMATH_CALUDE_parabola_focus_on_x_eq_one_l2376_237670

/-- A parabola is a conic section with a focus and directrix. -/
structure Parabola where
  /-- The focus of the parabola -/
  focus : ℝ × ℝ

/-- The standard form of a parabola equation -/
def standard_form (p : Parabola) : ℝ → ℝ → Prop :=
  fun x y => y^2 = 4 * (x - p.focus.1)

/-- Theorem: For a parabola with its focus on the line x = 1, its standard equation is y^2 = 4x -/
theorem parabola_focus_on_x_eq_one (p : Parabola) 
    (h : p.focus.1 = 1) : 
    ∀ x y, standard_form p x y ↔ y^2 = 4*x := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_on_x_eq_one_l2376_237670


namespace NUMINAMATH_CALUDE_largest_common_value_less_than_1000_l2376_237664

theorem largest_common_value_less_than_1000 :
  let seq1 := {a : ℕ | ∃ n : ℕ, a = 2 + 3 * n}
  let seq2 := {a : ℕ | ∃ m : ℕ, a = 4 + 8 * m}
  let common_values := seq1 ∩ seq2
  (∃ x ∈ common_values, x < 1000 ∧ ∀ y ∈ common_values, y < 1000 → y ≤ x) →
  (∃ x ∈ common_values, x = 980 ∧ ∀ y ∈ common_values, y < 1000 → y ≤ x) :=
by sorry

end NUMINAMATH_CALUDE_largest_common_value_less_than_1000_l2376_237664


namespace NUMINAMATH_CALUDE_monster_perimeter_l2376_237663

theorem monster_perimeter (r : ℝ) (θ : ℝ) : 
  r = 2 → θ = 270 * π / 180 → 
  r * θ + 2 * r = 3 * π + 4 := by sorry

end NUMINAMATH_CALUDE_monster_perimeter_l2376_237663


namespace NUMINAMATH_CALUDE_d_range_l2376_237644

/-- Circle C with center (3,4) and radius 1 -/
def CircleC (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 1

/-- Point A -/
def A : ℝ × ℝ := (0, 1)

/-- Point B -/
def B : ℝ × ℝ := (0, -1)

/-- Distance squared between two points -/
def distanceSquared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

/-- The function d for a point P on the circle -/
def d (x y : ℝ) : ℝ := distanceSquared (x, y) A + distanceSquared (x, y) B

theorem d_range :
  ∀ x y : ℝ, CircleC x y → 34 ≤ d x y ∧ d x y ≤ 74 :=
sorry

end NUMINAMATH_CALUDE_d_range_l2376_237644


namespace NUMINAMATH_CALUDE_no_equidistant_points_l2376_237683

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a plane, represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Configuration of a circle and two parallel tangent lines -/
structure CircleWithTangents where
  circle : Circle
  tangent1 : Line
  tangent2 : Line
  d : ℝ  -- distance from circle center to each tangent

/-- Predicate for a point being equidistant from a circle and a line -/
def isEquidistant (p : ℝ × ℝ) (c : Circle) (l : Line) : Prop := sorry

/-- Main theorem: No equidistant points exist when d > r -/
theorem no_equidistant_points (config : CircleWithTangents) 
  (h : config.d > config.circle.radius) :
  ¬∃ p : ℝ × ℝ, isEquidistant p config.circle config.tangent1 ∧ 
                isEquidistant p config.circle config.tangent2 := by
  sorry

end NUMINAMATH_CALUDE_no_equidistant_points_l2376_237683


namespace NUMINAMATH_CALUDE_sqrt_2023_bound_l2376_237654

theorem sqrt_2023_bound (n : ℤ) 
  (h1 : 43^2 = 1849)
  (h2 : 44^2 = 1936)
  (h3 : 45^2 = 2025)
  (h4 : 46^2 = 2116)
  (h5 : n < Real.sqrt 2023)
  (h6 : Real.sqrt 2023 < n + 1) : 
  n = 44 := by
sorry

end NUMINAMATH_CALUDE_sqrt_2023_bound_l2376_237654


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a9_l2376_237672

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a9 (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_sum : a 5 + a 7 = 16)
  (h_a3 : a 3 = 1) :
  a 9 = 15 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a9_l2376_237672


namespace NUMINAMATH_CALUDE_no_perfect_square_ends_2012_l2376_237613

theorem no_perfect_square_ends_2012 : ∀ a : ℤ, ¬(∃ k : ℤ, a^2 = 10000 * k + 2012) := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_ends_2012_l2376_237613


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l2376_237696

theorem smallest_solution_of_equation (y : ℝ) : 
  (3 * y^2 + 36 * y - 90 = y * (y + 18)) → y ≥ -15 := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l2376_237696


namespace NUMINAMATH_CALUDE_smaller_screen_diagonal_l2376_237657

theorem smaller_screen_diagonal (d : ℝ) : 
  d > 0 → d^2 / 2 = 200 - 38 → d = 18 := by
  sorry

end NUMINAMATH_CALUDE_smaller_screen_diagonal_l2376_237657


namespace NUMINAMATH_CALUDE_certain_number_is_six_l2376_237676

theorem certain_number_is_six (a b n : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a > b) 
  (h4 : a % n = 2) (h5 : b % n = 3) (h6 : (a - b) % n = 5) : n = 6 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_is_six_l2376_237676


namespace NUMINAMATH_CALUDE_find_y_value_l2376_237691

theorem find_y_value (a b x y : ℤ) : 
  (a + b + 100 + 200300 + x) / 5 = 250 →
  (a + b + 300 + 150100 + x + y) / 6 = 200 →
  a % 5 = 0 →
  b % 5 = 0 →
  y = 49800 := by
sorry

end NUMINAMATH_CALUDE_find_y_value_l2376_237691


namespace NUMINAMATH_CALUDE_qin_jiushao_triangle_area_l2376_237604

theorem qin_jiushao_triangle_area : 
  let a : ℝ := 5
  let b : ℝ := 6
  let c : ℝ := 7
  let S := Real.sqrt ((1/4) * (a^2 * b^2 - ((a^2 + b^2 - c^2)/2)^2))
  S = 6 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_qin_jiushao_triangle_area_l2376_237604


namespace NUMINAMATH_CALUDE_complex_equation_implies_ratio_l2376_237647

theorem complex_equation_implies_ratio (m n : ℝ) :
  (2 + m * Complex.I) * (n - 2 * Complex.I) = -4 - 3 * Complex.I →
  m / n = 1 :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_implies_ratio_l2376_237647


namespace NUMINAMATH_CALUDE_max_sum_on_circle_l2376_237615

theorem max_sum_on_circle (x y : ℤ) : 
  x > 0 → y > 0 → x^2 + y^2 = 49 → x + y ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_on_circle_l2376_237615


namespace NUMINAMATH_CALUDE_car_overtake_distance_l2376_237618

/-- Represents the distance between two cars -/
def distance_between_cars (v1 v2 t : ℝ) : ℝ := (v2 - v1) * t

/-- Theorem stating the distance between two cars under given conditions -/
theorem car_overtake_distance :
  let red_speed : ℝ := 30
  let black_speed : ℝ := 50
  let overtake_time : ℝ := 1
  distance_between_cars red_speed black_speed overtake_time = 20 := by
  sorry

end NUMINAMATH_CALUDE_car_overtake_distance_l2376_237618


namespace NUMINAMATH_CALUDE_workshop_professionals_l2376_237602

theorem workshop_professionals (total : ℕ) (laptops tablets coffee : ℕ)
  (laptops_and_tablets laptops_and_coffee tablets_and_coffee : ℕ)
  (all_three : ℕ) :
  total = 40 →
  laptops = 18 →
  tablets = 14 →
  coffee = 16 →
  laptops_and_tablets = 7 →
  laptops_and_coffee = 5 →
  tablets_and_coffee = 4 →
  all_three = 3 →
  total - (laptops + tablets + coffee -
    laptops_and_tablets - laptops_and_coffee - tablets_and_coffee + all_three) = 5 :=
by sorry

end NUMINAMATH_CALUDE_workshop_professionals_l2376_237602


namespace NUMINAMATH_CALUDE_sams_new_crime_books_l2376_237621

theorem sams_new_crime_books 
  (used_adventure : ℝ) 
  (used_mystery : ℝ) 
  (total_books : ℝ) 
  (h1 : used_adventure = 13.0)
  (h2 : used_mystery = 17.0)
  (h3 : total_books = 45.0) :
  total_books - (used_adventure + used_mystery) = 15.0 := by
  sorry

end NUMINAMATH_CALUDE_sams_new_crime_books_l2376_237621


namespace NUMINAMATH_CALUDE_binomial_expectation_five_l2376_237699

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- Probability mass function for a binomial distribution -/
def pmf (ξ : BinomialRV) (k : ℕ) : ℝ :=
  (ξ.n.choose k) * (ξ.p ^ k) * ((1 - ξ.p) ^ (ξ.n - k))

/-- Expected value of a binomial distribution -/
def expectation (ξ : BinomialRV) : ℝ := ξ.n * ξ.p

theorem binomial_expectation_five (ξ : BinomialRV) 
    (h_p : ξ.p = 1/2) 
    (h_pmf : pmf ξ 2 = 45 / 2^10) : 
  expectation ξ = 5 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expectation_five_l2376_237699


namespace NUMINAMATH_CALUDE_range_of_f_range_of_g_l2376_237678

-- Define the functions f and g
def f (a x : ℝ) : ℝ := x^2 + 4*a*x + 2*a + 6
def g (a : ℝ) : ℝ := 2 - a * |a + 3|

-- Theorem for part (1)
theorem range_of_f (a : ℝ) :
  (∀ y : ℝ, y ≥ 0 → ∃ x : ℝ, f a x = y) ∧ (∀ x : ℝ, f a x ≥ 0) ↔ a = -1 ∨ a = 3/2 :=
sorry

-- Theorem for part (2)
theorem range_of_g :
  (∀ a x : ℝ, f a x ≥ 0) →
  ∀ y : ℝ, -19/4 ≤ y ∧ y ≤ 4 ↔ ∃ a : ℝ, g a = y :=
sorry

end NUMINAMATH_CALUDE_range_of_f_range_of_g_l2376_237678


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2376_237630

theorem solution_set_quadratic_inequality (x : ℝ) :
  2 * x + 3 - x^2 > 0 ↔ -1 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2376_237630


namespace NUMINAMATH_CALUDE_arithmetic_simplification_l2376_237603

theorem arithmetic_simplification :
  (2 * Real.sqrt 12 - (1/2) * Real.sqrt 18) - (Real.sqrt 75 - (1/4) * Real.sqrt 32) = -Real.sqrt 3 - (1/2) * Real.sqrt 2 ∧
  (Real.sqrt 5 + 2) * (Real.sqrt 5 - 2) + Real.sqrt 48 / (2 * Real.sqrt (1/2)) - Real.sqrt 30 / Real.sqrt 5 = 1 + Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_simplification_l2376_237603


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_one_l2376_237659

theorem fraction_zero_implies_x_equals_one (x : ℝ) : 
  (x - 1) / (x + 3) = 0 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_one_l2376_237659


namespace NUMINAMATH_CALUDE_correct_division_result_l2376_237662

theorem correct_division_result (incorrect_divisor incorrect_quotient correct_divisor : ℕ) 
  (h1 : incorrect_divisor = 48)
  (h2 : incorrect_quotient = 24)
  (h3 : correct_divisor = 36) :
  (incorrect_divisor * incorrect_quotient) / correct_divisor = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_correct_division_result_l2376_237662


namespace NUMINAMATH_CALUDE_total_rectangles_in_diagram_l2376_237673

/-- Represents a rectangle in the diagram -/
structure Rectangle where
  id : Nat

/-- Represents the diagram with rectangles -/
structure Diagram where
  rectangles : List Rectangle

/-- Counts the number of unique rectangles in the diagram -/
def count_unique_rectangles (d : Diagram) : Nat :=
  d.rectangles.length

/-- Theorem stating the total number of unique rectangles in the specific diagram -/
theorem total_rectangles_in_diagram :
  ∃ (d : Diagram),
    (∃ (r1 r2 r3 : Rectangle), r1 ∈ d.rectangles ∧ r2 ∈ d.rectangles ∧ r3 ∈ d.rectangles) ∧  -- 3 large rectangles
    (∃ (r4 r5 r6 r7 : Rectangle), r4 ∈ d.rectangles ∧ r5 ∈ d.rectangles ∧ r6 ∈ d.rectangles ∧ r7 ∈ d.rectangles) ∧  -- 4 small rectangles
    (∀ (r s : Rectangle), r ∈ d.rectangles → s ∈ d.rectangles → ∃ (t : Rectangle), t ∈ d.rectangles) →  -- Combination of rectangles
    count_unique_rectangles d = 11 :=
by
  sorry


end NUMINAMATH_CALUDE_total_rectangles_in_diagram_l2376_237673


namespace NUMINAMATH_CALUDE_journey_time_increase_l2376_237666

theorem journey_time_increase (total_distance : ℝ) (first_half_speed : ℝ) (overall_speed : ℝ)
  (h1 : total_distance = 640)
  (h2 : first_half_speed = 80)
  (h3 : overall_speed = 40) :
  let first_half_distance := total_distance / 2
  let first_half_time := first_half_distance / first_half_speed
  let total_time := total_distance / overall_speed
  let second_half_time := total_time - first_half_time
  (second_half_time - first_half_time) / first_half_time = 2 := by
  sorry

end NUMINAMATH_CALUDE_journey_time_increase_l2376_237666


namespace NUMINAMATH_CALUDE_common_divisors_75_90_l2376_237687

theorem common_divisors_75_90 : ∃ (s : Finset Int), 
  (∀ x ∈ s, x ∣ 75 ∧ x ∣ 90) ∧ 
  (∀ y : Int, y ∣ 75 ∧ y ∣ 90 → y ∈ s) ∧ 
  s.card = 8 := by
  sorry

end NUMINAMATH_CALUDE_common_divisors_75_90_l2376_237687


namespace NUMINAMATH_CALUDE_largest_invertible_interval_for_f_l2376_237661

/-- The quadratic function f(x) = 3x^2 - 6x - 9 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 6 * x - 9

/-- The theorem stating that [1, ∞) is the largest interval containing x=2 where f is invertible -/
theorem largest_invertible_interval_for_f :
  ∃ (a : ℝ), a = 1 ∧ 
  (∀ x ∈ Set.Ici a, Function.Injective (f ∘ (λ t => t + a))) ∧
  (∀ b < a, ¬ Function.Injective (f ∘ (λ t => t + b))) ∧
  (2 ∈ Set.Ici a) :=
sorry

end NUMINAMATH_CALUDE_largest_invertible_interval_for_f_l2376_237661


namespace NUMINAMATH_CALUDE_triangle_property_l2376_237631

theorem triangle_property (a b c : ℝ) (A B C : ℝ) :
  b = 5 →
  c = 7 →
  (a + c) / b = (Real.sin B + Real.sin A) / (Real.sin C - Real.sin A) →
  C = 2 * Real.pi / 3 ∧
  (1 / 2) * a * b * Real.sin C = 15 * Real.sqrt 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_property_l2376_237631


namespace NUMINAMATH_CALUDE_sequence_product_l2376_237675

def is_arithmetic_sequence (x y z : ℝ) : Prop :=
  y - x = z - y

def is_geometric_sequence (a b c d e : ℝ) : Prop :=
  b / a = c / b ∧ c / b = d / c ∧ d / c = e / d

theorem sequence_product (a b m n : ℝ) :
  is_arithmetic_sequence (-9) a (-1) →
  is_geometric_sequence (-9) m b n (-1) →
  a * b = 15 := by
  sorry

end NUMINAMATH_CALUDE_sequence_product_l2376_237675


namespace NUMINAMATH_CALUDE_sqrt_sum_equality_l2376_237623

theorem sqrt_sum_equality : 
  Real.sqrt ((5 - 4 * Real.sqrt 2) ^ 2) + Real.sqrt ((5 + 4 * Real.sqrt 2) ^ 2) + 1 = 8 * Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equality_l2376_237623


namespace NUMINAMATH_CALUDE_fiona_hoodies_l2376_237620

theorem fiona_hoodies (total : ℕ) (casey_extra : ℕ) : 
  total = 8 → casey_extra = 2 → ∃ (fiona : ℕ), 
    fiona + (fiona + casey_extra) = total ∧ fiona = 3 := by
  sorry

end NUMINAMATH_CALUDE_fiona_hoodies_l2376_237620


namespace NUMINAMATH_CALUDE_mother_three_times_daughter_age_l2376_237608

/-- Proves that the number of years until the mother is three times as old as her daughter is 9,
    given that the mother is currently 42 years old and the daughter is currently 8 years old. -/
theorem mother_three_times_daughter_age (mother_age : ℕ) (daughter_age : ℕ) 
  (h1 : mother_age = 42) (h2 : daughter_age = 8) : 
  ∃ (years : ℕ), mother_age + years = 3 * (daughter_age + years) ∧ years = 9 := by
  sorry

end NUMINAMATH_CALUDE_mother_three_times_daughter_age_l2376_237608


namespace NUMINAMATH_CALUDE_farm_legs_count_l2376_237606

/-- The number of legs for a given animal type -/
def legs_per_animal (animal : String) : ℕ :=
  match animal with
  | "chicken" => 2
  | "sheep" => 4
  | _ => 0

/-- The total number of animals in the farm -/
def total_animals : ℕ := 20

/-- The number of sheep in the farm -/
def num_sheep : ℕ := 10

/-- The number of chickens in the farm -/
def num_chickens : ℕ := total_animals - num_sheep

theorem farm_legs_count : 
  (num_sheep * legs_per_animal "sheep") + (num_chickens * legs_per_animal "chicken") = 60 := by
  sorry

end NUMINAMATH_CALUDE_farm_legs_count_l2376_237606
