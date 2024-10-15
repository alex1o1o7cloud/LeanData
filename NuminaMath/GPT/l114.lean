import Mathlib

namespace NUMINAMATH_GPT_total_copies_l114_11463

theorem total_copies (rate1 : ℕ) (rate2 : ℕ) (time : ℕ) (total : ℕ) 
  (h1 : rate1 = 25) (h2 : rate2 = 55) (h3 : time = 30) : 
  total = rate1 * time + rate2 * time := 
  sorry

end NUMINAMATH_GPT_total_copies_l114_11463


namespace NUMINAMATH_GPT_intersect_A_B_complement_l114_11452

-- Define the sets A and B
def A := {x : ℝ | -1 < x ∧ x < 2}
def B := {x : ℝ | x > 1}

-- Find the complement of B in ℝ
def B_complement := {x : ℝ | x ≤ 1}

-- Prove that the intersection of A and the complement of B is equal to (-1, 1]
theorem intersect_A_B_complement : A ∩ B_complement = {x : ℝ | -1 < x ∧ x ≤ 1} :=
by
  -- Proof is to be provided
  sorry

end NUMINAMATH_GPT_intersect_A_B_complement_l114_11452


namespace NUMINAMATH_GPT_semicircle_parametric_equation_correct_l114_11413

-- Define the conditions of the problem in terms of Lean definitions and propositions.

def semicircle_parametric_equation : Prop :=
  ∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ (Real.pi / 2) →
    ∃ α : ℝ, α = 2 * θ ∧ 0 ≤ α ∧ α ≤ Real.pi ∧
    (∃ (x y : ℝ), x = 1 + Real.cos α ∧ y = Real.sin α)

-- Statement that we will prove
theorem semicircle_parametric_equation_correct : semicircle_parametric_equation :=
  sorry

end NUMINAMATH_GPT_semicircle_parametric_equation_correct_l114_11413


namespace NUMINAMATH_GPT_chips_probability_l114_11427

def total_chips : ℕ := 12
def blue_chips : ℕ := 4
def green_chips : ℕ := 3
def red_chips : ℕ := 5

def total_ways : ℕ := Nat.factorial total_chips

def blue_group_ways : ℕ := Nat.factorial blue_chips
def green_group_ways : ℕ := Nat.factorial green_chips
def red_group_ways : ℕ := Nat.factorial red_chips
def group_permutations : ℕ := Nat.factorial 3

def satisfying_arrangements : ℕ :=
  group_permutations * blue_group_ways * green_group_ways * red_group_ways

noncomputable def probability_of_event_B : ℚ :=
  (satisfying_arrangements : ℚ) / (total_ways : ℚ)

theorem chips_probability :
  probability_of_event_B = 1 / 4620 :=
by
  sorry

end NUMINAMATH_GPT_chips_probability_l114_11427


namespace NUMINAMATH_GPT_sector_central_angle_l114_11487

theorem sector_central_angle (r l α : ℝ) (h1 : 2 * r + l = 6) (h2 : 1/2 * l * r = 2) :
  α = l / r → (α = 1 ∨ α = 4) :=
by
  sorry

end NUMINAMATH_GPT_sector_central_angle_l114_11487


namespace NUMINAMATH_GPT_average_runs_in_30_matches_l114_11409

theorem average_runs_in_30_matches 
  (avg1 : ℕ) (matches1 : ℕ) (avg2 : ℕ) (matches2 : ℕ) (total_matches : ℕ)
  (h1 : avg1 = 40) (h2 : matches1 = 20) (h3 : avg2 = 13) (h4 : matches2 = 10) (h5 : total_matches = 30) :
  ((avg1 * matches1 + avg2 * matches2) / total_matches) = 31 := by
  sorry

end NUMINAMATH_GPT_average_runs_in_30_matches_l114_11409


namespace NUMINAMATH_GPT_solve_eq_solution_l114_11475

def eq_solution (x y : ℕ) : Prop := 3 ^ x = 2 ^ x * y + 1

theorem solve_eq_solution (x y : ℕ) (h1 : x > 0) (h2 : y > 0) : 
  eq_solution x y ↔ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 2) ∨ (x = 4 ∧ y = 5) :=
sorry

end NUMINAMATH_GPT_solve_eq_solution_l114_11475


namespace NUMINAMATH_GPT_spade_evaluation_l114_11489

def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spade_evaluation : spade 5 (spade 6 7 + 2) = -96 := by
  sorry

end NUMINAMATH_GPT_spade_evaluation_l114_11489


namespace NUMINAMATH_GPT_leak_empty_time_l114_11492

theorem leak_empty_time (P L : ℝ) (h1 : P = 1 / 6) (h2 : P - L = 1 / 12) : 1 / L = 12 :=
by
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_leak_empty_time_l114_11492


namespace NUMINAMATH_GPT_joe_two_different_fruits_in_a_day_l114_11448

def joe_meal_event : Type := {meal : ℕ // meal = 4}
def joe_fruit_choice : Type := {fruit : ℕ // fruit ≤ 4}

noncomputable def prob_all_same_fruit : ℚ := (1 / 4) ^ 4 * 4
noncomputable def prob_at_least_two_diff_fruits : ℚ := 1 - prob_all_same_fruit

theorem joe_two_different_fruits_in_a_day :
  prob_at_least_two_diff_fruits = 63 / 64 :=
by
  sorry

end NUMINAMATH_GPT_joe_two_different_fruits_in_a_day_l114_11448


namespace NUMINAMATH_GPT_inequality_solution_l114_11491

noncomputable def solution_set_inequality : Set ℝ := {x | -2 < x ∧ x < 1 / 3}

theorem inequality_solution :
  {x : ℝ | (2 * x - 1) / (3 * x + 1) > 1} = solution_set_inequality :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l114_11491


namespace NUMINAMATH_GPT_area_dodecagon_equals_rectangle_l114_11442

noncomputable def area_regular_dodecagon (r : ℝ) : ℝ := 3 * r^2

theorem area_dodecagon_equals_rectangle (r : ℝ) :
  let area_dodecagon := area_regular_dodecagon r
  let area_rectangle := r * (3 * r)
  area_dodecagon = area_rectangle :=
by
  let area_dodecagon := area_regular_dodecagon r
  let area_rectangle := r * (3 * r)
  show area_dodecagon = area_rectangle
  sorry

end NUMINAMATH_GPT_area_dodecagon_equals_rectangle_l114_11442


namespace NUMINAMATH_GPT_soccer_goal_difference_l114_11446

theorem soccer_goal_difference (n : ℕ) (h : n = 2020) :
  ¬ ∃ g : Fin n → ℤ,
    (∀ i j : Fin n, i < j → (g i < g j)) ∧ 
    (∀ i : Fin n, ∃ x y : ℕ, x + y = n - 1 ∧ 3 * x = (n - 1 - x) ∧ g i = x - y) :=
by
  sorry

end NUMINAMATH_GPT_soccer_goal_difference_l114_11446


namespace NUMINAMATH_GPT_correct_operation_l114_11480

theorem correct_operation (a b : ℝ) : a * b^2 - b^2 * a = 0 := by
  sorry

end NUMINAMATH_GPT_correct_operation_l114_11480


namespace NUMINAMATH_GPT_complete_square_l114_11498

theorem complete_square 
  (x : ℝ) : 
  (2 * x^2 - 3 * x - 1 = 0) → 
  ((x - (3/4))^2 = (17/16)) :=
sorry

end NUMINAMATH_GPT_complete_square_l114_11498


namespace NUMINAMATH_GPT_vehicle_speed_increase_l114_11453

/-- Vehicle dynamics details -/
structure Vehicle := 
  (initial_speed : ℝ) 
  (deceleration : ℝ)
  (initial_distance_from_A : ℝ)

/-- Given conditions -/
def conditions (A B C : Vehicle) : Prop :=
  A.initial_speed = 80 ∧
  B.initial_speed = 60 ∧
  C.initial_speed = 70 ∧ 
  C.deceleration = 2 ∧
  B.initial_distance_from_A = 40 ∧
  C.initial_distance_from_A = 260

/-- Prove A needs to increase its speed by 5 mph -/
theorem vehicle_speed_increase (A B C : Vehicle) (h : conditions A B C) : 
  ∃ dA : ℝ, dA = 5 ∧ A.initial_speed + dA > B.initial_speed → 
    (A.initial_distance_from_A / (A.initial_speed + dA - B.initial_speed)) < 
    (C.initial_distance_from_A / (A.initial_speed + dA + C.initial_speed - C.deceleration)) :=
sorry

end NUMINAMATH_GPT_vehicle_speed_increase_l114_11453


namespace NUMINAMATH_GPT_expression_change_l114_11450

variable (x b : ℝ)

-- The conditions
def expression (x : ℝ) : ℝ := x^3 - 5 * x + 1
def expr_change_plus (x b : ℝ) : ℝ := (x + b)^3 - 5 * (x + b) + 1
def expr_change_minus (x b : ℝ) : ℝ := (x - b)^3 - 5 * (x - b) + 1

-- The Lean statement to prove
theorem expression_change (h_b_pos : 0 < b) :
  expr_change_plus x b - expression x = 3 * b * x^2 + 3 * b^2 * x + b^3 - 5 * b ∨ 
  expr_change_minus x b - expression x = -3 * b * x^2 + 3 * b^2 * x - b^3 + 5 * b := 
by
  sorry

end NUMINAMATH_GPT_expression_change_l114_11450


namespace NUMINAMATH_GPT_problem_1_problem_2_l114_11411

def setA (x : ℝ) : Prop := 2 ≤ x ∧ x < 7
def setB (x : ℝ) : Prop := 3 < x ∧ x ≤ 10
def setC (a : ℝ) (x : ℝ) : Prop := a - 5 < x ∧ x < a

theorem problem_1 (x : ℝ) :
  (setA x ∧ setB x ↔ 3 < x ∧ x < 7) ∧
  (setA x ∨ setB x ↔ 2 ≤ x ∧ x ≤ 10) := 
by sorry

theorem problem_2 (a : ℝ) :
  (∀ x, setC a x → (2 ≤ x ∧ x ≤ 10)) ↔ (7 ≤ a ∧ a ≤ 10) :=
by sorry

end NUMINAMATH_GPT_problem_1_problem_2_l114_11411


namespace NUMINAMATH_GPT_find_smallest_n_l114_11403

theorem find_smallest_n (k : ℕ) (hk: 0 < k) :
        ∃ n : ℕ, (∀ (s : Finset ℤ), s.card = n → 
        ∃ (x y : ℤ), x ∈ s ∧ y ∈ s ∧ x ≠ y ∧ (x + y) % (2 * k) = 0 ∨ (x - y) % (2 * k) = 0) 
        ∧ n = k + 2 :=
sorry

end NUMINAMATH_GPT_find_smallest_n_l114_11403


namespace NUMINAMATH_GPT_michael_payment_correct_l114_11469

def suit_price : ℕ := 430
def suit_discount : ℕ := 100
def shoes_price : ℕ := 190
def shoes_discount : ℕ := 30
def shirt_price : ℕ := 80
def tie_price: ℕ := 50
def combined_discount : ℕ := (shirt_price + tie_price) * 20 / 100

def total_price_paid : ℕ :=
    suit_price - suit_discount + shoes_price - shoes_discount + (shirt_price + tie_price - combined_discount)

theorem michael_payment_correct :
    total_price_paid = 594 :=
by
    -- skipping the proof
    sorry

end NUMINAMATH_GPT_michael_payment_correct_l114_11469


namespace NUMINAMATH_GPT_length_of_AB_l114_11406

noncomputable def isosceles_triangle (a b c : ℕ) : Prop :=
  a = b ∨ a = c ∨ b = c

theorem length_of_AB 
  (a b c d e : ℕ)
  (h_iso_ABC : isosceles_triangle a b c)
  (h_iso_CDE : isosceles_triangle c d e)
  (h_perimeter_CDE : c + d + e = 25)
  (h_perimeter_ABC : a + b + c = 24)
  (h_CE : c = 9)
  (h_AB_DE : a = e) : a = 7 :=
by
  sorry

end NUMINAMATH_GPT_length_of_AB_l114_11406


namespace NUMINAMATH_GPT_triangle_inequality_l114_11431

theorem triangle_inequality (a : ℝ) (h₁ : a > 5) (h₂ : a < 19) : 5 < a ∧ a < 19 :=
by
  exact ⟨h₁, h₂⟩

end NUMINAMATH_GPT_triangle_inequality_l114_11431


namespace NUMINAMATH_GPT_playerA_winning_strategy_playerB_winning_strategy_no_winning_strategy_l114_11495

def hasWinningStrategyA (n : ℕ) : Prop :=
  n ≥ 8

def hasWinningStrategyB (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5

def draw (n : ℕ) : Prop :=
  n = 6 ∨ n = 7

theorem playerA_winning_strategy (n : ℕ) : n ≥ 8 → hasWinningStrategyA n :=
by
  sorry

theorem playerB_winning_strategy (n : ℕ) : (n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5) → hasWinningStrategyB n :=
by
  sorry

theorem no_winning_strategy (n : ℕ) : n = 6 ∨ n = 7 → draw n :=
by
  sorry

end NUMINAMATH_GPT_playerA_winning_strategy_playerB_winning_strategy_no_winning_strategy_l114_11495


namespace NUMINAMATH_GPT_common_tangent_l114_11466

-- Definition of the ellipse and hyperbola
def ellipse (x y : ℝ) : Prop := 9 * x^2 + 16 * y^2 = 144
def hyperbola (x y : ℝ) : Prop := 7 * x^2 - 32 * y^2 = 224

-- The statement to prove
theorem common_tangent :
  (∀ x y : ℝ, ellipse x y → hyperbola x y → ((x + y + 5 = 0) ∨ (x + y - 5 = 0) ∨ (x - y + 5 = 0) ∨ (x - y - 5 = 0))) := 
sorry

end NUMINAMATH_GPT_common_tangent_l114_11466


namespace NUMINAMATH_GPT_shaded_area_is_20_l114_11444

-- Represents the square PQRS with the necessary labeled side lengths
noncomputable def square_side_length : ℝ := 8

-- Represents the four labeled smaller squares' positions and their side lengths
noncomputable def smaller_square_side_lengths : List ℝ := [2, 2, 2, 6]

-- The coordinates or relations to describe their overlaying positions are not needed for the proof.

-- Define the calculated areas from the solution steps
noncomputable def vertical_rectangle_area : ℝ := 6 * 2
noncomputable def horizontal_rectangle_area : ℝ := 6 * 2
noncomputable def overlap_area : ℝ := 2 * 2

-- The total shaded T-shaped region area calculation
noncomputable def total_shaded_area : ℝ := vertical_rectangle_area + horizontal_rectangle_area - overlap_area

-- Theorem statement to prove the area of the T-shaped region is 20
theorem shaded_area_is_20 : total_shaded_area = 20 :=
by
  -- Proof steps are not required as per the instruction.
  sorry

end NUMINAMATH_GPT_shaded_area_is_20_l114_11444


namespace NUMINAMATH_GPT_initial_water_amount_gallons_l114_11496

theorem initial_water_amount_gallons 
  (cup_capacity_oz : ℕ)
  (rows : ℕ)
  (chairs_per_row : ℕ)
  (water_left_oz : ℕ)
  (oz_per_gallon : ℕ)
  (total_gallons : ℕ)
  (h1 : cup_capacity_oz = 6)
  (h2 : rows = 5)
  (h3 : chairs_per_row = 10)
  (h4 : water_left_oz = 84)
  (h5 : oz_per_gallon = 128)
  (h6 : total_gallons = (rows * chairs_per_row * cup_capacity_oz + water_left_oz) / oz_per_gallon) :
  total_gallons = 3 := 
by sorry

end NUMINAMATH_GPT_initial_water_amount_gallons_l114_11496


namespace NUMINAMATH_GPT_incorrect_transformation_is_not_valid_l114_11416

-- Define the system of linear equations
def eq1 (x y : ℝ) := 2 * x + y = 5
def eq2 (x y : ℝ) := 3 * x + 4 * y = 7

-- The definition of the correct transformation for x from equation eq2
def correct_transformation (x y : ℝ) := x = (7 - 4 * y) / 3

-- The definition of the incorrect transformation for x from equation eq2
def incorrect_transformation (x y : ℝ) := x = (7 + 4 * y) / 3

theorem incorrect_transformation_is_not_valid (x y : ℝ) 
  (h1 : eq1 x y) 
  (h2 : eq2 x y) :
  ¬ incorrect_transformation x y := 
by
  sorry

end NUMINAMATH_GPT_incorrect_transformation_is_not_valid_l114_11416


namespace NUMINAMATH_GPT_math_proof_problem_l114_11405

noncomputable def f (a b : ℚ) : ℝ := sorry

axiom f_cond1 (a b c : ℚ) : f (a * b) c = f a c * f b c ∧ f c (a * b) = f c a * f c b
axiom f_cond2 (a : ℚ) : f a (1 - a) = 1

theorem math_proof_problem (a b : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (f a a = 1) ∧ 
  (f a (-a) = 1) ∧
  (f a b * f b a = 1) := 
by 
  sorry

end NUMINAMATH_GPT_math_proof_problem_l114_11405


namespace NUMINAMATH_GPT_product_of_slopes_l114_11477

theorem product_of_slopes (p : ℝ) (hp : 0 < p) :
  let T := (p, 0)
  let parabola := fun x y => y^2 = 2*p*x
  let line := fun x y => y = x - p
  -- Define intersection points A and B on the parabola satisfying the line equation
  ∃ A B : ℝ × ℝ, 
  parabola A.1 A.2 ∧ line A.1 A.2 ∧
  parabola B.1 B.2 ∧ line B.1 B.2 ∧
  -- O is the origin
  let O := (0, 0)
  -- define slope function
  let slope (P Q : ℝ × ℝ) := (Q.2 - P.2) / (Q.1 - P.1)
  -- slopes of OA and OB
  let k_OA := slope O A
  let k_OB := slope O B
  -- product of slopes
  k_OA * k_OB = -2 := sorry

end NUMINAMATH_GPT_product_of_slopes_l114_11477


namespace NUMINAMATH_GPT_problem1_problem2_l114_11443

def f (x : ℝ) (a : ℝ) : ℝ := x^3 - a * x^2 - 3 * x

theorem problem1 (a : ℝ) : (∀ x : ℝ, x ≥ 1 → 3 * x^2 - 2 * a * x - 3 ≥ 0) → a ≤ 0 :=
sorry

theorem problem2 (a : ℝ) (h : a = 6) :
  x = 3 ∧ (∀ x : ℝ, 1 ≤ x ∧ x ≤ 6 → f x 6 ≤ -6 ∧ f x 6 ≥ -18) :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l114_11443


namespace NUMINAMATH_GPT_bhishma_speed_l114_11456

-- Given definitions based on conditions
def track_length : ℝ := 600
def bruce_speed : ℝ := 30
def time_meet : ℝ := 90

-- Main theorem we want to prove
theorem bhishma_speed : ∃ v : ℝ, v = 23.33 ∧ (bruce_speed * time_meet) = (v * time_meet + track_length) :=
  by
    sorry

end NUMINAMATH_GPT_bhishma_speed_l114_11456


namespace NUMINAMATH_GPT_abs_inequality_solution_l114_11457

theorem abs_inequality_solution (x : ℝ) : 
  abs (2 * x - 1) < abs x + 1 ↔ 0 < x ∧ x < 2 :=
by
  sorry

end NUMINAMATH_GPT_abs_inequality_solution_l114_11457


namespace NUMINAMATH_GPT_find_n_l114_11488

theorem find_n : ∃ (n : ℤ), -150 < n ∧ n < 150 ∧ Real.tan (n * Real.pi / 180) = Real.tan (1600 * Real.pi / 180) :=
sorry

end NUMINAMATH_GPT_find_n_l114_11488


namespace NUMINAMATH_GPT_polynomial_horner_form_operations_l114_11454

noncomputable def horner_eval (coeffs : List ℕ) (x : ℕ) : ℕ :=
  coeffs.foldr (fun a acc => a + acc * x) 0

theorem polynomial_horner_form_operations :
  let p := [1, 1, 2, 3, 4, 5]
  let x := 2
  horner_eval p x = ((((5 * x + 4) * x + 3) * x + 2) * x + 1) * x + 1 ∧
  (∀ x, x = 2 → (((((5 * x + 4) * x + 3) * x + 2) * x + 1) * x + 1 =  5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + 1 * x + 1)) ∧ 
  (∃ m a, m = 5 ∧ a = 5) := sorry

end NUMINAMATH_GPT_polynomial_horner_form_operations_l114_11454


namespace NUMINAMATH_GPT_find_weekday_rate_l114_11439

-- Definitions of given conditions
def num_people : ℕ := 6
def days_weekdays : ℕ := 2
def days_weekend : ℕ := 2
def weekend_rate : ℕ := 540
def payment_per_person : ℕ := 320

-- Theorem to prove the weekday rental rate
theorem find_weekday_rate (W : ℕ) :
  (num_people * payment_per_person) = (days_weekdays * W) + (days_weekend * weekend_rate) →
  W = 420 :=
by 
  intros h
  sorry

end NUMINAMATH_GPT_find_weekday_rate_l114_11439


namespace NUMINAMATH_GPT_difference_between_c_and_a_l114_11410

variables (a b c : ℝ)

theorem difference_between_c_and_a
  (h1 : (a + b) / 2 = 45)
  (h2 : (b + c) / 2 = 90) :
  c - a = 90 := by
  sorry

end NUMINAMATH_GPT_difference_between_c_and_a_l114_11410


namespace NUMINAMATH_GPT_smallest_integer_in_set_l114_11464

theorem smallest_integer_in_set (median : ℤ) (greatest : ℤ) (h1 : median = 144) (h2 : greatest = 153) : ∃ x : ℤ, x = 135 :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_in_set_l114_11464


namespace NUMINAMATH_GPT_find_non_negative_integers_l114_11499

def has_exactly_two_distinct_solutions (a : ℕ) (m : ℕ) : Prop :=
  ∃ (x₁ x₂ : ℕ), (x₁ < m) ∧ (x₂ < m) ∧ (x₁ ≠ x₂) ∧ (x₁^2 + a) % m = 0 ∧ (x₂^2 + a) % m = 0

theorem find_non_negative_integers (a : ℕ) (m : ℕ := 2007) : 
  a < m ∧ has_exactly_two_distinct_solutions a m ↔ a = 446 ∨ a = 1115 ∨ a = 1784 :=
sorry

end NUMINAMATH_GPT_find_non_negative_integers_l114_11499


namespace NUMINAMATH_GPT_composite_2011_2014_composite_2012_2015_l114_11447

theorem composite_2011_2014 :
  let N := 2011 * 2012 * 2013 * 2014 + 1
  ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ N = a * b := 
by
  let N := 2011 * 2012 * 2013 * 2014 + 1
  sorry
  
theorem composite_2012_2015 :
  let N := 2012 * 2013 * 2014 * 2015 + 1
  ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ N = a * b := 
by
  let N := 2012 * 2013 * 2014 * 2015 + 1
  sorry

end NUMINAMATH_GPT_composite_2011_2014_composite_2012_2015_l114_11447


namespace NUMINAMATH_GPT_arithmetic_sequence_a10_l114_11423

theorem arithmetic_sequence_a10 
  (S : ℕ → ℕ)
  (a : ℕ → ℕ)
  (h_seq : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2)
  (h_S4 : S 4 = 10)
  (h_S9 : S 9 = 45) :
  a 10 = 10 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_a10_l114_11423


namespace NUMINAMATH_GPT_value_of_abs_sum_l114_11433

noncomputable def cos_squared (θ : ℝ) : ℝ := (Real.cos θ) ^ 2

theorem value_of_abs_sum (θ x : ℝ) (h : Real.log x / Real.log 2 = 3 - 2 * cos_squared θ) :
  |x - 2| + |x - 8| = 6 := by
    sorry

end NUMINAMATH_GPT_value_of_abs_sum_l114_11433


namespace NUMINAMATH_GPT_abs_floor_value_l114_11474

theorem abs_floor_value : (Int.floor (|(-56.3: Real)|)) = 56 := 
by
  sorry

end NUMINAMATH_GPT_abs_floor_value_l114_11474


namespace NUMINAMATH_GPT_find_number_l114_11408

theorem find_number
  (a b c : ℕ)
  (h1 : 0 ≤ a ∧ a ≤ 9)
  (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : 0 ≤ c ∧ c ≤ 9)
  (h4 : 328 - (100 * a + 10 * b + c) = a + b + c) :
  100 * a + 10 * b + c = 317 :=
sorry

end NUMINAMATH_GPT_find_number_l114_11408


namespace NUMINAMATH_GPT_washer_dryer_cost_diff_l114_11490

-- conditions
def total_cost : ℕ := 1200
def washer_cost : ℕ := 710
def dryer_cost : ℕ := total_cost - washer_cost

-- proof statement
theorem washer_dryer_cost_diff : (washer_cost - dryer_cost) = 220 :=
by
  sorry

end NUMINAMATH_GPT_washer_dryer_cost_diff_l114_11490


namespace NUMINAMATH_GPT_greatest_integer_a_for_domain_of_expression_l114_11432

theorem greatest_integer_a_for_domain_of_expression :
  ∃ a : ℤ, (a^2 < 60 ∧ (∀ b : ℤ, b^2 < 60 → b ≤ a)) :=
  sorry

end NUMINAMATH_GPT_greatest_integer_a_for_domain_of_expression_l114_11432


namespace NUMINAMATH_GPT_greatest_value_of_x_l114_11418

theorem greatest_value_of_x (x : ℕ) : (Nat.lcm (Nat.lcm x 12) 18 = 180) → x ≤ 180 :=
by
  sorry

end NUMINAMATH_GPT_greatest_value_of_x_l114_11418


namespace NUMINAMATH_GPT_range_of_a_l114_11468

theorem range_of_a (x a : ℝ) (h₁ : 0 < x) (h₂ : x < 2) (h₃ : a - 1 < x) (h₄ : x ≤ a) :
  1 ≤ a ∧ a < 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l114_11468


namespace NUMINAMATH_GPT_f_of_6_l114_11479

noncomputable def f (u : ℝ) : ℝ := 
  let x := (u + 2) / 4
  x^3 - x + 2

theorem f_of_6 : f 6 = 8 :=
by
  sorry

end NUMINAMATH_GPT_f_of_6_l114_11479


namespace NUMINAMATH_GPT_price_of_one_table_l114_11436

variable (C T : ℝ)

def cond1 := 2 * C + T = 0.6 * (C + 2 * T)
def cond2 := C + T = 60
def solution := T = 52.5

theorem price_of_one_table (h1 : cond1 C T) (h2 : cond2 C T) : solution T :=
by
  sorry

end NUMINAMATH_GPT_price_of_one_table_l114_11436


namespace NUMINAMATH_GPT_cups_of_rice_morning_l114_11486

variable (cupsMorning : Nat) -- Number of cups of rice Robbie eats in the morning
variable (cupsAfternoon : Nat := 2) -- Cups of rice in the afternoon
variable (cupsEvening : Nat := 5) -- Cups of rice in the evening
variable (fatPerCup : Nat := 10) -- Fat in grams per cup of rice
variable (weeklyFatIntake : Nat := 700) -- Total fat in grams per week

theorem cups_of_rice_morning :
  ((cupsMorning + cupsAfternoon + cupsEvening) * fatPerCup) = (weeklyFatIntake / 7) → cupsMorning = 3 :=
  by
    sorry

end NUMINAMATH_GPT_cups_of_rice_morning_l114_11486


namespace NUMINAMATH_GPT_reaction2_follows_markovnikov_l114_11425

-- Define Markovnikov's rule - applying to case with protic acid (HX) to an alkene.
def follows_markovnikov_rule (HX : String) (initial_molecule final_product : String) : Prop :=
  initial_molecule = "CH3-CH=CH2 + HBr" ∧ final_product = "CH3-CHBr-CH3"

-- Example reaction data
def reaction1_initial : String := "CH2=CH2 + Br2"
def reaction1_final : String := "CH2Br-CH2Br"

def reaction2_initial : String := "CH3-CH=CH2 + HBr"
def reaction2_final : String := "CH3-CHBr-CH3"

def reaction3_initial : String := "CH4 + Cl2"
def reaction3_final : String := "CH3Cl + HCl"

def reaction4_initial : String := "CH ≡ CH + HOH"
def reaction4_final : String := "CH3''-C-H"

-- Proof statement
theorem reaction2_follows_markovnikov : follows_markovnikov_rule "HBr" reaction2_initial reaction2_final := by
  sorry

end NUMINAMATH_GPT_reaction2_follows_markovnikov_l114_11425


namespace NUMINAMATH_GPT_intersection_of_M_and_complementN_l114_11462

def UniversalSet := Set ℝ
def setM : Set ℝ := {-1, 0, 1, 3}
def setN : Set ℝ := {x | x^2 - x - 2 ≥ 0}
def complementSetN : Set ℝ := {x | -1 < x ∧ x < 2}

theorem intersection_of_M_and_complementN :
  setM ∩ complementSetN = {0, 1} :=
sorry

end NUMINAMATH_GPT_intersection_of_M_and_complementN_l114_11462


namespace NUMINAMATH_GPT_simplify_expression_l114_11419

theorem simplify_expression (x : ℝ) : 3 * x + 4 * x^2 + 2 - (5 - 3 * x - 5 * x^2 + x^3) = -x^3 + 9 * x^2 + 6 * x - 3 :=
by
  sorry -- Proof is omitted.

end NUMINAMATH_GPT_simplify_expression_l114_11419


namespace NUMINAMATH_GPT_trigonometric_identity_l114_11424

theorem trigonometric_identity :
  8 * Real.cos (4 * Real.pi / 9) * Real.cos (2 * Real.pi / 9) * Real.cos (Real.pi / 9) = 1 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l114_11424


namespace NUMINAMATH_GPT_ellipse_eq_range_m_l114_11471

theorem ellipse_eq_range_m (m : ℝ) : 
  (∃ x y : ℝ, (x^2 / (m - 1) + y^2 / (3 - m) = 1)) ↔ (1 < m ∧ m < 2) ∨ (2 < m ∧ m < 3) :=
sorry

end NUMINAMATH_GPT_ellipse_eq_range_m_l114_11471


namespace NUMINAMATH_GPT_man_l114_11400

theorem man's_age (x : ℕ) : 6 * (x + 6) - 6 * (x - 6) = x → x = 72 :=
by
  sorry

end NUMINAMATH_GPT_man_l114_11400


namespace NUMINAMATH_GPT_royal_children_count_l114_11458

-- Defining the initial conditions
def king_age := 35
def queen_age := 35
def sons := 3
def daughters_min := 1
def initial_children_age := 35
def max_children := 20

-- Statement of the problem
theorem royal_children_count (d n C : ℕ) 
    (h1 : king_age = 35)
    (h2 : queen_age = 35)
    (h3 : sons = 3)
    (h4 : daughters_min ≥ 1)
    (h5 : initial_children_age = 35)
    (h6 : 70 + 2 * n = 35 + (d + sons) * n)
    (h7 : C = d + sons)
    (h8 : C ≤ max_children) : 
    C = 7 ∨ C = 9 := 
sorry

end NUMINAMATH_GPT_royal_children_count_l114_11458


namespace NUMINAMATH_GPT_find_ax5_plus_by5_l114_11459

variable (a b x y : ℝ)

-- Conditions
axiom h1 : a * x + b * y = 3
axiom h2 : a * x^2 + b * y^2 = 7
axiom h3 : a * x^3 + b * y^3 = 16
axiom h4 : a * x^4 + b * y^4 = 42

-- Theorem (what we need to prove)
theorem find_ax5_plus_by5 : a * x^5 + b * y^5 = 20 :=
sorry

end NUMINAMATH_GPT_find_ax5_plus_by5_l114_11459


namespace NUMINAMATH_GPT_smallest_possible_bob_number_l114_11402

theorem smallest_possible_bob_number : 
  let alices_number := 60
  let bobs_smallest_number := 30
  ∃ (bob_number : ℕ), (∀ p : ℕ, Prime p → p ∣ alices_number → p ∣ bob_number) ∧ bob_number = bobs_smallest_number :=
by
  sorry

end NUMINAMATH_GPT_smallest_possible_bob_number_l114_11402


namespace NUMINAMATH_GPT_min_lcm_leq_six_floor_l114_11455

theorem min_lcm_leq_six_floor (n : ℕ) (h : n ≠ 4) (a : Fin n → ℕ) 
  (h1 : ∀ i, 0 < a i ∧ a i ≤ 2 * n) : 
  ∃ i j, i < j ∧ Nat.lcm (a i) (a j) ≤ 6 * (n / 2 + 1) :=
by
  sorry

end NUMINAMATH_GPT_min_lcm_leq_six_floor_l114_11455


namespace NUMINAMATH_GPT_v2004_eq_1_l114_11441

def g (x: ℕ) : ℕ :=
  match x with
  | 1 => 5
  | 2 => 3
  | 3 => 1
  | 4 => 2
  | 5 => 4
  | _ => 0  -- assuming default value for undefined cases

def v : ℕ → ℕ
| 0     => 3
| (n+1) => g (v n + 1)

theorem v2004_eq_1 : v 2004 = 1 :=
  sorry

end NUMINAMATH_GPT_v2004_eq_1_l114_11441


namespace NUMINAMATH_GPT_number_of_valid_3_digit_numbers_l114_11484

theorem number_of_valid_3_digit_numbers : 
  ∃ (n : ℕ), 
    (∀ (h t u : ℕ), 
      (n = h * 100 + t * 10 + u) ∧ 
      (1 ≤ h ∧ h ≤ 9) ∧ 
      (0 ≤ t ∧ t ≤ 9) ∧ 
      (0 ≤ u ∧ u ≤ 9) ∧ 
      (u ≥ 3 * t)) →
      n = 198 := 
by
  sorry

end NUMINAMATH_GPT_number_of_valid_3_digit_numbers_l114_11484


namespace NUMINAMATH_GPT_remainder_2468135792_mod_101_l114_11494

theorem remainder_2468135792_mod_101 : 
  2468135792 % 101 = 47 := 
sorry

end NUMINAMATH_GPT_remainder_2468135792_mod_101_l114_11494


namespace NUMINAMATH_GPT_find_blue_beads_per_row_l114_11472

-- Given the conditions of the problem:
def number_of_purple_beads : ℕ := 50 * 20
def number_of_gold_beads : ℕ := 80
def total_cost : ℕ := 180

-- Define the main theorem to solve for the number of blue beads per row.
theorem find_blue_beads_per_row (x : ℕ) :
  (number_of_purple_beads + 40 * x + number_of_gold_beads = total_cost) → x = (total_cost - (number_of_purple_beads + number_of_gold_beads)) / 40 := 
by {
  -- Proof steps would go here
  sorry
}

end NUMINAMATH_GPT_find_blue_beads_per_row_l114_11472


namespace NUMINAMATH_GPT_factorize_difference_of_squares_l114_11417

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end NUMINAMATH_GPT_factorize_difference_of_squares_l114_11417


namespace NUMINAMATH_GPT_gcf_3150_7350_l114_11407

theorem gcf_3150_7350 : Nat.gcd 3150 7350 = 525 := by
  sorry

end NUMINAMATH_GPT_gcf_3150_7350_l114_11407


namespace NUMINAMATH_GPT_sum_of_first_53_odd_numbers_l114_11438

theorem sum_of_first_53_odd_numbers :
  let first_term := 1
  let last_term := first_term + (53 - 1) * 2
  let sum := 53 / 2 * (first_term + last_term)
  sum = 2809 :=
by
  let first_term := 1
  let last_term := first_term + (53 - 1) * 2
  have last_term_val : last_term = 105 := by
    sorry
  let sum := 53 / 2 * (first_term + last_term)
  have sum_val : sum = 2809 := by
    sorry
  exact sum_val

end NUMINAMATH_GPT_sum_of_first_53_odd_numbers_l114_11438


namespace NUMINAMATH_GPT_profit_percentage_l114_11434

theorem profit_percentage (CP SP : ℝ) (h : 18 * CP = 16 * SP) : 
  (SP - CP) / CP * 100 = 12.5 := by
sorry

end NUMINAMATH_GPT_profit_percentage_l114_11434


namespace NUMINAMATH_GPT_fraction_of_track_Scottsdale_to_Forest_Grove_l114_11437

def distance_between_Scottsdale_and_Sherbourne : ℝ := 200
def round_trip_duration : ℝ := 5
def time_Harsha_to_Sherbourne : ℝ := 2

theorem fraction_of_track_Scottsdale_to_Forest_Grove :
  ∃ f : ℝ, f = 1/5 ∧
    ∀ (d : ℝ) (t : ℝ) (h : ℝ),
    d = distance_between_Scottsdale_and_Sherbourne →
    t = round_trip_duration →
    h = time_Harsha_to_Sherbourne →
    (2.5 - h) / t = f :=
sorry

end NUMINAMATH_GPT_fraction_of_track_Scottsdale_to_Forest_Grove_l114_11437


namespace NUMINAMATH_GPT_find_number_l114_11440

theorem find_number:
  ∃ x: ℕ, (∃ k: ℕ, ∃ r: ℕ, 5 * (x + 3) = 8 * k + r ∧ k = 156 ∧ r = 2) ∧ x = 247 :=
by 
  sorry

end NUMINAMATH_GPT_find_number_l114_11440


namespace NUMINAMATH_GPT_possible_red_ball_draws_l114_11412

/-- 
Given two balls in a bag where one is white and the other is red, 
if a ball is drawn and returned, and then another ball is drawn, 
prove that the possible number of times a red ball is drawn is 0, 1, or 2.
-/
theorem possible_red_ball_draws : 
  (∀ balls : Finset (ℕ × ℕ), 
    balls = {(0, 1), (1, 0)} →
    ∀ draw1 draw2 : ℕ × ℕ, 
    draw1 ∈ balls →
    draw2 ∈ balls →
    ∃ n : ℕ, (n = 0 ∨ n = 1 ∨ n = 2) ∧ 
    n = (if draw1 = (1, 0) then 1 else 0) + 
        (if draw2 = (1, 0) then 1 else 0)) → 
    True := sorry

end NUMINAMATH_GPT_possible_red_ball_draws_l114_11412


namespace NUMINAMATH_GPT_repeating_decimal_to_fraction_l114_11421

theorem repeating_decimal_to_fraction (x : ℚ) (h : x = 0.3 + 56 / 9900) : x = 3969 / 11100 := 
sorry

end NUMINAMATH_GPT_repeating_decimal_to_fraction_l114_11421


namespace NUMINAMATH_GPT_inscribed_rectangle_sides_l114_11481

theorem inscribed_rectangle_sides {a b c : ℕ} (h₀ : a = 3) (h₁ : b = 4) (h₂ : c = 5) (ratio : ℚ) (h_ratio : ratio = 1 / 3) :
  ∃ (x y : ℚ), x = 20 / 29 ∧ y = 60 / 29 ∧ x = ratio * y :=
by
  sorry

end NUMINAMATH_GPT_inscribed_rectangle_sides_l114_11481


namespace NUMINAMATH_GPT_value_of_k_l114_11435

theorem value_of_k (k : ℕ) (h : 24 / k = 4) : k = 6 := by
  sorry

end NUMINAMATH_GPT_value_of_k_l114_11435


namespace NUMINAMATH_GPT_calculate_earths_atmosphere_mass_l114_11414

noncomputable def mass_of_earths_atmosphere (R p0 g : ℝ) : ℝ :=
  (4 * Real.pi * R^2 * p0) / g

theorem calculate_earths_atmosphere_mass (R p0 g : ℝ) (h : 0 < g) : 
  mass_of_earths_atmosphere R p0 g = 5 * 10^18 := 
sorry

end NUMINAMATH_GPT_calculate_earths_atmosphere_mass_l114_11414


namespace NUMINAMATH_GPT_symmetric_diff_cardinality_l114_11497

theorem symmetric_diff_cardinality (X Y : Finset ℤ) 
  (hX : X.card = 8) 
  (hY : Y.card = 10) 
  (hXY : (X ∩ Y).card = 6) : 
  (X \ Y ∪ Y \ X).card = 6 := 
by
  sorry

end NUMINAMATH_GPT_symmetric_diff_cardinality_l114_11497


namespace NUMINAMATH_GPT_difference_of_squares_example_l114_11461

theorem difference_of_squares_example : 169^2 - 168^2 = 337 :=
by
  -- The proof steps using the difference of squares formula is omitted here.
  sorry

end NUMINAMATH_GPT_difference_of_squares_example_l114_11461


namespace NUMINAMATH_GPT_mandy_more_than_three_friends_l114_11428

noncomputable def stickers_given_to_three_friends : ℕ := 4 * 3
noncomputable def total_initial_stickers : ℕ := 72
noncomputable def stickers_left : ℕ := 42
noncomputable def total_given_away : ℕ := total_initial_stickers - stickers_left
noncomputable def mandy_justin_total : ℕ := total_given_away - stickers_given_to_three_friends
noncomputable def mandy_stickers : ℕ := 14
noncomputable def three_friends_stickers : ℕ := stickers_given_to_three_friends

theorem mandy_more_than_three_friends : 
  mandy_stickers - three_friends_stickers = 2 :=
by
  sorry

end NUMINAMATH_GPT_mandy_more_than_three_friends_l114_11428


namespace NUMINAMATH_GPT_distance_to_post_office_l114_11429

variable (D : ℚ)
variable (rate_to_post : ℚ := 25)
variable (rate_back : ℚ := 4)
variable (total_time : ℚ := 5 + 48 / 60)

theorem distance_to_post_office : (D / rate_to_post + D / rate_back = total_time) → D = 20 := by
  sorry

end NUMINAMATH_GPT_distance_to_post_office_l114_11429


namespace NUMINAMATH_GPT_line_equation_through_point_and_area_l114_11493

theorem line_equation_through_point_and_area (k b : ℝ) :
  (∃ (P : ℝ × ℝ), P = (4/3, 2)) ∧
  (∀ (A B : ℝ × ℝ), A = (- b / k, 0) ∧ B = (0, b) → 
  1 / 2 * abs ((- b / k) * b) = 6) →
  (y = k * x + b ↔ (y = -3/4 * x + 3 ∨ y = -3 * x + 6)) :=
by
  sorry

end NUMINAMATH_GPT_line_equation_through_point_and_area_l114_11493


namespace NUMINAMATH_GPT_solve_for_y_l114_11422

theorem solve_for_y (y : ℝ) (h : (5 - 1 / y)^(1/3) = -3) : y = 1 / 32 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l114_11422


namespace NUMINAMATH_GPT_part_one_part_two_l114_11473

-- Part (1)
theorem part_one (a : ℝ) (h : a ≤ 2) (x : ℝ) :
  (|x - 1| + |x - a| ≥ 2 ↔ x ≤ 0.5 ∨ x ≥ 2.5) :=
sorry

-- Part (2)
theorem part_two (a : ℝ) (h1 : a > 1) (h2 : ∀ x : ℝ, |x - 1| + |x - a| + |x - 1| ≥ 1) :
  a ≥ 2 :=
sorry

end NUMINAMATH_GPT_part_one_part_two_l114_11473


namespace NUMINAMATH_GPT_parallel_lines_eq_a_l114_11478

theorem parallel_lines_eq_a (a : ℝ) :
  (∀ x y : ℝ, x + 2 * a * y - 1 = 0 → (a + 1) * x - a * y = 0) → (a = -3/2 ∨ a = 0) :=
by sorry

end NUMINAMATH_GPT_parallel_lines_eq_a_l114_11478


namespace NUMINAMATH_GPT_beef_weight_after_processing_l114_11476

noncomputable def initial_weight : ℝ := 840
noncomputable def lost_percentage : ℝ := 35
noncomputable def retained_percentage : ℝ := 100 - lost_percentage
noncomputable def final_weight : ℝ := retained_percentage / 100 * initial_weight

theorem beef_weight_after_processing : final_weight = 546 := by
  sorry

end NUMINAMATH_GPT_beef_weight_after_processing_l114_11476


namespace NUMINAMATH_GPT_eval_expression_l114_11420

theorem eval_expression : 10 * 1.8 - 2 * 1.5 / 0.3 = 8 := 
by
  sorry

end NUMINAMATH_GPT_eval_expression_l114_11420


namespace NUMINAMATH_GPT_an_gt_bn_l114_11485

theorem an_gt_bn (a b : ℕ → ℕ) (h₁ : a 1 = 2013) (h₂ : ∀ n, a (n + 1) = 2013^(a n))
                            (h₃ : b 1 = 1) (h₄ : ∀ n, b (n + 1) = 2013^(2012 * (b n))) :
  ∀ n, a n > b n := 
sorry

end NUMINAMATH_GPT_an_gt_bn_l114_11485


namespace NUMINAMATH_GPT_glued_cubes_surface_area_l114_11426

theorem glued_cubes_surface_area (L l : ℝ) (h1 : L = 2) (h2 : l = L / 2) : 
  6 * L^2 + 4 * l^2 = 28 :=
by
  sorry

end NUMINAMATH_GPT_glued_cubes_surface_area_l114_11426


namespace NUMINAMATH_GPT_muffin_cost_l114_11470

theorem muffin_cost (m : ℝ) :
  let fruit_cup_cost := 3
  let francis_cost := 2 * m + 2 * fruit_cup_cost
  let kiera_cost := 2 * m + 1 * fruit_cup_cost
  let total_cost := 17
  (francis_cost + kiera_cost = total_cost) → m = 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_muffin_cost_l114_11470


namespace NUMINAMATH_GPT_find_y_l114_11467

theorem find_y (x : ℤ) (y : ℤ) (h : x = 5) (h1 : 3 * x = (y - x) + 4) : y = 16 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l114_11467


namespace NUMINAMATH_GPT_sequence_solution_l114_11449

theorem sequence_solution 
  (a : ℕ → ℝ)
  (h1 : ∀ n, a (n + 1) = a n / (2 + a n))
  (h2 : a 1 = 1) :
  ∀ n, a n = 1 / (2^n - 1) :=
sorry

end NUMINAMATH_GPT_sequence_solution_l114_11449


namespace NUMINAMATH_GPT_solve_inequality_l114_11483

theorem solve_inequality : { x : ℝ | 3 * x^2 - 1 > 13 - 5 * x } = { x : ℝ | x < -7 ∨ x > 2 } :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l114_11483


namespace NUMINAMATH_GPT_dot_product_a_b_l114_11404

def a : ℝ × ℝ := (-2, 3)
def b : ℝ × ℝ := (1, 2)

theorem dot_product_a_b : a.1 * b.1 + a.2 * b.2 = 4 := by
  sorry

end NUMINAMATH_GPT_dot_product_a_b_l114_11404


namespace NUMINAMATH_GPT_parabola_vertex_l114_11482

theorem parabola_vertex (c d : ℝ) (h : ∀ x : ℝ, - x^2 + c * x + d ≤ 0 ↔ (x ≤ -4 ∨ x ≥ 6)) :
  (∃ v : ℝ × ℝ, v = (5, 1)) :=
sorry

end NUMINAMATH_GPT_parabola_vertex_l114_11482


namespace NUMINAMATH_GPT_trigonometric_sign_l114_11430

open Real

theorem trigonometric_sign :
  (0 < 1 ∧ 1 < π / 2) ∧ 
  (∀ x y, (0 ≤ x ∧ x ≤ y ∧ y ≤ π / 2 → sin x ≤ sin y)) ∧ 
  (∀ x y, (0 ≤ x ∧ x ≤ y ∧ y ≤ π / 2 → cos x ≥ cos y)) →
  (cos (cos 1) - cos 1) * (sin (sin 1) - sin 1) < 0 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_sign_l114_11430


namespace NUMINAMATH_GPT_rounded_diff_greater_l114_11445

variable (x y ε : ℝ)
variable (h1 : x > y)
variable (h2 : y > 0)
variable (h3 : ε > 0)

theorem rounded_diff_greater : (x + ε) - (y - ε) > x - y :=
  by
  sorry

end NUMINAMATH_GPT_rounded_diff_greater_l114_11445


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l114_11415

theorem isosceles_triangle_perimeter (a b : ℕ) (h_isosceles : a = 3 ∨ a = 7 ∨ b = 3 ∨ b = 7) (h_ineq1 : 3 + 3 ≤ b ∨ b + b ≤ 3) (h_ineq2 : 7 + 7 ≥ a ∨ a + a ≥ 7) :
  (a = 3 ∧ b = 7) → 3 + 7 + 7 = 17 :=
by
  -- To be completed
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l114_11415


namespace NUMINAMATH_GPT_circles_externally_tangent_l114_11465

noncomputable def circle1_center : ℝ × ℝ := (-1, 1)
noncomputable def circle1_radius : ℝ := 2
noncomputable def circle2_center : ℝ × ℝ := (2, -3)
noncomputable def circle2_radius : ℝ := 3

noncomputable def distance_centers : ℝ :=
  Real.sqrt ((circle1_center.1 - circle2_center.1)^2 + (circle1_center.2 - circle2_center.2)^2)

theorem circles_externally_tangent :
  distance_centers = circle1_radius + circle2_radius :=
by
  -- The proof will show that the distance between the centers is equal to the sum of the radii, 
  -- indicating they are externally tangent.
  sorry

end NUMINAMATH_GPT_circles_externally_tangent_l114_11465


namespace NUMINAMATH_GPT_abs_diff_eq_five_l114_11451

theorem abs_diff_eq_five (a b : ℝ) (h1 : a * b = 6) (h2 : a + b = 7) : |a - b| = 5 :=
by
  sorry

end NUMINAMATH_GPT_abs_diff_eq_five_l114_11451


namespace NUMINAMATH_GPT_probability_x_lt_2y_is_2_over_5_l114_11401

noncomputable def rectangle_area : ℝ :=
  5 * 2

noncomputable def triangle_area : ℝ :=
  1 / 2 * 4 * 2

noncomputable def probability_x_lt_2y : ℝ :=
  triangle_area / rectangle_area

theorem probability_x_lt_2y_is_2_over_5 :
  probability_x_lt_2y = 2 / 5 := by
  sorry

end NUMINAMATH_GPT_probability_x_lt_2y_is_2_over_5_l114_11401


namespace NUMINAMATH_GPT_a_divisible_by_11_iff_b_divisible_by_11_l114_11460

-- Define the relevant functions
def a (n : ℕ) : ℕ := n^5 + 5^n
def b (n : ℕ) : ℕ := n^5 * 5^n + 1

-- State that for a positive integer n, a(n) is divisible by 11 if and only if b(n) is also divisible by 11
theorem a_divisible_by_11_iff_b_divisible_by_11 (n : ℕ) (hn : 0 < n) : 
  (a n % 11 = 0) ↔ (b n % 11 = 0) :=
sorry

end NUMINAMATH_GPT_a_divisible_by_11_iff_b_divisible_by_11_l114_11460
