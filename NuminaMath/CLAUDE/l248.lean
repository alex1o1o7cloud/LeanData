import Mathlib

namespace NUMINAMATH_CALUDE_sin_sum_of_zero_points_l248_24893

/-- Given that x₁ and x₂ are two zero points of f(x) = 2sin(2x) + cos(2x) - m
    within the interval [0, π/2], prove that sin(x₁ + x₂) = 2√5/5 -/
theorem sin_sum_of_zero_points (x₁ x₂ m : ℝ) : 
  x₁ ∈ Set.Icc 0 (π/2) →
  x₂ ∈ Set.Icc 0 (π/2) →
  2 * Real.sin (2 * x₁) + Real.cos (2 * x₁) - m = 0 →
  2 * Real.sin (2 * x₂) + Real.cos (2 * x₂) - m = 0 →
  Real.sin (x₁ + x₂) = 2 * Real.sqrt 5 / 5 :=
by sorry

end NUMINAMATH_CALUDE_sin_sum_of_zero_points_l248_24893


namespace NUMINAMATH_CALUDE_train_cars_distribution_l248_24887

theorem train_cars_distribution (soldiers_train1 soldiers_train2 soldiers_train3 : ℕ) 
  (h1 : soldiers_train1 = 462)
  (h2 : soldiers_train2 = 546)
  (h3 : soldiers_train3 = 630) :
  let max_soldiers_per_car := Nat.gcd soldiers_train1 (Nat.gcd soldiers_train2 soldiers_train3)
  (cars_train1, cars_train2, cars_train3) = (soldiers_train1 / max_soldiers_per_car,
                                             soldiers_train2 / max_soldiers_per_car,
                                             soldiers_train3 / max_soldiers_per_car) →
  (cars_train1, cars_train2, cars_train3) = (11, 13, 15) := by
sorry

end NUMINAMATH_CALUDE_train_cars_distribution_l248_24887


namespace NUMINAMATH_CALUDE_max_quarters_sasha_l248_24805

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 25 / 100

/-- The value of a dime in dollars -/
def dime_value : ℚ := 10 / 100

/-- The total value Sasha has in dollars -/
def total_value : ℚ := 210 / 100

theorem max_quarters_sasha : 
  ∃ (q : ℕ), q * quarter_value + q * dime_value = total_value ∧ 
  ∀ (n : ℕ), n * quarter_value + n * dime_value = total_value → n ≤ q :=
by sorry

end NUMINAMATH_CALUDE_max_quarters_sasha_l248_24805


namespace NUMINAMATH_CALUDE_integral_one_plus_sin_x_l248_24836

theorem integral_one_plus_sin_x : ∫ x in (0)..(π/2), (1 + Real.sin x) = π/2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_integral_one_plus_sin_x_l248_24836


namespace NUMINAMATH_CALUDE_bake_sale_cookies_l248_24881

/-- The number of lemon cookies Marcus brought to the bake sale -/
def lemon_cookies : ℕ := 20

/-- The number of peanut butter cookies Jenny brought -/
def jenny_pb_cookies : ℕ := 40

/-- The number of chocolate chip cookies Jenny brought -/
def jenny_cc_cookies : ℕ := 50

/-- The number of peanut butter cookies Marcus brought -/
def marcus_pb_cookies : ℕ := 30

/-- The total number of cookies at the bake sale -/
def total_cookies : ℕ := jenny_pb_cookies + jenny_cc_cookies + marcus_pb_cookies + lemon_cookies

/-- The probability of picking a peanut butter cookie -/
def pb_probability : ℚ := 1/2

theorem bake_sale_cookies : 
  (jenny_pb_cookies + marcus_pb_cookies : ℚ) / total_cookies = pb_probability := by sorry

end NUMINAMATH_CALUDE_bake_sale_cookies_l248_24881


namespace NUMINAMATH_CALUDE_tysons_races_l248_24825

/-- Tyson's swimming races problem -/
theorem tysons_races (lake_speed ocean_speed race_distance total_time : ℝ) 
  (h1 : lake_speed = 3)
  (h2 : ocean_speed = 2.5)
  (h3 : race_distance = 3)
  (h4 : total_time = 11) : 
  ∃ (num_races : ℕ), 
    (num_races : ℝ) / 2 * (race_distance / lake_speed) + 
    (num_races : ℝ) / 2 * (race_distance / ocean_speed) = total_time ∧ 
    num_races = 10 := by
  sorry

#check tysons_races

end NUMINAMATH_CALUDE_tysons_races_l248_24825


namespace NUMINAMATH_CALUDE_nut_is_composed_of_prism_and_cylinder_l248_24852

-- Define the types for geometric bodies
inductive GeometricBody
| RegularHexagonalPrism
| Cylinder
| Nut

-- Define the composition of a nut
def nut_composition : List GeometricBody := [GeometricBody.RegularHexagonalPrism, GeometricBody.Cylinder]

-- Theorem statement
theorem nut_is_composed_of_prism_and_cylinder :
  (GeometricBody.Nut ∈ [GeometricBody.RegularHexagonalPrism, GeometricBody.Cylinder]) →
  (nut_composition.length = 2) →
  (nut_composition = [GeometricBody.RegularHexagonalPrism, GeometricBody.Cylinder]) :=
by sorry

end NUMINAMATH_CALUDE_nut_is_composed_of_prism_and_cylinder_l248_24852


namespace NUMINAMATH_CALUDE_curve_equation_min_distance_l248_24804

-- Define the points A and B
def A : ℝ × ℝ := (-3, 0)
def B : ℝ × ℝ := (3, 0)

-- Define the curve C
def C : Set (ℝ × ℝ) := {P | (P.1 - 5)^2 + P.2^2 = 16}

-- Define the line l1
def l1 : Set (ℝ × ℝ) := {Q | Q.1 + Q.2 + 3 = 0}

-- Theorem for the equation of curve C
theorem curve_equation :
  C = {P : ℝ × ℝ | Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) = 2 * Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2)} :=
sorry

-- Theorem for the minimum distance
theorem min_distance :
  ∀ Q ∈ l1, ∃ M ∈ C, ∀ M' ∈ C,
    (∃ t : ℝ, M' = (1 - t) • Q + t • M) →
    Real.sqrt ((M.1 - Q.1)^2 + (M.2 - Q.2)^2) ≤ Real.sqrt ((M'.1 - Q.1)^2 + (M'.2 - Q.2)^2) ∧
    Real.sqrt ((M.1 - Q.1)^2 + (M.2 - Q.2)^2) = 4 :=
sorry

end NUMINAMATH_CALUDE_curve_equation_min_distance_l248_24804


namespace NUMINAMATH_CALUDE_cube_three_minus_seven_equals_square_four_plus_four_l248_24865

theorem cube_three_minus_seven_equals_square_four_plus_four :
  3^3 - 7 = 4^2 + 4 := by
  sorry

end NUMINAMATH_CALUDE_cube_three_minus_seven_equals_square_four_plus_four_l248_24865


namespace NUMINAMATH_CALUDE_persimmons_in_jungkooks_house_l248_24899

theorem persimmons_in_jungkooks_house : 
  let num_boxes : ℕ := 4
  let persimmons_per_box : ℕ := 5
  num_boxes * persimmons_per_box = 20 := by
  sorry

end NUMINAMATH_CALUDE_persimmons_in_jungkooks_house_l248_24899


namespace NUMINAMATH_CALUDE_fourth_root_inequality_l248_24870

theorem fourth_root_inequality (x : ℝ) :
  (x ^ (1/4) - 3 / (x ^ (1/4) + 4) ≥ 0) ↔ (0 ≤ x ∧ x ≤ 81) :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_inequality_l248_24870


namespace NUMINAMATH_CALUDE_survey_sample_is_opinions_of_selected_parents_l248_24824

/-- Represents a parent of a student -/
structure Parent : Type :=
  (id : ℕ)

/-- Represents an opinion on the school rule -/
structure Opinion : Type :=
  (value : Bool)

/-- Represents a school survey -/
structure Survey : Type :=
  (participants : Finset Parent)
  (opinions : Parent → Option Opinion)

/-- Definition of a sample in the context of this survey -/
def sample (s : Survey) : Set Opinion :=
  {o | ∃ p ∈ s.participants, s.opinions p = some o}

theorem survey_sample_is_opinions_of_selected_parents 
  (s : Survey) 
  (h_size : s.participants.card = 100) :
  sample s = {o | ∃ p ∈ s.participants, s.opinions p = some o} :=
by
  sorry

#check survey_sample_is_opinions_of_selected_parents

end NUMINAMATH_CALUDE_survey_sample_is_opinions_of_selected_parents_l248_24824


namespace NUMINAMATH_CALUDE_bobby_candy_problem_l248_24878

/-- Proves that Bobby ate 9 pieces of candy at the start -/
theorem bobby_candy_problem (initial : ℕ) (eaten_start : ℕ) (eaten_more : ℕ) (left : ℕ)
  (h1 : initial = 22)
  (h2 : eaten_more = 5)
  (h3 : left = 8)
  (h4 : initial = eaten_start + eaten_more + left) :
  eaten_start = 9 := by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_problem_l248_24878


namespace NUMINAMATH_CALUDE_gcd_of_squares_sum_l248_24867

theorem gcd_of_squares_sum : Nat.gcd (100^2 + 221^2 + 320^2) (101^2 + 220^2 + 321^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_squares_sum_l248_24867


namespace NUMINAMATH_CALUDE_inequality_solution_l248_24831

theorem inequality_solution (x : ℝ) :
  x ≠ -4 ∧ x ≠ -10/3 →
  ((2*x + 3) / (x + 4) > (4*x + 5) / (3*x + 10) ↔ 
   x < -5/2 ∨ x > -2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l248_24831


namespace NUMINAMATH_CALUDE_ferry_position_after_202_trips_l248_24864

/-- Represents the shore where the ferry can be docked --/
inductive Shore : Type
  | South : Shore
  | North : Shore

/-- Determines the shore where the ferry is docked after a given number of trips --/
def ferry_position (start : Shore) (trips : ℕ) : Shore :=
  if trips % 2 = 0 then start else
    match start with
    | Shore.South => Shore.North
    | Shore.North => Shore.South

/-- Theorem stating that after 202 trips starting from the south shore, the ferry ends up on the south shore --/
theorem ferry_position_after_202_trips :
  ferry_position Shore.South 202 = Shore.South := by
  sorry

end NUMINAMATH_CALUDE_ferry_position_after_202_trips_l248_24864


namespace NUMINAMATH_CALUDE_total_spending_theorem_l248_24826

/-- Represents the spending of Terry, Maria, and Raj over a week -/
structure WeeklySpending where
  terry : List Float
  maria : List Float
  raj : List Float

/-- Calculates the total spending of all three people over the week -/
def totalSpending (ws : WeeklySpending) : Float :=
  (ws.terry.sum + ws.maria.sum + ws.raj.sum)

/-- Theorem stating that the total spending equals $752.50 -/
theorem total_spending_theorem (ws : WeeklySpending) : 
  ws.terry = [6, 12, 36, 18, 14, 21, 33] ∧ 
  ws.maria = [3, 10, 72, 8, 14, 12, 33] ∧ 
  ws.raj = [7.5, 10, 216, 108, 14, 21, 84] → 
  totalSpending ws = 752.5 := by
  sorry

#eval totalSpending {
  terry := [6, 12, 36, 18, 14, 21, 33],
  maria := [3, 10, 72, 8, 14, 12, 33],
  raj := [7.5, 10, 216, 108, 14, 21, 84]
}

end NUMINAMATH_CALUDE_total_spending_theorem_l248_24826


namespace NUMINAMATH_CALUDE_one_friend_no_meat_l248_24802

/-- Represents the cookout scenario with given conditions -/
structure Cookout where
  total_friends : ℕ
  burgers_per_guest : ℕ
  buns_per_pack : ℕ
  packs_bought : ℕ
  friends_no_bread : ℕ

/-- Calculates the number of friends who don't eat meat -/
def friends_no_meat (c : Cookout) : ℕ :=
  c.total_friends - (c.packs_bought * c.buns_per_pack / c.burgers_per_guest + c.friends_no_bread)

/-- Theorem stating that exactly one friend doesn't eat meat -/
theorem one_friend_no_meat (c : Cookout) 
  (h1 : c.total_friends = 10)
  (h2 : c.burgers_per_guest = 3)
  (h3 : c.buns_per_pack = 8)
  (h4 : c.packs_bought = 3)
  (h5 : c.friends_no_bread = 1) :
  friends_no_meat c = 1 := by
  sorry

#eval friends_no_meat { 
  total_friends := 10, 
  burgers_per_guest := 3, 
  buns_per_pack := 8, 
  packs_bought := 3, 
  friends_no_bread := 1 
}

end NUMINAMATH_CALUDE_one_friend_no_meat_l248_24802


namespace NUMINAMATH_CALUDE_sum_product_equals_negative_one_l248_24813

theorem sum_product_equals_negative_one 
  (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (sum_squares_one : a^2 + b^2 + c^2 = 1) : 
  a*(b+c) + b*(a+c) + c*(a+b) = -1 := by
sorry

end NUMINAMATH_CALUDE_sum_product_equals_negative_one_l248_24813


namespace NUMINAMATH_CALUDE_square_root_theorem_l248_24888

theorem square_root_theorem (x : ℝ) :
  Real.sqrt (x + 3) = 3 → (x + 3)^2 = 81 := by
  sorry

end NUMINAMATH_CALUDE_square_root_theorem_l248_24888


namespace NUMINAMATH_CALUDE_carols_age_ratio_l248_24838

theorem carols_age_ratio (carol alice betty : ℕ) : 
  carol = 5 * alice →
  alice = carol - 12 →
  betty = 6 →
  (carol : ℚ) / betty = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_carols_age_ratio_l248_24838


namespace NUMINAMATH_CALUDE_largest_solution_of_equation_l248_24896

theorem largest_solution_of_equation :
  let f (x : ℚ) := 7 * (9 * x^2 + 11 * x + 12) - x * (9 * x - 46)
  ∃ (x : ℚ), f x = 0 ∧ (∀ (y : ℚ), f y = 0 → y ≤ x) ∧ x = -7/6 := by
  sorry

end NUMINAMATH_CALUDE_largest_solution_of_equation_l248_24896


namespace NUMINAMATH_CALUDE_power_of_power_l248_24850

theorem power_of_power (a : ℝ) : (a^4)^4 = a^16 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l248_24850


namespace NUMINAMATH_CALUDE_andys_hourly_wage_l248_24800

/-- Calculates Andy's hourly wage based on his shift earnings and activities. -/
theorem andys_hourly_wage (shift_hours : ℕ) (racquets_strung : ℕ) (grommets_changed : ℕ) (stencils_painted : ℕ)
  (restring_pay : ℕ) (grommet_pay : ℕ) (stencil_pay : ℕ) (total_earnings : ℕ) :
  shift_hours = 8 →
  racquets_strung = 7 →
  grommets_changed = 2 →
  stencils_painted = 5 →
  restring_pay = 15 →
  grommet_pay = 10 →
  stencil_pay = 1 →
  total_earnings = 202 →
  (total_earnings - (racquets_strung * restring_pay + grommets_changed * grommet_pay + stencils_painted * stencil_pay)) / shift_hours = 9 :=
by sorry

end NUMINAMATH_CALUDE_andys_hourly_wage_l248_24800


namespace NUMINAMATH_CALUDE_history_class_grade_distribution_l248_24863

theorem history_class_grade_distribution (total_students : ℕ) 
  (prob_A prob_B prob_C prob_D : ℝ) (B_count : ℕ) : 
  total_students = 52 →
  prob_A = 0.5 * prob_B →
  prob_C = 2 * prob_B →
  prob_D = 0.5 * prob_B →
  prob_A + prob_B + prob_C + prob_D = 1 →
  B_count = 13 →
  (0.5 * B_count : ℝ) + B_count + (2 * B_count) + (0.5 * B_count) = total_students := by
  sorry

end NUMINAMATH_CALUDE_history_class_grade_distribution_l248_24863


namespace NUMINAMATH_CALUDE_intersection_points_l248_24856

/-- The set of possible values for k such that the graph of |z - 3| = 3|z + 3| 
    intersects the graph of |z| = k in exactly one point -/
def possible_k_values : Set ℝ :=
  {k : ℝ | k = 1.5 ∨ k = 6}

/-- The condition that |z - 3| = 3|z + 3| -/
def condition (z : ℂ) : Prop :=
  Complex.abs (z - 3) = 3 * Complex.abs (z + 3)

/-- The theorem stating that the only values of k for which the graph of |z - 3| = 3|z + 3| 
    intersects the graph of |z| = k in exactly one point are 1.5 and 6 -/
theorem intersection_points (k : ℝ) :
  (∃! z : ℂ, condition z ∧ Complex.abs z = k) ↔ k ∈ possible_k_values := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_l248_24856


namespace NUMINAMATH_CALUDE_tom_initial_balloons_l248_24855

/-- The number of balloons Tom gave to Fred -/
def balloons_given : ℕ := 16

/-- The number of balloons Tom has left -/
def balloons_left : ℕ := 14

/-- The initial number of balloons Tom had -/
def initial_balloons : ℕ := balloons_given + balloons_left

theorem tom_initial_balloons : initial_balloons = 30 := by
  sorry

end NUMINAMATH_CALUDE_tom_initial_balloons_l248_24855


namespace NUMINAMATH_CALUDE_base6_subtraction_l248_24851

-- Define a function to convert a list of digits in base 6 to a natural number
def base6ToNat (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 6 * acc + d) 0

-- Define a function to convert a natural number to a list of digits in base 6
def natToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
    aux n []

-- State the theorem
theorem base6_subtraction :
  let a := base6ToNat [5, 5, 5]
  let b := base6ToNat [5, 5]
  let c := base6ToNat [2, 0, 2]
  let result := base6ToNat [6, 1, 4]
  (a + b) - c = result := by sorry

end NUMINAMATH_CALUDE_base6_subtraction_l248_24851


namespace NUMINAMATH_CALUDE_cost_price_of_article_l248_24882

/-- 
Given an article where the profit when selling it for Rs. 57 is equal to the loss 
when selling it for Rs. 43, prove that the cost price of the article is Rs. 50.
-/
theorem cost_price_of_article (cost_price : ℕ) : cost_price = 50 := by
  sorry

/--
Helper function to calculate profit
-/
def profit (selling_price cost_price : ℕ) : ℤ :=
  (selling_price : ℤ) - (cost_price : ℤ)

/--
Helper function to calculate loss
-/
def loss (cost_price selling_price : ℕ) : ℤ :=
  (cost_price : ℤ) - (selling_price : ℤ)

/--
Assumption that profit when selling at Rs. 57 equals loss when selling at Rs. 43
-/
axiom profit_loss_equality (cost_price : ℕ) :
  profit 57 cost_price = loss cost_price 43

end NUMINAMATH_CALUDE_cost_price_of_article_l248_24882


namespace NUMINAMATH_CALUDE_license_plate_theorem_l248_24884

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of letter positions on the license plate -/
def letter_positions : ℕ := 6

/-- The number of digit positions on the license plate -/
def digit_positions : ℕ := 3

/-- The number of possible license plate combinations with 6 letters 
(where exactly two different letters are repeated once) followed by 3 non-repeating digits -/
def license_plate_combinations : ℕ :=
  (Nat.choose alphabet_size 2) *
  (Nat.choose letter_positions 2) *
  (Nat.choose (letter_positions - 2) 2) *
  (Nat.choose (alphabet_size - 2) 2) *
  (10 * 9 * 8)

theorem license_plate_theorem : license_plate_combinations = 84563400000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_theorem_l248_24884


namespace NUMINAMATH_CALUDE_multiplicative_inverse_207_mod_397_l248_24883

theorem multiplicative_inverse_207_mod_397 :
  ∃ a : ℕ, a < 397 ∧ (207 * a) % 397 = 1 :=
by
  use 66
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_207_mod_397_l248_24883


namespace NUMINAMATH_CALUDE_prime_sum_1998_l248_24817

theorem prime_sum_1998 (p q r : ℕ) (s t u : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) 
  (h_eq : 1998 = p^s * q^t * r^u) : p + q + r = 42 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_1998_l248_24817


namespace NUMINAMATH_CALUDE_circle_symmetry_l248_24811

-- Define the given circle
def given_circle (x y : ℝ) : Prop :=
  (x + 2)^2 + y^2 = 5

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop :=
  x - y + 1 = 0

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop :=
  (x + 1)^2 + (y + 1)^2 = 5

-- Theorem statement
theorem circle_symmetry :
  ∀ (x y x' y' : ℝ),
    given_circle x y →
    symmetry_line ((x + x') / 2) ((y + y') / 2) →
    symmetric_circle x' y' :=
by sorry

end NUMINAMATH_CALUDE_circle_symmetry_l248_24811


namespace NUMINAMATH_CALUDE_triangle_lines_l248_24828

-- Define the triangle ABC
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (-1, 3)
def C : ℝ × ℝ := (3, 4)

-- Define the altitude line l₁
def l₁ (x y : ℝ) : Prop := 4 * x + y - 5 = 0

-- Define the two possible equations for line l₂
def l₂_1 (x y : ℝ) : Prop := x + y - 7 = 0
def l₂_2 (x y : ℝ) : Prop := 2 * x - 3 * y + 6 = 0

-- Theorem statement
theorem triangle_lines :
  -- l₁ is the altitude from A to BC
  (∀ x y : ℝ, l₁ x y ↔ (y - A.2 = -(B.2 - C.2)/(B.1 - C.1) * (x - A.1))) ∧
  -- l₂ passes through C
  ((∀ x y : ℝ, l₂_1 x y → x = C.1 ∧ y = C.2) ∨
   (∀ x y : ℝ, l₂_2 x y → x = C.1 ∧ y = C.2)) ∧
  -- Distances from A and B to l₂ are equal
  ((∀ x y : ℝ, l₂_1 x y → 
    (|x + y - (A.1 + A.2)|/Real.sqrt 2 = |x + y - (B.1 + B.2)|/Real.sqrt 2)) ∨
   (∀ x y : ℝ, l₂_2 x y → 
    (|2*x - 3*y + 6 - (2*A.1 - 3*A.2 + 6)|/Real.sqrt 13 = 
     |2*x - 3*y + 6 - (2*B.1 - 3*B.2 + 6)|/Real.sqrt 13))) :=
by sorry


end NUMINAMATH_CALUDE_triangle_lines_l248_24828


namespace NUMINAMATH_CALUDE_distance_y_to_earth_l248_24809

-- Define the distances
def distance_earth_to_x : ℝ := 0.5
def distance_x_to_y : ℝ := 0.1
def total_distance : ℝ := 0.7

-- Theorem to prove
theorem distance_y_to_earth : 
  total_distance - (distance_earth_to_x + distance_x_to_y) = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_distance_y_to_earth_l248_24809


namespace NUMINAMATH_CALUDE_angle_BAD_measure_l248_24895

-- Define the geometric configuration
structure GeometricConfiguration where
  -- We don't need to explicitly define points, just the angles
  angleABC : ℝ
  angleBDE : ℝ
  angleDBE : ℝ
  -- We'll define angleABD in terms of angleABC

-- Define the theorem
theorem angle_BAD_measure (config : GeometricConfiguration) 
  (h1 : config.angleABC = 132)
  (h2 : config.angleBDE = 31)
  (h3 : config.angleDBE = 30)
  : 180 - (180 - config.angleABC) - config.angleBDE - config.angleDBE = 119 := by
  sorry


end NUMINAMATH_CALUDE_angle_BAD_measure_l248_24895


namespace NUMINAMATH_CALUDE_defective_product_selection_l248_24874

theorem defective_product_selection (n m k : ℕ) (hn : n = 100) (hm : m = 98) (hk : k = 3) :
  let total := n
  let qualified := m
  let defective := n - m
  let select := k
  Nat.choose n k - Nat.choose m k = 
    Nat.choose defective 1 * Nat.choose qualified 2 + 
    Nat.choose defective 2 * Nat.choose qualified 1 :=
by sorry

end NUMINAMATH_CALUDE_defective_product_selection_l248_24874


namespace NUMINAMATH_CALUDE_pyramid_volume_l248_24860

theorem pyramid_volume (base_length : ℝ) (base_width : ℝ) (height : ℝ) :
  base_length = 1/2 →
  base_width = 1/4 →
  height = 1 →
  (1/3) * base_length * base_width * height = 1/24 := by
sorry

end NUMINAMATH_CALUDE_pyramid_volume_l248_24860


namespace NUMINAMATH_CALUDE_expression_evaluation_l248_24849

theorem expression_evaluation (a b : ℤ) (ha : a = -1) (hb : b = 1) :
  (a - 2*b) * (a^2 + 2*a*b + 4*b^2) - a * (a - 5*b) * (a + 3*b) = -21 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l248_24849


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l248_24820

theorem triangle_angle_measure (a b : ℝ) (A B : Real) :
  0 < a ∧ 0 < b ∧ 0 < A ∧ A < π ∧ 0 < B ∧ B < π →
  b = 2 * a * Real.sin B →
  Real.sin A = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l248_24820


namespace NUMINAMATH_CALUDE_simple_interest_rate_calculation_l248_24885

/-- Given an initial sum of money that amounts to 9800 after 5 years
    and 12005 after 8 years at the same rate of simple interest,
    prove that the rate of interest per annum is 7.5% -/
theorem simple_interest_rate_calculation (P : ℝ) (R : ℝ) :
  P * (1 + 5 * R / 100) = 9800 →
  P * (1 + 8 * R / 100) = 12005 →
  R = 7.5 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_calculation_l248_24885


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l248_24819

def U : Set Int := {-2, -1, 0, 1, 2}

def M : Set Int := {y | ∃ x, y = 2^x}

def N : Set Int := {x | x^2 - x - 2 = 0}

theorem complement_M_intersect_N :
  (U \ M) ∩ N = {-1} := by sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l248_24819


namespace NUMINAMATH_CALUDE_tony_pills_l248_24894

/-- The number of pills left in Tony's bottle after his treatment --/
def pills_left : ℕ :=
  let initial_pills : ℕ := 50
  let first_two_days : ℕ := 2 * 3 * 2
  let next_three_days : ℕ := 1 * 3 * 3
  let last_day : ℕ := 2
  initial_pills - (first_two_days + next_three_days + last_day)

theorem tony_pills : pills_left = 27 := by
  sorry

end NUMINAMATH_CALUDE_tony_pills_l248_24894


namespace NUMINAMATH_CALUDE_value_of_a_l248_24857

theorem value_of_a (a b c d : ℤ) 
  (eq1 : 2 * a + 2 = b)
  (eq2 : 2 * b + 2 = c)
  (eq3 : 2 * c + 2 = d)
  (eq4 : 2 * d + 2 = 62) : 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l248_24857


namespace NUMINAMATH_CALUDE_angle_measure_theorem_l248_24889

theorem angle_measure_theorem (x : ℝ) : x = 2 * (90 - x) - 60 ↔ x = 40 := by sorry

end NUMINAMATH_CALUDE_angle_measure_theorem_l248_24889


namespace NUMINAMATH_CALUDE_bus_children_count_l248_24823

/-- The total number of children on a bus after more children got on is equal to the sum of the initial number of children and the additional children who got on. -/
theorem bus_children_count (initial_children additional_children : ℕ) :
  initial_children + additional_children = initial_children + additional_children :=
by sorry

end NUMINAMATH_CALUDE_bus_children_count_l248_24823


namespace NUMINAMATH_CALUDE_seats_per_row_l248_24812

/-- Proves that given the specified conditions, the number of seats in each row is 8 -/
theorem seats_per_row (rows : ℕ) (base_cost : ℚ) (discount_rate : ℚ) (discount_group : ℕ) (total_cost : ℚ) :
  rows = 5 →
  base_cost = 30 →
  discount_rate = 1/10 →
  discount_group = 10 →
  total_cost = 1080 →
  ∃ (seats_per_row : ℕ),
    seats_per_row = 8 ∧
    total_cost = rows * (seats_per_row * base_cost - (seats_per_row / discount_group) * (discount_rate * base_cost * discount_group)) :=
by sorry

end NUMINAMATH_CALUDE_seats_per_row_l248_24812


namespace NUMINAMATH_CALUDE_polynomial_expansion_l248_24844

theorem polynomial_expansion :
  ∀ x : ℝ, (2 * x^2 - 3 * x + 1) * (x^2 + x + 3) = 2 * x^4 - x^3 + 4 * x^2 - 8 * x + 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l248_24844


namespace NUMINAMATH_CALUDE_num_common_tangents_for_given_circles_l248_24877

/-- Circle represented by its equation in the form (x - h)² + (y - k)² = r² --/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- Number of common tangents between two circles --/
def num_common_tangents (c1 c2 : Circle) : ℕ :=
  sorry

/-- Convert from x² + y² - 2ax = 0 form to Circle structure --/
def circle_from_equation (a : ℝ) : Circle :=
  { h := a, k := 0, r := a }

theorem num_common_tangents_for_given_circles :
  let c1 := circle_from_equation 1
  let c2 := circle_from_equation 2
  num_common_tangents c1 c2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_num_common_tangents_for_given_circles_l248_24877


namespace NUMINAMATH_CALUDE_survey_result_l248_24873

theorem survey_result (total_surveyed : ℕ) 
  (believed_spread_diseases : ℕ) 
  (believed_flu : ℕ) : 
  (believed_spread_diseases : ℝ) / total_surveyed = 0.905 →
  (believed_flu : ℝ) / believed_spread_diseases = 0.503 →
  believed_flu = 26 →
  total_surveyed = 57 := by
sorry

end NUMINAMATH_CALUDE_survey_result_l248_24873


namespace NUMINAMATH_CALUDE_water_tank_emptying_time_water_tank_empties_in_12_minutes_l248_24846

/-- Represents the time it takes to empty a water tank given specific conditions. -/
theorem water_tank_emptying_time 
  (initial_fill : ℚ) 
  (pipe_a_fill_rate : ℚ) 
  (pipe_b_empty_rate : ℚ) : ℚ :=
  let net_rate := pipe_a_fill_rate - pipe_b_empty_rate
  let time_to_empty := initial_fill / (-net_rate)
  by
    -- Assuming initial_fill = 4/5
    -- pipe_a_fill_rate = 1/10
    -- pipe_b_empty_rate = 1/6
    sorry

/-- The main theorem stating it takes 12 minutes to empty the tank. -/
theorem water_tank_empties_in_12_minutes : 
  water_tank_emptying_time (4/5) (1/10) (1/6) = 12 :=
by sorry

end NUMINAMATH_CALUDE_water_tank_emptying_time_water_tank_empties_in_12_minutes_l248_24846


namespace NUMINAMATH_CALUDE_solve_a_and_m_solve_inequality_l248_24879

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Theorem 1
theorem solve_a_and_m (a m : ℝ) : 
  (∀ x, f a x ≤ m ↔ -1 ≤ x ∧ x ≤ 5) → a = 2 ∧ m = 3 :=
sorry

-- Theorem 2
theorem solve_inequality (t : ℝ) (h : 0 ≤ t ∧ t < 2) :
  {x : ℝ | f 2 x + t ≥ f 2 (x + 2)} = Set.Iic ((t + 2) / 2) :=
sorry

end NUMINAMATH_CALUDE_solve_a_and_m_solve_inequality_l248_24879


namespace NUMINAMATH_CALUDE_tuesday_to_monday_ratio_l248_24808

/-- Proves the ratio of square feet painted on Tuesday to Monday is 2:1 -/
theorem tuesday_to_monday_ratio (monday : ℝ) (wednesday : ℝ) (total : ℝ) : 
  monday = 30 →
  wednesday = monday / 2 →
  total = monday + wednesday + (total - monday - wednesday) →
  (total - monday - wednesday) / monday = 2 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_to_monday_ratio_l248_24808


namespace NUMINAMATH_CALUDE_val_money_value_l248_24816

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The number of nickels Val initially has -/
def initial_nickels : ℕ := 20

/-- The number of dimes Val has -/
def dimes : ℕ := 3 * initial_nickels

/-- The number of additional nickels Val finds -/
def additional_nickels : ℕ := 2 * initial_nickels

/-- The total value of money Val has after taking the additional nickels -/
def total_value : ℚ := 
  (initial_nickels : ℚ) * nickel_value + 
  (dimes : ℚ) * dime_value + 
  (additional_nickels : ℚ) * nickel_value

theorem val_money_value : total_value = 9 := by
  sorry

end NUMINAMATH_CALUDE_val_money_value_l248_24816


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_1419_l248_24859

def product_of_evens (n : Nat) : Nat :=
  (List.range ((n / 2) - 1)).foldl (fun acc i => acc * (2 * (i + 2))) 2

theorem smallest_n_divisible_by_1419 :
  (∀ m : Nat, m < 106 → m % 2 = 0 → ¬(product_of_evens m % 1419 = 0)) ∧
  (106 % 2 = 0 ∧ product_of_evens 106 % 1419 = 0) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_1419_l248_24859


namespace NUMINAMATH_CALUDE_divisible_by_nine_l248_24845

theorem divisible_by_nine (h : ℕ) (h_single_digit : h < 10) :
  (7600 + 100 * h + 4) % 9 = 0 ↔ h = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_nine_l248_24845


namespace NUMINAMATH_CALUDE_michael_twice_jacob_age_l248_24876

/-- 
Given that Jacob is 13 years younger than Michael and Jacob will be 8 years old in 4 years,
this theorem proves that Michael will be twice as old as Jacob in 9 years.
-/
theorem michael_twice_jacob_age (jacob_current_age : ℕ) (michael_current_age : ℕ) :
  jacob_current_age + 4 = 8 →
  michael_current_age = jacob_current_age + 13 →
  ∃ (years : ℕ), years = 9 ∧ michael_current_age + years = 2 * (jacob_current_age + years) :=
by sorry

end NUMINAMATH_CALUDE_michael_twice_jacob_age_l248_24876


namespace NUMINAMATH_CALUDE_temperature_conversion_l248_24842

theorem temperature_conversion (t k : ℝ) : 
  t = 5 / 9 * (k - 32) → t = 105 → k = 221 := by
sorry

end NUMINAMATH_CALUDE_temperature_conversion_l248_24842


namespace NUMINAMATH_CALUDE_local_election_vote_count_l248_24835

theorem local_election_vote_count (candidate1_percent : ℝ) (candidate2_percent : ℝ) 
  (candidate3_percent : ℝ) (candidate4_percent : ℝ) (candidate2_votes : ℕ) :
  candidate1_percent = 0.45 →
  candidate2_percent = 0.25 →
  candidate3_percent = 0.20 →
  candidate4_percent = 0.10 →
  candidate2_votes = 600 →
  candidate1_percent + candidate2_percent + candidate3_percent + candidate4_percent = 1 →
  ∃ (total_votes : ℕ), total_votes = 2400 ∧ 
    (candidate2_percent : ℝ) * total_votes = candidate2_votes := by
  sorry

end NUMINAMATH_CALUDE_local_election_vote_count_l248_24835


namespace NUMINAMATH_CALUDE_ratio_evaluation_and_closest_integer_l248_24854

theorem ratio_evaluation_and_closest_integer : 
  let r := (2^3000 + 2^3003) / (2^3001 + 2^3002)
  r = 3/2 ∧ ∀ n : ℤ, |r - 2| ≤ |r - n| :=
by
  sorry

end NUMINAMATH_CALUDE_ratio_evaluation_and_closest_integer_l248_24854


namespace NUMINAMATH_CALUDE_characterize_inequality_l248_24892

theorem characterize_inequality (x y : ℝ) :
  x^2 * y - y ≥ 0 ↔ (y ≥ 0 ∧ abs x ≥ 1) ∨ (y ≤ 0 ∧ abs x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_characterize_inequality_l248_24892


namespace NUMINAMATH_CALUDE_jessica_purchase_cost_l248_24886

def chocolate_bars : ℕ := 10
def gummy_bears : ℕ := 10
def chocolate_chips : ℕ := 20

def price_chocolate_bar : ℕ := 3
def price_gummy_bears : ℕ := 2
def price_chocolate_chips : ℕ := 5

def total_cost : ℕ := chocolate_bars * price_chocolate_bar +
                      gummy_bears * price_gummy_bears +
                      chocolate_chips * price_chocolate_chips

theorem jessica_purchase_cost : total_cost = 150 := by
  sorry

end NUMINAMATH_CALUDE_jessica_purchase_cost_l248_24886


namespace NUMINAMATH_CALUDE_area_of_triangle_MOI_l248_24840

/-- Triangle ABC with given side lengths -/
structure Triangle where
  AB : ℝ
  AC : ℝ
  BC : ℝ

/-- Circumcenter of a triangle -/
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

/-- Incenter of a triangle -/
def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- Point M such that a circle centered at M is tangent to AC, BC, and the circumcircle -/
def tangentPoint (t : Triangle) : ℝ × ℝ := sorry

/-- Area of a triangle given three points -/
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Main theorem -/
theorem area_of_triangle_MOI (t : Triangle) 
  (h1 : t.AB = 15) (h2 : t.AC = 8) (h3 : t.BC = 7) : 
  triangleArea (tangentPoint t) (circumcenter t) (incenter t) = 1.765 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_MOI_l248_24840


namespace NUMINAMATH_CALUDE_red_balls_count_l248_24815

/-- Given a bag of 16 balls with red and blue balls, if the probability of drawing
    exactly 2 red balls when 3 are drawn at random is 1/10, then there are 7 red balls. -/
theorem red_balls_count (total : ℕ) (red : ℕ) (blue : ℕ) :
  total = 16 ∧
  total = red + blue ∧
  (Nat.choose red 2 * blue : ℚ) / Nat.choose total 3 = 1 / 10 →
  red = 7 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l248_24815


namespace NUMINAMATH_CALUDE_symmetry_implies_periodicity_l248_24861

/-- A function f: ℝ → ℝ is symmetric with respect to the point (a, y₀) -/
def SymmetricPoint (f : ℝ → ℝ) (a y₀ : ℝ) : Prop :=
  ∀ x, f (a + x) - y₀ = y₀ - f (a - x)

/-- A function f: ℝ → ℝ is symmetric with respect to the line x = b -/
def SymmetricLine (f : ℝ → ℝ) (b : ℝ) : Prop :=
  ∀ x, f (b + x) = f (b - x)

/-- A function f: ℝ → ℝ is periodic with period p -/
def Periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem symmetry_implies_periodicity (f : ℝ → ℝ) (a b y₀ : ℝ) (hb : b > a) 
    (h1 : SymmetricPoint f a y₀) (h2 : SymmetricLine f b) :
    Periodic f (4 * (b - a)) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_periodicity_l248_24861


namespace NUMINAMATH_CALUDE_inequality_system_solution_l248_24847

theorem inequality_system_solution (x : ℝ) :
  3 * x > x - 4 ∧ (4 + x) / 3 > x + 2 → -2 < x ∧ x < -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l248_24847


namespace NUMINAMATH_CALUDE_largest_solution_quadratic_l248_24829

theorem largest_solution_quadratic : 
  let f : ℝ → ℝ := fun y ↦ 6 * y^2 - 31 * y + 35
  ∃ y : ℝ, f y = 0 ∧ ∀ z : ℝ, f z = 0 → z ≤ y ∧ y = (5 : ℝ) / 2 := by sorry

end NUMINAMATH_CALUDE_largest_solution_quadratic_l248_24829


namespace NUMINAMATH_CALUDE_plate_arrangement_theorem_l248_24848

/-- The number of ways to arrange plates around a circular table. -/
def circularArrangements (total : ℕ) (blue red green orange : ℕ) : ℕ :=
  (Nat.factorial total) / (Nat.factorial blue * Nat.factorial red * Nat.factorial green * Nat.factorial orange) / total

/-- The number of ways to arrange plates around a circular table with one pair adjacent. -/
def circularArrangementsWithPairAdjacent (total : ℕ) (blue red green orange : ℕ) : ℕ :=
  (Nat.factorial (total - 1)) / (Nat.factorial blue * Nat.factorial red * Nat.factorial green * Nat.factorial orange) / (total - 1)

/-- The number of ways to arrange plates around a circular table with both pairs adjacent. -/
def circularArrangementsWithBothPairsAdjacent (total : ℕ) (blue red green orange : ℕ) : ℕ :=
  (Nat.factorial (total - 2)) / (Nat.factorial blue * Nat.factorial red * Nat.factorial green * Nat.factorial orange) / (total - 2)

theorem plate_arrangement_theorem :
  let total := 11
  let blue := 6
  let red := 2
  let green := 2
  let orange := 1
  circularArrangements total blue red green orange -
  circularArrangementsWithPairAdjacent total blue red green orange -
  circularArrangementsWithPairAdjacent total blue red green orange +
  circularArrangementsWithBothPairsAdjacent total blue red green orange = 1568 := by
  sorry

end NUMINAMATH_CALUDE_plate_arrangement_theorem_l248_24848


namespace NUMINAMATH_CALUDE_work_completion_time_l248_24868

/-- Given workers A, B, and C with their individual work rates, 
    prove that B and C together can complete the work in 3 hours. -/
theorem work_completion_time 
  (rate_A : ℝ) 
  (rate_B : ℝ) 
  (rate_C : ℝ) 
  (h1 : rate_A = 1 / 4) 
  (h2 : rate_A + rate_C = 1 / 2) 
  (h3 : rate_B = 1 / 12) : 
  1 / (rate_B + rate_C) = 3 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l248_24868


namespace NUMINAMATH_CALUDE_thousands_digit_of_common_remainder_l248_24810

theorem thousands_digit_of_common_remainder (n : ℕ) 
  (h1 : n > 1000000)
  (h2 : n % 40 = n % 625) : 
  (n / 1000) % 10 = 0 ∨ (n / 1000) % 10 = 5 := by
sorry

end NUMINAMATH_CALUDE_thousands_digit_of_common_remainder_l248_24810


namespace NUMINAMATH_CALUDE_angle_sum_pi_half_l248_24841

theorem angle_sum_pi_half (α β : Real) (h1 : 0 < α) (h2 : α < π/2) (h3 : 0 < β) (h4 : β < π/2)
  (h5 : 3 * Real.sin α ^ 2 + 2 * Real.sin β ^ 2 = 1)
  (h6 : 3 * Real.sin (2 * α) - 2 * Real.sin (2 * β) = 0) :
  α + 2 * β = π/2 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_pi_half_l248_24841


namespace NUMINAMATH_CALUDE_johns_paintball_expense_l248_24834

/-- The amount John spends on paintballs per month -/
def monthly_paintball_expense (plays_per_month : ℕ) (boxes_per_play : ℕ) (cost_per_box : ℕ) : ℕ :=
  plays_per_month * boxes_per_play * cost_per_box

/-- Theorem stating John's monthly paintball expense -/
theorem johns_paintball_expense :
  monthly_paintball_expense 3 3 25 = 225 := by
  sorry

end NUMINAMATH_CALUDE_johns_paintball_expense_l248_24834


namespace NUMINAMATH_CALUDE_min_value_of_x_l248_24821

theorem min_value_of_x (x : ℝ) (h1 : x > 0) 
  (h2 : Real.log x / Real.log 3 ≥ Real.log 9 / Real.log 3 - (1/3) * (Real.log x / Real.log 3)) :
  x ≥ Real.sqrt 27 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_x_l248_24821


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_given_condition_l248_24890

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |2*x - 1|

-- Theorem 1
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≥ 2} = {x : ℝ | x ≤ 0 ∨ x ≥ 2/3} := by sorry

-- Theorem 2
theorem range_of_a_given_condition (a : ℝ) :
  (∀ x ∈ Set.Icc (1/2 : ℝ) 1, f a x ≤ 2*x) →
  a ∈ Set.Icc (-3/2 : ℝ) 0 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_given_condition_l248_24890


namespace NUMINAMATH_CALUDE_sum_of_powers_of_five_squares_l248_24832

theorem sum_of_powers_of_five_squares (m n : ℕ+) :
  (∃ a b : ℤ, (5 : ℤ)^(n : ℕ) + (5 : ℤ)^(m : ℕ) = a^2 + b^2) ↔ Even (n - m) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_five_squares_l248_24832


namespace NUMINAMATH_CALUDE_grain_output_scientific_notation_l248_24830

/-- Represents the total grain output of China in 2021 in tons -/
def china_grain_output : ℝ := 682.85e6

/-- The scientific notation representation of China's grain output -/
def scientific_notation : ℝ := 6.8285e8

/-- Theorem stating that the grain output is equal to its scientific notation representation -/
theorem grain_output_scientific_notation : china_grain_output = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_grain_output_scientific_notation_l248_24830


namespace NUMINAMATH_CALUDE_least_number_for_divisibility_l248_24806

theorem least_number_for_divisibility : ∃ (n : ℕ), n = 11 ∧
  (∀ (m : ℕ), m < n → ¬((1789 + m) % 5 = 0 ∧ (1789 + m) % 6 = 0 ∧ (1789 + m) % 4 = 0 ∧ (1789 + m) % 3 = 0)) ∧
  ((1789 + n) % 5 = 0 ∧ (1789 + n) % 6 = 0 ∧ (1789 + n) % 4 = 0 ∧ (1789 + n) % 3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_least_number_for_divisibility_l248_24806


namespace NUMINAMATH_CALUDE_ellipse_ratio_l248_24837

/-- Given an ellipse with semi-major axis a, semi-minor axis b, and semi-focal length c,
    if a² + b² - 3c² = 0, then (a+c)/(a-c) = 3 + 2√2 -/
theorem ellipse_ratio (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
    (h4 : a^2 + b^2 - 3*c^2 = 0) : (a + c) / (a - c) = 3 + 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ellipse_ratio_l248_24837


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l248_24862

def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}
def N : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}

theorem intersection_complement_theorem :
  N ∩ Mᶜ = {x : ℝ | 3 < x ∧ x ≤ 4} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l248_24862


namespace NUMINAMATH_CALUDE_log_equation_solution_l248_24869

theorem log_equation_solution (x : ℝ) :
  x > 0 → ((Real.log x / Real.log 2) * (Real.log x / Real.log 5) = Real.log 10 / Real.log 2) ↔
  x = 2 ^ Real.sqrt (6 * Real.log 2) :=
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l248_24869


namespace NUMINAMATH_CALUDE_tom_read_70_books_l248_24871

/-- The number of books Tom read each month -/
def books_per_month : List Nat := [2, 6, 12, 20, 30]

/-- The total number of books Tom read over five months -/
def total_books : Nat := books_per_month.sum

theorem tom_read_70_books : total_books = 70 := by
  sorry

end NUMINAMATH_CALUDE_tom_read_70_books_l248_24871


namespace NUMINAMATH_CALUDE_cube_volume_surface_area_l248_24858

theorem cube_volume_surface_area (x : ℝ) : 
  (∃ (s : ℝ), s > 0 ∧ s^3 = 8*x ∧ 6*s^2 = 2*x) → x = 1728 :=
by sorry

end NUMINAMATH_CALUDE_cube_volume_surface_area_l248_24858


namespace NUMINAMATH_CALUDE_lauras_blocks_l248_24843

/-- Calculates the total number of blocks given the number of friends and blocks per friend -/
def total_blocks (num_friends : ℕ) (blocks_per_friend : ℕ) : ℕ :=
  num_friends * blocks_per_friend

/-- Proves that given 4 friends and 7 blocks per friend, the total number of blocks is 28 -/
theorem lauras_blocks : total_blocks 4 7 = 28 := by
  sorry

end NUMINAMATH_CALUDE_lauras_blocks_l248_24843


namespace NUMINAMATH_CALUDE_orthocenter_locus_l248_24814

-- Define the circle
def Circle (O : ℝ × ℝ) (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - O.1)^2 + (p.2 - O.2)^2 = r^2}

-- Define the secant line
def SecantLine (K : ℝ × ℝ) (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = m * (p.1 - K.1)}

-- Define the orthocenter of a triangle
def Orthocenter (A P Q : ℝ × ℝ) : ℝ × ℝ :=
  sorry

-- State the theorem
theorem orthocenter_locus
  (O : ℝ × ℝ)
  (r k : ℝ)
  (h_k : k > r) :
  ∀ (A : ℝ × ℝ) (m : ℝ),
  A ∈ Circle O r →
  ∃ (P Q : ℝ × ℝ),
  P ∈ Circle O r ∩ SecantLine (k, 0) m ∧
  Q ∈ Circle O r ∩ SecantLine (k, 0) m ∧
  P ≠ Q ∧
  let H := Orthocenter A P Q
  (H.1 - 2*k*m^2/(m^2 + 1))^2 + (H.2 + 2*k*m/(m^2 + 1))^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_orthocenter_locus_l248_24814


namespace NUMINAMATH_CALUDE_ratio_equality_l248_24897

theorem ratio_equality (a b c x y z : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_abc : a^2 + b^2 + c^2 = 49)
  (sum_xyz : x^2 + y^2 + z^2 = 16)
  (dot_product : a*x + b*y + c*z = 28) :
  (a + b + c) / (x + y + z) = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l248_24897


namespace NUMINAMATH_CALUDE_matthew_rebecca_age_difference_l248_24872

/-- Represents the ages of three children and their properties --/
structure ChildrenAges where
  freddy : ℕ
  matthew : ℕ
  rebecca : ℕ
  total_age : freddy + matthew + rebecca = 35
  freddy_age : freddy = 15
  matthew_younger : matthew = freddy - 4
  matthew_older : matthew > rebecca

/-- Theorem stating that Matthew is 2 years older than Rebecca --/
theorem matthew_rebecca_age_difference (ages : ChildrenAges) : ages.matthew = ages.rebecca + 2 := by
  sorry

end NUMINAMATH_CALUDE_matthew_rebecca_age_difference_l248_24872


namespace NUMINAMATH_CALUDE_vector_calculation_l248_24853

def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![1, -1]

theorem vector_calculation :
  (1/3 : ℝ) • a - (4/3 : ℝ) • b = ![(-1 : ℝ), 2] := by sorry

end NUMINAMATH_CALUDE_vector_calculation_l248_24853


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_5_and_6_l248_24818

theorem smallest_perfect_square_divisible_by_5_and_6 :
  ∀ n : ℕ, n > 0 → n * n < 900 → ¬(5 ∣ (n * n) ∧ 6 ∣ (n * n)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_5_and_6_l248_24818


namespace NUMINAMATH_CALUDE_square_root_positive_l248_24839

theorem square_root_positive (x : ℝ) (h : x > 0) : Real.sqrt x > 0 := by
  sorry

end NUMINAMATH_CALUDE_square_root_positive_l248_24839


namespace NUMINAMATH_CALUDE_inequality_proof_l248_24875

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 ∧
  ((x + y) / z + (y + z) / x + (z + x) / y = 7 ↔ x = 2*y ∧ y = z) :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l248_24875


namespace NUMINAMATH_CALUDE_stream_speed_l248_24833

/-- Given a canoe's upstream and downstream speeds, calculate the stream speed -/
theorem stream_speed (upstream_speed downstream_speed : ℝ) 
  (h_upstream : upstream_speed = 3)
  (h_downstream : downstream_speed = 12) :
  ∃ (stream_speed : ℝ), stream_speed = 4.5 ∧
    upstream_speed = (downstream_speed - upstream_speed) / 2 - stream_speed ∧
    downstream_speed = (downstream_speed - upstream_speed) / 2 + stream_speed :=
by sorry

end NUMINAMATH_CALUDE_stream_speed_l248_24833


namespace NUMINAMATH_CALUDE_distance_to_focus_is_two_l248_24880

/-- Parabola E defined by y² = 4x -/
def parabola_E (x y : ℝ) : Prop := y^2 = 4*x

/-- Point P with coordinates (x₀, 2) -/
structure Point_P where
  x₀ : ℝ

/-- P lies on parabola E -/
def P_on_E (P : Point_P) : Prop := parabola_E P.x₀ 2

/-- Distance from a point to the focus of parabola E -/
def distance_to_focus (P : Point_P) : ℝ := sorry

theorem distance_to_focus_is_two (P : Point_P) (h : P_on_E P) : 
  distance_to_focus P = 2 := by sorry

end NUMINAMATH_CALUDE_distance_to_focus_is_two_l248_24880


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l248_24827

/-- Given a = 15, b = 19, c = 25, and S = a + b + c = 59, prove that the expression
    (a² * (1/b - 1/c) + b² * (1/c - 1/a) + c² * (1/a - 1/b) + 37) /
    (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b) + 2)
    equals 77.5 -/
theorem complex_fraction_evaluation (a b c S : ℚ) 
    (ha : a = 15) (hb : b = 19) (hc : c = 25) (hS : S = a + b + c) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b) + 37) /
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b) + 2) = 77.5 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l248_24827


namespace NUMINAMATH_CALUDE_k_increasing_on_neg_reals_l248_24807

/-- The function k(x) = 3 - x is increasing on the interval (-∞, 0). -/
theorem k_increasing_on_neg_reals :
  StrictMonoOn (fun x : ℝ => 3 - x) (Set.Iio 0) := by
  sorry

end NUMINAMATH_CALUDE_k_increasing_on_neg_reals_l248_24807


namespace NUMINAMATH_CALUDE_largest_convex_polygon_on_grid_l248_24898

/-- Represents a point on the grid -/
structure GridPoint where
  x : ℕ
  y : ℕ
  hx : x < 2004
  hy : y < 2004

/-- Represents a convex polygon on the grid -/
structure ConvexPolygon where
  vertices : List GridPoint
  is_convex : Bool  -- We assume there's a function to check convexity

/-- The main theorem stating the largest possible n-gon on the grid -/
theorem largest_convex_polygon_on_grid :
  ∃ (p : ConvexPolygon), p.vertices.length = 561 ∧
  ∀ (q : ConvexPolygon), q.vertices.length ≤ 561 :=
sorry

end NUMINAMATH_CALUDE_largest_convex_polygon_on_grid_l248_24898


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_f_formula_correct_l248_24822

/-- Given two distinct real numbers α and β, we define a function f on natural numbers. -/
noncomputable def f (α β : ℝ) (n : ℕ) : ℝ :=
  (α^(n+1) - β^(n+1)) / (α - β)

/-- The main theorem stating that f satisfies the given recurrence relation and initial conditions. -/
theorem f_satisfies_conditions (α β : ℝ) (h : α ≠ β) :
  (f α β 1 = (α^2 - β^2) / (α - β)) ∧
  (f α β 2 = (α^3 - β^3) / (α - β)) ∧
  (∀ n : ℕ, f α β (n+2) = (α + β) * f α β (n+1) - α * β * f α β n) :=
by sorry

/-- The main theorem proving that the given formula for f is correct for all natural numbers. -/
theorem f_formula_correct (α β : ℝ) (h : α ≠ β) (n : ℕ) :
  f α β n = (α^(n+1) - β^(n+1)) / (α - β) :=
by sorry

end NUMINAMATH_CALUDE_f_satisfies_conditions_f_formula_correct_l248_24822


namespace NUMINAMATH_CALUDE_perimeter_MNO_value_l248_24891

/-- A right prism with regular hexagonal bases -/
structure HexagonalPrism where
  height : ℝ
  base_side_length : ℝ

/-- A point on an edge of the prism -/
structure EdgePoint where
  fraction : ℝ  -- Fraction of the edge length from the base

/-- Triangle MNO formed by three points on different edges of the prism -/
structure TriangleMNO where
  prism : HexagonalPrism
  m : EdgePoint
  n : EdgePoint
  o : EdgePoint

/-- Calculate the perimeter of triangle MNO -/
def perimeter_MNO (t : TriangleMNO) : ℝ :=
  sorry

theorem perimeter_MNO_value (t : TriangleMNO) 
  (h1 : t.prism.height = 20)
  (h2 : t.prism.base_side_length = 10)
  (h3 : t.m.fraction = 1/3)
  (h4 : t.n.fraction = 1/4)
  (h5 : t.o.fraction = 1/2) :
  perimeter_MNO t = 10 + Real.sqrt (925/9) + 5 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_perimeter_MNO_value_l248_24891


namespace NUMINAMATH_CALUDE_smallest_drama_club_size_l248_24866

theorem smallest_drama_club_size : ∃ n : ℕ, n > 0 ∧ 
  (∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = n ∧
    (22 * n < 100 * a) ∧ (100 * a < 27 * n) ∧
    (25 * n < 100 * b) ∧ (100 * b < 35 * n) ∧
    (35 * n < 100 * c) ∧ (100 * c < 45 * n)) ∧
  (∀ m : ℕ, m > 0 → m < n →
    ¬(∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧
      a + b + c = m ∧
      (22 * m < 100 * a) ∧ (100 * a < 27 * m) ∧
      (25 * m < 100 * b) ∧ (100 * b < 35 * m) ∧
      (35 * m < 100 * c) ∧ (100 * c < 45 * m))) ∧
  n = 9 :=
by sorry

end NUMINAMATH_CALUDE_smallest_drama_club_size_l248_24866


namespace NUMINAMATH_CALUDE_pink_flowers_in_bag_B_l248_24801

theorem pink_flowers_in_bag_B : 
  let bag_A_red : ℕ := 6
  let bag_A_pink : ℕ := 3
  let bag_B_red : ℕ := 2
  let bag_B_pink : ℕ := 7
  let total_flowers_A : ℕ := bag_A_red + bag_A_pink
  let total_flowers_B : ℕ := bag_B_red + bag_B_pink
  let prob_pink_A : ℚ := bag_A_pink / total_flowers_A
  let prob_pink_B : ℚ := bag_B_pink / total_flowers_B
  let overall_prob_pink : ℚ := (prob_pink_A + prob_pink_B) / 2
  overall_prob_pink = 5555555555555556 / 10000000000000000 →
  bag_B_pink = 7 :=
by sorry

end NUMINAMATH_CALUDE_pink_flowers_in_bag_B_l248_24801


namespace NUMINAMATH_CALUDE_sock_combination_count_l248_24803

/-- The number of ways to choose a pair of socks of different colors with at least one blue sock. -/
def sock_combinations (white brown blue : ℕ) : ℕ :=
  (blue * white) + (blue * brown)

/-- Theorem: Given 5 white socks, 3 brown socks, and 4 blue socks, there are 32 ways to choose
    a pair of socks of different colors with at least one blue sock. -/
theorem sock_combination_count :
  sock_combinations 5 3 4 = 32 := by
  sorry

end NUMINAMATH_CALUDE_sock_combination_count_l248_24803
