import Mathlib

namespace NUMINAMATH_CALUDE_car_profit_percent_l843_84319

/-- Calculate the profit percent from buying, repairing, and selling a car -/
theorem car_profit_percent (purchase_price repair_cost selling_price : ℚ) : 
  purchase_price = 48000 →
  repair_cost = 14000 →
  selling_price = 72900 →
  (selling_price - (purchase_price + repair_cost)) / (purchase_price + repair_cost) * 100 = 1758/100 := by
sorry

end NUMINAMATH_CALUDE_car_profit_percent_l843_84319


namespace NUMINAMATH_CALUDE_cow_count_is_24_l843_84332

/-- Represents the number of animals in the group -/
structure AnimalCount where
  ducks : ℕ
  cows : ℕ

/-- The total number of legs in the group -/
def totalLegs (a : AnimalCount) : ℕ := 2 * a.ducks + 4 * a.cows

/-- The total number of heads in the group -/
def totalHeads (a : AnimalCount) : ℕ := a.ducks + a.cows

/-- The condition given in the problem -/
def satisfiesCondition (a : AnimalCount) : Prop :=
  totalLegs a = 2 * totalHeads a + 48

theorem cow_count_is_24 (a : AnimalCount) (h : satisfiesCondition a) : a.cows = 24 := by
  sorry

end NUMINAMATH_CALUDE_cow_count_is_24_l843_84332


namespace NUMINAMATH_CALUDE_comic_books_total_l843_84324

theorem comic_books_total (jake_books : ℕ) (brother_difference : ℕ) : 
  jake_books = 36 → brother_difference = 15 → 
  jake_books + (jake_books + brother_difference) = 87 := by
  sorry

end NUMINAMATH_CALUDE_comic_books_total_l843_84324


namespace NUMINAMATH_CALUDE_tangent_line_proof_l843_84352

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + 5*x^2 - 5

-- Define the given line
def l₁ (z y : ℝ) : Prop := 2*z - 6*y + 1 = 0

-- Define the tangent line
def l₂ (x y : ℝ) : Prop := 3*x + y + 6 = 0

-- Theorem statement
theorem tangent_line_proof :
  ∃ (x₀ y₀ : ℝ),
    -- The point (x₀, y₀) lies on the curve
    f x₀ = y₀ ∧
    -- The tangent line passes through (x₀, y₀)
    l₂ x₀ y₀ ∧
    -- The slope of the tangent line at (x₀, y₀) is the derivative of f at x₀
    (3*x₀^2 + 10*x₀ = -3) ∧
    -- The two lines are perpendicular
    ∀ (z₁ y₁ z₂ y₂ : ℝ),
      l₁ z₁ y₁ ∧ l₁ z₂ y₂ ∧ z₁ ≠ z₂ →
      (y₁ - y₂) / (z₁ - z₂) * (-1/3) = -1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_proof_l843_84352


namespace NUMINAMATH_CALUDE_system_solution_l843_84314

theorem system_solution (a b : ℚ) : 
  (∃ b, 2 * 1 - b * 2 = 1) →
  (∃ a, a * 1 + 1 = 2) →
  (∃! x y : ℚ, a * x + y = 2 ∧ 2 * x - b * y = 1 ∧ x = 4/5 ∧ y = 6/5) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l843_84314


namespace NUMINAMATH_CALUDE_max_visible_faces_sum_l843_84379

/-- Represents a single die -/
structure Die :=
  (top : ℕ)
  (bottom : ℕ)
  (left : ℕ)
  (right : ℕ)
  (front : ℕ)
  (back : ℕ)

/-- The grid of dice -/
def DiceGrid := Matrix (Fin 10) (Fin 10) Die

/-- Condition: sum of dots on opposite faces is 7 -/
def oppositeFacesSum7 (d : Die) : Prop :=
  d.top + d.bottom = 7 ∧ d.left + d.right = 7 ∧ d.front + d.back = 7

/-- All dice in the grid satisfy the opposite faces sum condition -/
def allDiceSatisfyCondition (grid : DiceGrid) : Prop :=
  ∀ i j, oppositeFacesSum7 (grid i j)

/-- Count of visible faces -/
def visibleFacesCount : ℕ := 240

/-- Sum of dots on visible faces -/
def visibleFacesSum (grid : DiceGrid) : ℕ :=
  sorry  -- Definition would involve summing specific faces based on visibility

/-- Main theorem -/
theorem max_visible_faces_sum (grid : DiceGrid) 
  (h1 : allDiceSatisfyCondition grid) : 
  visibleFacesSum grid ≤ 920 :=
sorry

end NUMINAMATH_CALUDE_max_visible_faces_sum_l843_84379


namespace NUMINAMATH_CALUDE_simplified_expression_evaluation_l843_84372

theorem simplified_expression_evaluation (x y : ℝ) 
  (hx : x = -1) (hy : y = 1/2) : 
  2 * (3 * x^2 + x * y^2) - 3 * (2 * x * y^2 - x^2) - 10 * x^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_simplified_expression_evaluation_l843_84372


namespace NUMINAMATH_CALUDE_count_equal_f_is_501_l843_84318

/-- f(n) denotes the number of 1's in the base-2 representation of n -/
def f (n : ℕ) : ℕ := sorry

/-- Counts the number of integers n between 1 and 2002 (inclusive) where f(n) = f(n+1) -/
def count_equal_f : ℕ := sorry

theorem count_equal_f_is_501 : count_equal_f = 501 := by sorry

end NUMINAMATH_CALUDE_count_equal_f_is_501_l843_84318


namespace NUMINAMATH_CALUDE_probability_two_girls_chosen_l843_84348

-- Define the total number of members
def total_members : ℕ := 12

-- Define the number of girls
def num_girls : ℕ := 6

-- Define the number of boys
def num_boys : ℕ := 6

-- Define a function to calculate combinations
def combination (n k : ℕ) : ℕ := (n.factorial) / (k.factorial * (n - k).factorial)

-- Theorem statement
theorem probability_two_girls_chosen :
  (combination num_girls 2 : ℚ) / (combination total_members 2) = 5 / 22 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_girls_chosen_l843_84348


namespace NUMINAMATH_CALUDE_rectangle_ratio_square_l843_84326

theorem rectangle_ratio_square (a b : ℝ) (h : a > 0 ∧ b > 0 ∧ a < b) :
  let d := Real.sqrt (a^2 + b^2)
  (a / b = b / d) → (a / b)^2 = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_square_l843_84326


namespace NUMINAMATH_CALUDE_min_fraction_sum_l843_84325

def ValidDigits : Finset Nat := {1, 3, 4, 5, 6, 8, 9}

theorem min_fraction_sum (A B C D : Nat) 
  (hA : A ∈ ValidDigits) (hB : B ∈ ValidDigits) 
  (hC : C ∈ ValidDigits) (hD : D ∈ ValidDigits)
  (hAB : A ≠ B) (hAC : A ≠ C) (hAD : A ≠ D) 
  (hBC : B ≠ C) (hBD : B ≠ D) (hCD : C ≠ D)
  (hB_pos : B > 0) (hD_pos : D > 0) :
  (A : ℚ) / B + (C : ℚ) / D ≥ 11 / 24 := by
  sorry

end NUMINAMATH_CALUDE_min_fraction_sum_l843_84325


namespace NUMINAMATH_CALUDE_tyler_meal_choices_l843_84363

-- Define the number of options for each category
def num_meats : ℕ := 3
def num_vegetables : ℕ := 5
def num_desserts : ℕ := 4
def num_drinks : ℕ := 3

-- Define the number of vegetables to be chosen
def vegetables_to_choose : ℕ := 3

-- Theorem statement
theorem tyler_meal_choices :
  (num_meats) * (Nat.choose num_vegetables vegetables_to_choose) * (num_desserts) * (num_drinks) = 360 := by
  sorry


end NUMINAMATH_CALUDE_tyler_meal_choices_l843_84363


namespace NUMINAMATH_CALUDE_competition_necessarily_laughable_l843_84303

/-- Represents the number of questions in the math competition -/
def num_questions : ℕ := 10

/-- Represents the threshold for laughable performance -/
def laughable_threshold : ℕ := 57

/-- Represents the minimum number of students for which the performance is necessarily laughable -/
def min_laughable_students : ℕ := 253

/-- Represents a student's performance on the math competition -/
structure StudentPerformance where
  correct_answers : Finset (Fin num_questions)

/-- Represents the collective performance of students in the math competition -/
def Competition (n : ℕ) := Fin n → StudentPerformance

/-- Defines when a competition performance is laughable -/
def is_laughable (comp : Competition n) : Prop :=
  ∃ (i j : Fin num_questions), i ≠ j ∧
    (∃ (students : Finset (Fin n)), students.card = laughable_threshold ∧
      (∀ s ∈ students, (i ∈ (comp s).correct_answers ∧ j ∈ (comp s).correct_answers) ∨
                       (i ∉ (comp s).correct_answers ∧ j ∉ (comp s).correct_answers)))

/-- The main theorem: any competition with at least min_laughable_students is necessarily laughable -/
theorem competition_necessarily_laughable (n : ℕ) (h : n ≥ min_laughable_students) :
  ∀ (comp : Competition n), is_laughable comp :=
sorry

end NUMINAMATH_CALUDE_competition_necessarily_laughable_l843_84303


namespace NUMINAMATH_CALUDE_intersection_equality_l843_84305

def A (m : ℝ) : Set ℝ := {-1, 3, m}
def B : Set ℝ := {3, 4}

theorem intersection_equality (m : ℝ) : B ∩ A m = B → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_l843_84305


namespace NUMINAMATH_CALUDE_paula_and_karl_ages_sum_l843_84392

theorem paula_and_karl_ages_sum (P K : ℕ) : 
  (P - 5 = 3 * (K - 5)) →  -- 5 years ago, Paula was 3 times as old as Karl
  (P + 6 = 2 * (K + 6)) →  -- In 6 years, Paula will be twice as old as Karl
  P + K = 54 :=            -- The sum of their current ages is 54
by sorry

end NUMINAMATH_CALUDE_paula_and_karl_ages_sum_l843_84392


namespace NUMINAMATH_CALUDE_central_angle_for_given_arc_central_angle_proof_l843_84315

/-- Given a circle with radius 100mm and an arc length of 300mm,
    the central angle corresponding to this arc is 3 radians. -/
theorem central_angle_for_given_arc : ℝ → ℝ → ℝ → Prop :=
  λ radius arc_length angle =>
    radius = 100 ∧ arc_length = 300 → angle = 3

/-- The theorem proof -/
theorem central_angle_proof :
  ∃ (angle : ℝ), central_angle_for_given_arc 100 300 angle :=
by
  sorry

end NUMINAMATH_CALUDE_central_angle_for_given_arc_central_angle_proof_l843_84315


namespace NUMINAMATH_CALUDE_apple_theorem_l843_84311

def apple_problem (initial_apples : ℕ) : ℕ :=
  let after_jill := initial_apples - (initial_apples * 30 / 100)
  let after_june := after_jill - (after_jill * 20 / 100)
  let after_friend := after_june - 2
  after_friend - (after_friend * 10 / 100)

theorem apple_theorem :
  apple_problem 150 = 74 := by sorry

end NUMINAMATH_CALUDE_apple_theorem_l843_84311


namespace NUMINAMATH_CALUDE_celias_savings_l843_84380

def weeks : ℕ := 4
def food_budget_per_week : ℕ := 100
def rent : ℕ := 1500
def streaming_cost : ℕ := 30
def phone_cost : ℕ := 50
def savings_rate : ℚ := 1 / 10

def total_spending : ℕ := weeks * food_budget_per_week + rent + streaming_cost + phone_cost

def savings_amount : ℚ := (total_spending : ℚ) * savings_rate

theorem celias_savings : savings_amount = 198 := by
  sorry

end NUMINAMATH_CALUDE_celias_savings_l843_84380


namespace NUMINAMATH_CALUDE_battery_charging_time_l843_84313

/-- Represents the charging characteristics of a mobile battery -/
structure BatteryCharging where
  initial_rate : ℝ  -- Percentage charged per hour
  initial_time : ℝ  -- Time for initial charge in minutes
  additional_time : ℝ  -- Additional time to reach certain percentage in minutes

/-- Calculates the total charging time for a mobile battery -/
def total_charging_time (b : BatteryCharging) : ℝ :=
  b.initial_time + b.additional_time

/-- Theorem: The total charging time for the given battery is 255 minutes -/
theorem battery_charging_time :
  let b : BatteryCharging := {
    initial_rate := 20,
    initial_time := 60,
    additional_time := 195
  }
  total_charging_time b = 255 := by
  sorry

end NUMINAMATH_CALUDE_battery_charging_time_l843_84313


namespace NUMINAMATH_CALUDE_correct_average_weight_l843_84384

theorem correct_average_weight 
  (num_boys : ℕ) 
  (initial_avg : ℝ) 
  (misread_weight : ℝ) 
  (correct_weight : ℝ) : 
  num_boys = 20 → 
  initial_avg = 58.4 → 
  misread_weight = 56 → 
  correct_weight = 65 → 
  (num_boys * initial_avg + correct_weight - misread_weight) / num_boys = 58.85 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_weight_l843_84384


namespace NUMINAMATH_CALUDE_solution_value_l843_84308

theorem solution_value (m n : ℝ) : 
  (∀ x, x^2 - m*x + n ≤ 0 ↔ -5 ≤ x ∧ x ≤ 1) →
  ((-5)^2 - m*(-5) + n = 0) →
  (1^2 - m*1 + n = 0) →
  m - n = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_solution_value_l843_84308


namespace NUMINAMATH_CALUDE_equivalence_point_cost_effectiveness_l843_84387

-- Define the full ticket price
def full_price : ℝ := 240

-- Define the charge functions for Travel Agency A and B
def charge_A (x : ℝ) : ℝ := 120 * x + 240
def charge_B (x : ℝ) : ℝ := 144 * x + 144

-- Theorem for the equivalence point
theorem equivalence_point :
  ∃ x : ℝ, charge_A x = charge_B x ∧ x = 4 := by sorry

-- Theorem for cost-effectiveness comparison
theorem cost_effectiveness (x : ℝ) :
  (x < 4 → charge_B x < charge_A x) ∧
  (x > 4 → charge_A x < charge_B x) := by sorry

end NUMINAMATH_CALUDE_equivalence_point_cost_effectiveness_l843_84387


namespace NUMINAMATH_CALUDE_chef_potato_count_l843_84333

/-- The number of potatoes a chef needs to cook -/
def total_potatoes (cooked : ℕ) (cooking_time_per_potato : ℕ) (remaining_cooking_time : ℕ) : ℕ :=
  cooked + remaining_cooking_time / cooking_time_per_potato

/-- Proof that the chef needs to cook 13 potatoes in total -/
theorem chef_potato_count : total_potatoes 5 6 48 = 13 := by
  sorry

#eval total_potatoes 5 6 48

end NUMINAMATH_CALUDE_chef_potato_count_l843_84333


namespace NUMINAMATH_CALUDE_bill_face_value_l843_84316

/-- Calculates the face value of a bill given the true discount, interest rate, and time period. -/
def calculate_face_value (true_discount : ℚ) (interest_rate : ℚ) (time_months : ℚ) : ℚ :=
  (true_discount * 100) / (interest_rate * (time_months / 12))

/-- Proves that the face value of the bill is 1575 given the specified conditions. -/
theorem bill_face_value :
  let true_discount : ℚ := 189
  let interest_rate : ℚ := 16
  let time_months : ℚ := 9
  calculate_face_value true_discount interest_rate time_months = 1575 := by
  sorry

#eval calculate_face_value 189 16 9

end NUMINAMATH_CALUDE_bill_face_value_l843_84316


namespace NUMINAMATH_CALUDE_possible_square_values_l843_84343

/-- Represents a tiling of a 9x7 rectangle using L-trominoes and 2x2 squares. -/
structure Tiling :=
  (num_squares : ℕ)
  (num_trominoes : ℕ)

/-- The area of the rectangle is 63. -/
axiom rectangle_area : 63 = 9 * 7

/-- The area of a 2x2 square is 4. -/
axiom square_area : 4 = 2 * 2

/-- The area of an L-tromino is 3. -/
axiom tromino_area : 3 = 3

/-- The total area covered by tiles equals the rectangle area. -/
axiom area_equation (t : Tiling) : 4 * t.num_squares + 3 * t.num_trominoes = 63

/-- The number of 2x2 squares is a multiple of 3. -/
axiom squares_multiple_of_three (t : Tiling) : ∃ k : ℕ, t.num_squares = 3 * k

/-- The number of 2x2 squares is at most 3. -/
axiom max_squares (t : Tiling) : t.num_squares ≤ 3

/-- The possible values for the number of 2x2 squares are 0 and 3. -/
theorem possible_square_values (t : Tiling) : t.num_squares = 0 ∨ t.num_squares = 3 :=
sorry

end NUMINAMATH_CALUDE_possible_square_values_l843_84343


namespace NUMINAMATH_CALUDE_juice_box_days_l843_84395

theorem juice_box_days (num_children : ℕ) (school_weeks : ℕ) (total_juice_boxes : ℕ) :
  num_children = 3 →
  school_weeks = 25 →
  total_juice_boxes = 375 →
  (total_juice_boxes / (num_children * school_weeks) : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_juice_box_days_l843_84395


namespace NUMINAMATH_CALUDE_earthworm_investment_theorem_l843_84353

/-- Represents the earthworm investment scenario -/
structure EarthwormInvestment where
  okeydokey_apples : ℕ
  okeydokey_worms : ℕ
  artichokey_apples : ℕ
  total_worms : ℕ

/-- The earthworm investment theorem -/
theorem earthworm_investment_theorem (e : EarthwormInvestment) 
  (h1 : e.okeydokey_apples = 5)
  (h2 : e.okeydokey_worms = 25)
  (h3 : e.artichokey_apples = 7)
  (h4 : e.okeydokey_worms * e.artichokey_apples = e.okeydokey_apples * (e.total_worms - e.okeydokey_worms)) :
  e.total_worms = 60 := by
  sorry

#check earthworm_investment_theorem

end NUMINAMATH_CALUDE_earthworm_investment_theorem_l843_84353


namespace NUMINAMATH_CALUDE_existence_of_special_integers_l843_84371

theorem existence_of_special_integers : ∃ (a b c : ℤ), 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧   -- nonzero integers
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧   -- pairwise distinct
  a + b + c = 0 ∧           -- sum is zero
  ∃ (n : ℕ), a^13 + b^13 + c^13 = n^2  -- sum of 13th powers is a perfect square
  := by sorry

end NUMINAMATH_CALUDE_existence_of_special_integers_l843_84371


namespace NUMINAMATH_CALUDE_sixteen_squares_covered_l843_84396

/-- Represents a square on the checkerboard -/
structure Square where
  x : Int
  y : Int

/-- Represents the circular disc -/
structure Disc where
  diameter : ℝ
  center : Square

/-- Represents the checkerboard -/
structure Checkerboard where
  size : Nat
  squares : List Square

/-- Checks if a square is completely covered by the disc -/
def is_covered (s : Square) (d : Disc) : Bool :=
  sorry

/-- Counts the number of squares completely covered by the disc -/
def count_covered_squares (cb : Checkerboard) (d : Disc) : Nat :=
  sorry

/-- Main theorem: 16 squares are completely covered -/
theorem sixteen_squares_covered (cb : Checkerboard) (d : Disc) : 
  cb.size = 6 → d.diameter = 2 → count_covered_squares cb d = 16 := by
  sorry

end NUMINAMATH_CALUDE_sixteen_squares_covered_l843_84396


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l843_84361

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x + 2) ↔ x ≥ -2 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l843_84361


namespace NUMINAMATH_CALUDE_gcd_fx_x_l843_84312

def f (x : ℤ) : ℤ := (3*x+4)*(5*x+6)*(11*x+9)*(x+7)

theorem gcd_fx_x (x : ℤ) (h : ∃ k : ℤ, x = 35622 * k) : 
  Nat.gcd (Int.natAbs (f x)) (Int.natAbs x) = 378 := by
  sorry

end NUMINAMATH_CALUDE_gcd_fx_x_l843_84312


namespace NUMINAMATH_CALUDE_discount_equation_l843_84388

theorem discount_equation (original_price final_price : ℝ) (x : ℝ) 
  (h1 : original_price = 200)
  (h2 : final_price = 164)
  (h3 : final_price = original_price * (1 - x)^2) :
  200 * (1 - x)^2 = 164 := by
  sorry

end NUMINAMATH_CALUDE_discount_equation_l843_84388


namespace NUMINAMATH_CALUDE_angle_with_complement_40percent_of_supplement_is_30_degrees_l843_84304

theorem angle_with_complement_40percent_of_supplement_is_30_degrees :
  ∀ x : ℝ,
  (x > 0) →
  (x < 90) →
  (90 - x = (2/5) * (180 - x)) →
  x = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_angle_with_complement_40percent_of_supplement_is_30_degrees_l843_84304


namespace NUMINAMATH_CALUDE_max_non_managers_proof_l843_84393

structure Department where
  name : String
  managers : ℕ
  ratio_managers : ℕ
  ratio_non_managers : ℕ
  active_projects : ℕ

def calculate_non_managers (d : Department) : ℕ :=
  d.managers * d.ratio_non_managers / d.ratio_managers +
  (d.active_projects + 2) / 3 +
  2

def total_non_managers (departments : List Department) : ℕ :=
  departments.foldl (fun acc d => acc + calculate_non_managers d) 0

theorem max_non_managers_proof :
  let departments : List Department := [
    { name := "Marketing", managers := 9, ratio_managers := 9, ratio_non_managers := 38, active_projects := 6 },
    { name := "HR", managers := 5, ratio_managers := 5, ratio_non_managers := 23, active_projects := 4 },
    { name := "Finance", managers := 6, ratio_managers := 6, ratio_non_managers := 31, active_projects := 5 }
  ]
  total_non_managers departments = 104 :=
by sorry

end NUMINAMATH_CALUDE_max_non_managers_proof_l843_84393


namespace NUMINAMATH_CALUDE_cube_with_holes_surface_area_l843_84350

/-- Represents a cube with holes cut through each face -/
structure CubeWithHoles where
  edge_length : ℝ
  hole_side_length : ℝ

/-- Calculates the total surface area of a cube with holes, including inside surfaces -/
def total_surface_area (cube : CubeWithHoles) : ℝ :=
  let original_surface_area := 6 * cube.edge_length^2
  let area_removed_by_holes := 6 * cube.hole_side_length^2
  let new_exposed_area := 6 * 6 * cube.hole_side_length^2
  original_surface_area - area_removed_by_holes + new_exposed_area

/-- Theorem stating that a cube with edge length 5 and hole side length 2 has total surface area 270 -/
theorem cube_with_holes_surface_area :
  total_surface_area { edge_length := 5, hole_side_length := 2 } = 270 := by
  sorry

end NUMINAMATH_CALUDE_cube_with_holes_surface_area_l843_84350


namespace NUMINAMATH_CALUDE_distribute_five_into_three_l843_84377

/-- The number of ways to distribute n distinct objects into k distinct bins,
    where each bin must contain at least one object. -/
def distribute (n k : ℕ) : ℕ :=
  (Nat.choose n k + (Nat.choose n 2 * Nat.choose 3 2) / 2) * Nat.factorial k

/-- The theorem stating that distributing 5 distinct objects into 3 distinct bins,
    where each bin must contain at least one object, results in 150 ways. -/
theorem distribute_five_into_three :
  distribute 5 3 = 150 := by sorry

end NUMINAMATH_CALUDE_distribute_five_into_three_l843_84377


namespace NUMINAMATH_CALUDE_room_width_l843_84321

/-- 
Given a rectangular room with length 20 feet and 1 foot longer than its width,
prove that the width of the room is 19 feet.
-/
theorem room_width (length width : ℕ) : 
  length = 20 ∧ length = width + 1 → width = 19 := by
  sorry

end NUMINAMATH_CALUDE_room_width_l843_84321


namespace NUMINAMATH_CALUDE_estimate_wildlife_population_l843_84359

/-- Estimate the total number of animals in a wildlife reserve using the mark-recapture method. -/
theorem estimate_wildlife_population
  (initial_catch : ℕ)
  (second_catch : ℕ)
  (marked_in_second : ℕ)
  (h1 : initial_catch = 1200)
  (h2 : second_catch = 1000)
  (h3 : marked_in_second = 100) :
  (initial_catch * second_catch) / marked_in_second = 12000 :=
by sorry

end NUMINAMATH_CALUDE_estimate_wildlife_population_l843_84359


namespace NUMINAMATH_CALUDE_quadratic_three_axis_intersections_l843_84355

/-- A quadratic function f(x) = kx² - 4x - 3 has three common points with the coordinate axes if and only if k > -4/3 and k ≠ 0 -/
theorem quadratic_three_axis_intersections (k : ℝ) :
  (∃ x₁ x₂ y : ℝ, x₁ ≠ x₂ ∧ 
    (k * x₁^2 - 4 * x₁ - 3 = 0) ∧ 
    (k * x₂^2 - 4 * x₂ - 3 = 0) ∧ 
    (k * 0^2 - 4 * 0 - 3 = y)) ↔ 
  (k > -4/3 ∧ k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_three_axis_intersections_l843_84355


namespace NUMINAMATH_CALUDE_f_inequality_solutions_l843_84375

/-- The function f(x) = (x-a)(x-2) -/
def f (a : ℝ) (x : ℝ) : ℝ := (x - a) * (x - 2)

theorem f_inequality_solutions :
  (∀ x, f 1 x > 0 ↔ x ∈ Set.Ioi 2 ∪ Set.Iic 1) ∧
  (∀ x, f 2 x < 0 → False) ∧
  (∀ a, a > 2 → ∀ x, f a x < 0 ↔ x ∈ Set.Ioo 2 a) ∧
  (∀ a, a < 2 → ∀ x, f a x < 0 ↔ x ∈ Set.Ioo a 2) :=
by sorry

end NUMINAMATH_CALUDE_f_inequality_solutions_l843_84375


namespace NUMINAMATH_CALUDE_inequality_always_true_iff_a_in_range_l843_84394

theorem inequality_always_true_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ a ∈ Set.Ioc (-2) 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_always_true_iff_a_in_range_l843_84394


namespace NUMINAMATH_CALUDE_annulus_area_l843_84389

theorem annulus_area (R r s : ℝ) (h1 : R > r) (h2 : R^2 - r^2 = s^2) :
  π * s^2 = π * R^2 - π * r^2 := by
  sorry

end NUMINAMATH_CALUDE_annulus_area_l843_84389


namespace NUMINAMATH_CALUDE_triangle_area_with_sides_17_17_16_prove_triangle_area_with_sides_17_17_16_l843_84306

/-- The area of a triangle with two sides of length 17 and one side of length 16 is 120 -/
theorem triangle_area_with_sides_17_17_16 : ℝ → Prop :=
  fun area =>
    ∀ (D E F : ℝ × ℝ),
      let de := Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2)
      let ef := Real.sqrt ((F.1 - E.1)^2 + (F.2 - E.2)^2)
      let df := Real.sqrt ((F.1 - D.1)^2 + (F.2 - D.2)^2)
      de = 17 ∧ ef = 17 ∧ df = 16 →
      area = 120

/-- Proof of the theorem -/
theorem prove_triangle_area_with_sides_17_17_16 : triangle_area_with_sides_17_17_16 120 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_with_sides_17_17_16_prove_triangle_area_with_sides_17_17_16_l843_84306


namespace NUMINAMATH_CALUDE_f_inequality_l843_84309

-- Define the function f and its derivative f'
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Define the condition that f' is the derivative of f
variable (h_deriv : ∀ x, HasDerivAt f (f' x) x)

-- Define the condition that f(x) > f'(x) for all x
variable (h_cond : ∀ x, f x > f' x)

-- State the theorem to be proved
theorem f_inequality : 3 * f (Real.log 2) > 2 * f (Real.log 3) := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l843_84309


namespace NUMINAMATH_CALUDE_casper_candy_problem_l843_84340

def candy_sequence (initial : ℕ) : List ℕ :=
  let day1 := initial / 2 - 3
  let day2 := day1 / 2 - 5
  let day3 := day2 / 2 - 2
  let day4 := day3 / 2
  [initial, day1, day2, day3, day4]

theorem casper_candy_problem (initial : ℕ) :
  candy_sequence initial = [initial, initial / 2 - 3, (initial / 2 - 3) / 2 - 5, ((initial / 2 - 3) / 2 - 5) / 2 - 2, 10] →
  initial = 122 := by
  sorry

end NUMINAMATH_CALUDE_casper_candy_problem_l843_84340


namespace NUMINAMATH_CALUDE_range_of_f_l843_84368

-- Define the function
def f (x : ℝ) : ℝ := -x^2 + 4*x - 1

-- Define the domain
def Domain : Set ℝ := { x | -1 ≤ x ∧ x ≤ 3 }

-- State the theorem
theorem range_of_f :
  { y | ∃ x ∈ Domain, f x = y } = { y | -6 ≤ y ∧ y ≤ 3 } :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l843_84368


namespace NUMINAMATH_CALUDE_simplify_expression_l843_84301

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : a^4 + b^4 = a + b) (h2 : a^2 + b^2 = 2) :
  a^2 / b^2 + b^2 / a^2 - 1 / (a^2 * b^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l843_84301


namespace NUMINAMATH_CALUDE_midpoint_coordinates_l843_84391

/-- Given a segment with endpoints A(x₁, y₁) and B(x₂, y₂), and its midpoint M(x₀, y₀),
    prove that the coordinates of the midpoint are the averages of the endpoints' coordinates. -/
theorem midpoint_coordinates (x₀ x₁ x₂ y₀ y₁ y₂ : ℝ) :
  (∀ t : ℝ, t ∈ (Set.Icc 0 1) → 
    (x₀ = (1 - t) * x₁ + t * x₂ ∧ 
     y₀ = (1 - t) * y₁ + t * y₂)) →
  x₀ = (x₁ + x₂) / 2 ∧ y₀ = (y₁ + y₂) / 2 := by
  sorry


end NUMINAMATH_CALUDE_midpoint_coordinates_l843_84391


namespace NUMINAMATH_CALUDE_section_area_theorem_l843_84356

/-- Represents a regular quadrilateral pyramid -/
structure RegularQuadPyramid where
  lateral_area : ℝ

/-- Represents a plane intersecting the pyramid -/
structure IntersectingPlane where
  opposite_face_area : ℝ

/-- Calculates the lateral surface area of the section cut off by the plane -/
def section_lateral_area (pyramid : RegularQuadPyramid) (plane : IntersectingPlane) : ℝ :=
  sorry

/-- Theorem statement -/
theorem section_area_theorem (pyramid : RegularQuadPyramid) (plane : IntersectingPlane) :
  pyramid.lateral_area = 25 ∧ plane.opposite_face_area = 4 →
  section_lateral_area pyramid plane = 20.25 :=
sorry

end NUMINAMATH_CALUDE_section_area_theorem_l843_84356


namespace NUMINAMATH_CALUDE_f_at_five_l843_84329

def f (x : ℝ) : ℝ := 3 * x^4 - 22 * x^3 + 51 * x^2 - 58 * x + 24

theorem f_at_five : f 5 = 134 := by sorry

end NUMINAMATH_CALUDE_f_at_five_l843_84329


namespace NUMINAMATH_CALUDE_smallest_vector_norm_l843_84369

/-- Given a vector v such that ||v + (4, 2)|| = 10, 
    the smallest possible value of ||v|| is 10 - 2√5 -/
theorem smallest_vector_norm (v : ℝ × ℝ) 
    (h : ‖v + (4, 2)‖ = 10) : 
  ∃ (w : ℝ × ℝ), ‖w‖ = 10 - 2 * Real.sqrt 5 ∧ 
    ∀ (u : ℝ × ℝ), ‖u + (4, 2)‖ = 10 → ‖u‖ ≥ ‖w‖ := by
  sorry


end NUMINAMATH_CALUDE_smallest_vector_norm_l843_84369


namespace NUMINAMATH_CALUDE_fibonacci_type_sequence_count_l843_84310

/-- A Fibonacci-type sequence is an infinite sequence of integers where each term is the sum of the two preceding ones. -/
def FibonacciTypeSequence (a : ℤ → ℤ) : Prop :=
  ∀ n : ℤ, a n = a (n - 1) + a (n - 2)

/-- Count of Fibonacci-type sequences with two consecutive terms strictly positive and ≤ N -/
def countFibonacciTypeSequences (N : ℕ) : ℕ :=
  if N % 2 = 0 then
    (N / 2) * (N / 2 + 1)
  else
    ((N + 1) / 2) ^ 2

theorem fibonacci_type_sequence_count (N : ℕ) :
  (∃ a : ℤ → ℤ, FibonacciTypeSequence a ∧
    ∃ n : ℤ, 0 < a n ∧ 0 < a (n + 1) ∧ a n ≤ N ∧ a (n + 1) ≤ N) →
  countFibonacciTypeSequences N = 
    if N % 2 = 0 then
      (N / 2) * (N / 2 + 1)
    else
      ((N + 1) / 2) ^ 2 :=
by sorry

#check fibonacci_type_sequence_count

end NUMINAMATH_CALUDE_fibonacci_type_sequence_count_l843_84310


namespace NUMINAMATH_CALUDE_exists_n_for_digit_sum_ratio_l843_84330

/-- S(a) denotes the sum of the digits of the natural number a -/
def digit_sum (a : ℕ) : ℕ := sorry

/-- Theorem stating that for any natural number R, there exists a natural number n 
    such that the ratio of the digit sum of n^2 to the digit sum of n equals R -/
theorem exists_n_for_digit_sum_ratio (R : ℕ) : 
  ∃ n : ℕ, (digit_sum (n^2) : ℚ) / (digit_sum n : ℚ) = R := by sorry

end NUMINAMATH_CALUDE_exists_n_for_digit_sum_ratio_l843_84330


namespace NUMINAMATH_CALUDE_base_eight_solution_l843_84337

/-- Converts a list of digits in base h to its decimal representation -/
def to_decimal (digits : List Nat) (h : Nat) : Nat :=
  digits.foldl (fun acc d => acc * h + d) 0

/-- Checks if the equation holds for a given base h -/
def equation_holds (h : Nat) : Prop :=
  to_decimal [9, 8, 7, 6, 5, 4] h + to_decimal [6, 9, 8, 5, 5, 5] h = to_decimal [1, 7, 9, 6, 2, 2, 9] h

theorem base_eight_solution :
  ∃ (h : Nat), h > 0 ∧ equation_holds h ∧ ∀ (k : Nat), k > 0 ∧ equation_holds k → k = h :=
by
  sorry

end NUMINAMATH_CALUDE_base_eight_solution_l843_84337


namespace NUMINAMATH_CALUDE_hyperbola_center_is_3_5_l843_84373

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  4 * x^2 - 24 * x - 25 * y^2 + 250 * y - 489 = 0

/-- The center of the hyperbola -/
def hyperbola_center : ℝ × ℝ := (3, 5)

/-- Theorem: The center of the hyperbola is (3, 5) -/
theorem hyperbola_center_is_3_5 :
  ∀ (x y : ℝ), hyperbola_equation x y →
  ∃ (a b : ℝ), (x - hyperbola_center.1)^2 / a^2 - (y - hyperbola_center.2)^2 / b^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_center_is_3_5_l843_84373


namespace NUMINAMATH_CALUDE_platform_length_l843_84370

/-- The length of a platform given train parameters -/
theorem platform_length (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ) :
  train_length = 150 →
  train_speed_kmph = 75 →
  crossing_time = 24 →
  ∃ (platform_length : ℝ),
    platform_length = 350 ∧
    platform_length = (train_speed_kmph * 1000 / 3600 * crossing_time) - train_length :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l843_84370


namespace NUMINAMATH_CALUDE_largest_integer_solution_l843_84398

theorem largest_integer_solution (x : ℤ) : (3 - 2 * x > 0) → x ≤ 1 ∧ (∀ y : ℤ, 3 - 2 * y > 0 → y ≤ x) :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_solution_l843_84398


namespace NUMINAMATH_CALUDE_dave_deleted_eleven_apps_l843_84334

/-- The number of apps Dave deleted -/
def apps_deleted (initial_apps : ℕ) (remaining_apps : ℕ) : ℕ :=
  initial_apps - remaining_apps

/-- Theorem stating that Dave deleted 11 apps -/
theorem dave_deleted_eleven_apps : apps_deleted 16 5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_dave_deleted_eleven_apps_l843_84334


namespace NUMINAMATH_CALUDE_alternating_walk_forms_cycle_l843_84362

/-- Represents a direction of turn -/
inductive Direction
| Left
| Right

/-- Represents the island as a graph -/
structure Island where
  -- The set of vertices (junctions)
  vertices : Type
  -- The edges (roads) between vertices
  edges : vertices → vertices → Prop
  -- Every vertex has exactly three edges
  three_roads : ∀ v : vertices, ∃! (n : Nat), n = 3 ∧ (∃ (adjacent : Finset vertices), adjacent.card = n ∧ ∀ u ∈ adjacent, edges v u)

/-- Represents a walk on the island -/
def Walk (island : Island) : Type :=
  Nat → island.vertices × Direction

/-- A walk is alternating if it alternates between left and right turns -/
def IsAlternating (walk : Walk island) : Prop :=
  ∀ n : Nat, 
    (walk n).2 ≠ (walk (n + 1)).2

/-- The main theorem: any alternating walk on a finite island will eventually form a cycle -/
theorem alternating_walk_forms_cycle (island : Island) (walk : Walk island) 
    (finite_island : Finite island.vertices) (alternating : IsAlternating walk) : 
    ∃ (start finish : Nat), start < finish ∧ (walk start).1 = (walk finish).1 := by
  sorry


end NUMINAMATH_CALUDE_alternating_walk_forms_cycle_l843_84362


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_factorial_sum_l843_84357

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem largest_prime_divisor_of_factorial_sum :
  ∃ (p : ℕ), p.Prime ∧ p ∣ (factorial 13 + factorial 14) ∧
  ∀ (q : ℕ), q.Prime → q ∣ (factorial 13 + factorial 14) → q ≤ p :=
by
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_factorial_sum_l843_84357


namespace NUMINAMATH_CALUDE_relationship_abc_l843_84386

theorem relationship_abc (a b c : ℝ) : 
  a = Real.sqrt 2 → b = 2^(0.8 : ℝ) → c = 2 * Real.log 2 / Real.log 5 → c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l843_84386


namespace NUMINAMATH_CALUDE_crayons_difference_l843_84376

/-- Given the initial number of crayons, the number of crayons given away, and the number of crayons lost,
    prove that the difference between the number of crayons lost and the number of crayons given away is 322. -/
theorem crayons_difference (initial : ℕ) (given_away : ℕ) (lost : ℕ)
    (h1 : initial = 110)
    (h2 : given_away = 90)
    (h3 : lost = 412) :
    lost - given_away = 322 := by
  sorry

end NUMINAMATH_CALUDE_crayons_difference_l843_84376


namespace NUMINAMATH_CALUDE_tooth_fairy_payment_l843_84345

theorem tooth_fairy_payment (total_money : ℕ) (teeth_count : ℕ) (h1 : total_money = 54) (h2 : teeth_count = 18) :
  total_money / teeth_count = 3 := by
sorry

end NUMINAMATH_CALUDE_tooth_fairy_payment_l843_84345


namespace NUMINAMATH_CALUDE_half_angle_quadrant_l843_84336

-- Define the concept of quadrant
def in_first_quadrant (θ : ℝ) : Prop := 0 < θ ∧ θ < Real.pi / 2
def in_third_quadrant (θ : ℝ) : Prop := Real.pi < θ ∧ θ < 3 * Real.pi / 2

theorem half_angle_quadrant (α : ℝ) :
  in_first_quadrant α → in_first_quadrant (α / 2) ∨ in_third_quadrant (α / 2) := by
  sorry

end NUMINAMATH_CALUDE_half_angle_quadrant_l843_84336


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_A_necessary_not_sufficient_for_B_l843_84360

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 8 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x + 1 - m^2 ≤ 0}

-- Part 1
theorem intersection_A_complement_B : 
  (A ∩ (Set.univ \ B 2)) = {x | -2 ≤ x ∧ x < -1 ∨ 3 < x ∧ x ≤ 4} := by sorry

-- Part 2
theorem A_necessary_not_sufficient_for_B :
  (∀ x, x ∈ A → x ∈ B m) ∧ (∃ x, x ∈ B m ∧ x ∉ A) ↔ 0 < m ∧ m < 3 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_A_necessary_not_sufficient_for_B_l843_84360


namespace NUMINAMATH_CALUDE_cylinder_volume_equals_cube_surface_l843_84339

/-- The volume of a cylinder with surface area equal to a cube of side length 4 and height equal to its diameter --/
theorem cylinder_volume_equals_cube_surface (π : ℝ) (h : π > 0) : 
  ∃ (r : ℝ), r > 0 ∧ 
  6 * π * r^2 = 96 ∧ 
  π * r^2 * (2 * r) = 128 * Real.sqrt 2 / π :=
sorry

end NUMINAMATH_CALUDE_cylinder_volume_equals_cube_surface_l843_84339


namespace NUMINAMATH_CALUDE_max_value_fraction_l843_84300

theorem max_value_fraction (x y : ℝ) (hx : -4 ≤ x ∧ x ≤ -2) (hy : 2 ≤ y ∧ y ≤ 4) :
  (x + y) / x ≤ 1/2 := by
sorry

end NUMINAMATH_CALUDE_max_value_fraction_l843_84300


namespace NUMINAMATH_CALUDE_ceasar_pages_read_l843_84374

/-- The number of pages Ceasar has already read -/
def pages_read (total_pages remaining_pages : ℕ) : ℕ :=
  total_pages - remaining_pages

theorem ceasar_pages_read :
  pages_read 563 416 = 147 := by
  sorry

end NUMINAMATH_CALUDE_ceasar_pages_read_l843_84374


namespace NUMINAMATH_CALUDE_line_segment_both_symmetric_l843_84367

-- Define the shapes
inductive Shape
  | EquilateralTriangle
  | IsoscelesTriangle
  | Parallelogram
  | LineSegment

-- Define symmetry properties
def isCentrallySymmetric (s : Shape) : Prop :=
  match s with
  | Shape.EquilateralTriangle => False
  | Shape.IsoscelesTriangle => False
  | Shape.Parallelogram => True
  | Shape.LineSegment => True

def isAxiallySymmetric (s : Shape) : Prop :=
  match s with
  | Shape.EquilateralTriangle => True
  | Shape.IsoscelesTriangle => True
  | Shape.Parallelogram => False
  | Shape.LineSegment => True

-- Theorem statement
theorem line_segment_both_symmetric :
  ∀ s : Shape, (isCentrallySymmetric s ∧ isAxiallySymmetric s) ↔ s = Shape.LineSegment :=
by sorry

end NUMINAMATH_CALUDE_line_segment_both_symmetric_l843_84367


namespace NUMINAMATH_CALUDE_burger_cost_l843_84342

theorem burger_cost (burger soda : ℕ) 
  (alice_purchase : 4 * burger + 3 * soda = 440)
  (bob_purchase : 3 * burger + 2 * soda = 330) :
  burger = 110 := by
sorry

end NUMINAMATH_CALUDE_burger_cost_l843_84342


namespace NUMINAMATH_CALUDE_f_value_at_one_l843_84338

/-- A function f : ℝ → ℝ is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- A function g : ℝ → ℝ is odd if g(-x) = -g(x) for all x -/
def IsOdd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

theorem f_value_at_one
  (f g : ℝ → ℝ)
  (h_even : IsEven f)
  (h_odd : IsOdd g)
  (h_eq : ∀ x, f x - g x = x^2 - x + 1) :
  f 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_one_l843_84338


namespace NUMINAMATH_CALUDE_triangle_side_difference_range_l843_84382

theorem triangle_side_difference_range (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Side lengths are positive
  a = 1 ∧  -- Given condition
  C - B = π / 2 ∧  -- Given condition
  a / Real.sin A = b / Real.sin B ∧  -- Law of Sines
  b / Real.sin B = c / Real.sin C  -- Law of Sines
  → Real.sqrt 2 / 2 < c - b ∧ c - b < 1 := by sorry

end NUMINAMATH_CALUDE_triangle_side_difference_range_l843_84382


namespace NUMINAMATH_CALUDE_six_six_six_triangle_l843_84383

/-- Triangle Inequality Theorem: A set of three positive real numbers can form a triangle
    if and only if the sum of any two is greater than the third. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- The set (6, 6, 6) can form a triangle. -/
theorem six_six_six_triangle : can_form_triangle 6 6 6 := by
  sorry


end NUMINAMATH_CALUDE_six_six_six_triangle_l843_84383


namespace NUMINAMATH_CALUDE_right_triangle_shorter_leg_l843_84364

theorem right_triangle_shorter_leg : 
  ∀ (a b c : ℕ), 
    a^2 + b^2 = c^2 →  -- Pythagorean theorem
    c = 65 →  -- hypotenuse length
    a ≤ b →  -- a is the shorter leg
    a = 16 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_shorter_leg_l843_84364


namespace NUMINAMATH_CALUDE_bridge_length_l843_84317

/-- The length of a bridge given train parameters -/
theorem bridge_length (train_length : ℝ) (train_speed : ℝ) (crossing_time : ℝ) :
  train_length = 110 →
  train_speed = 20 →
  crossing_time = 12.099 →
  (train_speed * crossing_time) - train_length = 131.98 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l843_84317


namespace NUMINAMATH_CALUDE_l₁_passes_through_point_distance_when_parallel_l843_84365

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := (a + 2) * x + y + a + 1 = 0
def l₂ (a x y : ℝ) : Prop := 3 * x + a * y - 2 * a = 0

-- Statement 1: l₁ always passes through (-1, 1)
theorem l₁_passes_through_point (a : ℝ) : l₁ a (-1) 1 := by sorry

-- Helper function to check if lines are parallel
def parallel (a : ℝ) : Prop := a + 2 = 3 / a

-- Statement 2: When l₁ and l₂ are parallel, their distance is 2√10/5
theorem distance_when_parallel (a : ℝ) (h : parallel a) :
  ∃ d : ℝ, d = (2 * Real.sqrt 10) / 5 ∧ 
  (∀ x y : ℝ, l₁ a x y ↔ l₂ a (x + d * 3 / 5) (y - d * 4 / 5)) := by sorry

end NUMINAMATH_CALUDE_l₁_passes_through_point_distance_when_parallel_l843_84365


namespace NUMINAMATH_CALUDE_defective_books_relative_frequency_l843_84349

/-- The relative frequency of an event is the ratio of the number of times 
    the event occurs to the total number of trials or experiments. -/
def relative_frequency (event_occurrences : ℕ) (total_trials : ℕ) : ℚ :=
  event_occurrences / total_trials

/-- Given a batch of 100 randomly selected books with 5 defective books,
    prove that the relative frequency of defective books is 0.05. -/
theorem defective_books_relative_frequency :
  let total_books : ℕ := 100
  let defective_books : ℕ := 5
  relative_frequency defective_books total_books = 5 / 100 := by
  sorry

#eval (5 : ℚ) / 100  -- To verify the result is indeed 0.05

end NUMINAMATH_CALUDE_defective_books_relative_frequency_l843_84349


namespace NUMINAMATH_CALUDE_units_digit_of_seven_power_l843_84390

theorem units_digit_of_seven_power (n : ℕ) : 7^(6^5) ≡ 1 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_seven_power_l843_84390


namespace NUMINAMATH_CALUDE_cube_root_of_negative_eight_squared_l843_84358

theorem cube_root_of_negative_eight_squared :
  ((-8^2 : ℝ) ^ (1/3 : ℝ)) = -4 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_eight_squared_l843_84358


namespace NUMINAMATH_CALUDE_laura_minimum_score_l843_84366

def minimum_score (score1 score2 score3 : ℝ) (required_average : ℝ) : ℝ :=
  4 * required_average - (score1 + score2 + score3)

theorem laura_minimum_score :
  minimum_score 80 78 76 85 = 106 := by sorry

end NUMINAMATH_CALUDE_laura_minimum_score_l843_84366


namespace NUMINAMATH_CALUDE_negative_integer_square_plus_self_equals_twelve_l843_84341

theorem negative_integer_square_plus_self_equals_twelve (N : ℤ) : 
  N < 0 → N^2 + N = 12 → N = -4 := by
  sorry

end NUMINAMATH_CALUDE_negative_integer_square_plus_self_equals_twelve_l843_84341


namespace NUMINAMATH_CALUDE_initial_salary_correct_l843_84346

/-- Kirt's initial monthly salary -/
def initial_salary : ℝ := 6000

/-- Kirt's salary increase rate after one year -/
def salary_increase_rate : ℝ := 0.30

/-- Kirt's total earnings after 3 years -/
def total_earnings : ℝ := 259200

/-- Theorem stating that the initial salary satisfies the given conditions -/
theorem initial_salary_correct : 
  12 * initial_salary + 24 * (initial_salary * (1 + salary_increase_rate)) = total_earnings := by
  sorry

#eval initial_salary

end NUMINAMATH_CALUDE_initial_salary_correct_l843_84346


namespace NUMINAMATH_CALUDE_ninth_term_of_geometric_sequence_l843_84399

/-- A geometric sequence with a₃ = 16 and a₆ = 144 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ 
  (∀ n : ℕ, a (n + 1) = a n * r) ∧
  a 3 = 16 ∧ 
  a 6 = 144

theorem ninth_term_of_geometric_sequence 
  (a : ℕ → ℝ) 
  (h : geometric_sequence a) : 
  a 9 = 1296 := by
sorry

end NUMINAMATH_CALUDE_ninth_term_of_geometric_sequence_l843_84399


namespace NUMINAMATH_CALUDE_smallest_sum_abc_l843_84320

theorem smallest_sum_abc (a b c : ℕ+) : 
  (∃ x : ℝ, (Real.sin x)^2 + (Real.sin (3*x))^2 + (Real.sin (5*x))^2 + (Real.sin (7*x))^2 = 2.5 ∧
             Real.cos (a.val * x) * Real.cos (b.val * x) * Real.cos (c.val * x) = 0) →
  (∀ a' b' c' : ℕ+, 
    (∃ x : ℝ, (Real.sin x)^2 + (Real.sin (3*x))^2 + (Real.sin (5*x))^2 + (Real.sin (7*x))^2 = 2.5 ∧
               Real.cos (a'.val * x) * Real.cos (b'.val * x) * Real.cos (c'.val * x) = 0) →
    a'.val + b'.val + c'.val ≥ a.val + b.val + c.val) →
  a.val + b.val + c.val = 14 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_abc_l843_84320


namespace NUMINAMATH_CALUDE_min_value_theorem_l843_84307

theorem min_value_theorem (C D x : ℝ) (hC : C > 0) (hD : D > 0) (hx : x > 0)
  (h1 : x^2 + 1/x^2 = C) (h2 : x + 1/x = D) :
  ∃ (m : ℝ), m = 2 * Real.sqrt 3 + 3/2 ∧ ∀ y, y = C/(D-2) → y ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l843_84307


namespace NUMINAMATH_CALUDE_inequality_solution_set_l843_84397

theorem inequality_solution_set :
  {x : ℝ | 2 * x^2 + 2 * x - 3 > 7 - x} = {x : ℝ | x < -2 ∨ x > 5/2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l843_84397


namespace NUMINAMATH_CALUDE_work_completion_time_l843_84328

theorem work_completion_time (b : ℝ) (c : ℝ) (d : ℝ) (h1 : b = 14) (h2 : c = 2) (h3 : d = 5.000000000000001) : 
  ∃ a : ℝ, a = 4 ∧ 
  (c * (1 / a + 1 / b) + d * (1 / b) = 1) :=
sorry

end NUMINAMATH_CALUDE_work_completion_time_l843_84328


namespace NUMINAMATH_CALUDE_rotated_P_coordinates_l843_84351

/-- Square with side length 25 -/
def square_side_length : ℝ := 25

/-- Point Q coordinates -/
def Q : ℝ × ℝ := (0, 7)

/-- Point R is on x-axis -/
def R_on_x_axis (R : ℝ × ℝ) : Prop := R.2 = 0

/-- Line equation where S lies after rotation -/
def S_line_equation (x : ℝ) : Prop := x = 39

/-- Rotation of square about R -/
def rotated_square (P R S : ℝ × ℝ) : Prop :=
  R_on_x_axis R ∧ S_line_equation S.1 ∧ S.2 > 0

/-- Theorem: New coordinates of P after rotation -/
theorem rotated_P_coordinates (P R S : ℝ × ℝ) :
  square_side_length = 25 →
  Q = (0, 7) →
  rotated_square P R S →
  P = (19, 35) := by sorry

end NUMINAMATH_CALUDE_rotated_P_coordinates_l843_84351


namespace NUMINAMATH_CALUDE_choose_3_from_15_l843_84327

theorem choose_3_from_15 : Nat.choose 15 3 = 455 := by sorry

end NUMINAMATH_CALUDE_choose_3_from_15_l843_84327


namespace NUMINAMATH_CALUDE_line_perp_parallel_implies_planes_perp_l843_84354

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- Perpendicularity between a line and a plane -/
def Line3D.perp (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallelism between a line and a plane -/
def Line3D.parallel (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Perpendicularity between two planes -/
def Plane3D.perp (p1 p2 : Plane3D) : Prop :=
  sorry

/-- Theorem: If a line is perpendicular to one plane and parallel to another, 
    then the two planes are perpendicular to each other -/
theorem line_perp_parallel_implies_planes_perp 
  (l : Line3D) (α β : Plane3D) (h1 : l.perp α) (h2 : l.parallel β) : 
  α.perp β :=
sorry

end NUMINAMATH_CALUDE_line_perp_parallel_implies_planes_perp_l843_84354


namespace NUMINAMATH_CALUDE_obtuse_triangle_third_side_range_l843_84347

/-- A triangle with side lengths a, b, and c is obtuse if and only if
    one of its squared side lengths is greater than the sum of the squares of the other two side lengths. -/
def IsObtuse (a b c : ℝ) : Prop :=
  a^2 > b^2 + c^2 ∨ b^2 > a^2 + c^2 ∨ c^2 > a^2 + b^2

/-- The range of the third side length x in an obtuse triangle with side lengths 2, 3, and x. -/
theorem obtuse_triangle_third_side_range :
  ∀ x : ℝ, x > 0 →
    (IsObtuse 2 3 x ↔ (1 < x ∧ x < Real.sqrt 5) ∨ (Real.sqrt 13 < x ∧ x < 5)) :=
by sorry

end NUMINAMATH_CALUDE_obtuse_triangle_third_side_range_l843_84347


namespace NUMINAMATH_CALUDE_well_problem_solution_l843_84378

/-- The depth of the well and the rope lengths of five families -/
def well_problem (e : ℚ) : Prop :=
  ∃ (x a b c d : ℚ),
    -- Depth equations
    x = 2*a + b ∧
    x = 3*b + c ∧
    x = 4*c + d ∧
    x = 5*d + e ∧
    x = 6*e + a ∧
    -- Solutions
    x = (721/76)*e ∧
    a = (265/76)*e ∧
    b = (191/76)*e ∧
    c = (37/19)*e ∧
    d = (129/76)*e

/-- The well depth and rope lengths satisfy the given conditions -/
theorem well_problem_solution :
  ∀ e : ℚ, well_problem e :=
by sorry

end NUMINAMATH_CALUDE_well_problem_solution_l843_84378


namespace NUMINAMATH_CALUDE_radius_of_circle_B_l843_84331

/-- Given two circles A and B, prove that the radius of B is 10 cm -/
theorem radius_of_circle_B (diameter_A radius_A radius_B : ℝ) : 
  diameter_A = 80 → radius_A = diameter_A / 2 → radius_A = 4 * radius_B → radius_B = 10 := by
  sorry

end NUMINAMATH_CALUDE_radius_of_circle_B_l843_84331


namespace NUMINAMATH_CALUDE_choose_14_3_l843_84344

theorem choose_14_3 : Nat.choose 14 3 = 364 := by
  sorry

end NUMINAMATH_CALUDE_choose_14_3_l843_84344


namespace NUMINAMATH_CALUDE_arithmetic_sequence_second_term_l843_84323

/-- An arithmetic sequence is a sequence where the difference between consecutive terms is constant. -/
structure ArithmeticSequence where
  /-- The first term of the sequence -/
  first : ℤ
  /-- The common difference between consecutive terms -/
  diff : ℤ

/-- The nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  seq.first + (n - 1) * seq.diff

theorem arithmetic_sequence_second_term
  (seq : ArithmeticSequence)
  (h16 : seq.nthTerm 16 = 8)
  (h17 : seq.nthTerm 17 = 10) :
  seq.nthTerm 2 = -20 := by
  sorry

#check arithmetic_sequence_second_term

end NUMINAMATH_CALUDE_arithmetic_sequence_second_term_l843_84323


namespace NUMINAMATH_CALUDE_product_after_digit_reversal_mistake_l843_84385

/-- Reverses the digits of a two-digit number -/
def reverseDigits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- Checks if a number is prime -/
def isPrime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, 1 < m → m < p → ¬(p % m = 0)

theorem product_after_digit_reversal_mistake (a b : ℕ) :
  (10 ≤ a ∧ a < 100) →  -- a is a two-digit number
  isPrime b →           -- b is prime
  reverseDigits a * b = 280 →  -- product after mistake is 280
  a * b = 28 :=         -- correct product is 28
by sorry

end NUMINAMATH_CALUDE_product_after_digit_reversal_mistake_l843_84385


namespace NUMINAMATH_CALUDE_base_8_calculation_l843_84381

/-- Addition in base 8 -/
def add_base_8 (a b : ℕ) : ℕ := sorry

/-- Subtraction in base 8 -/
def sub_base_8 (a b : ℕ) : ℕ := sorry

/-- Convert a natural number to its base 8 representation -/
def to_base_8 (n : ℕ) : ℕ := sorry

/-- Convert a base 8 number to its decimal representation -/
def from_base_8 (n : ℕ) : ℕ := sorry

theorem base_8_calculation : 
  sub_base_8 (add_base_8 (from_base_8 452) (from_base_8 167)) (from_base_8 53) = from_base_8 570 := by
  sorry

end NUMINAMATH_CALUDE_base_8_calculation_l843_84381


namespace NUMINAMATH_CALUDE_hawking_implications_l843_84302

-- Define the philosophical implications
def unity_of_world_materiality : Prop := true
def thought_existence_identical : Prop := true

-- Define Hawking's statement
def hawking_statement : Prop := true

-- Theorem to prove
theorem hawking_implications :
  hawking_statement → unity_of_world_materiality ∧ thought_existence_identical :=
by
  sorry

end NUMINAMATH_CALUDE_hawking_implications_l843_84302


namespace NUMINAMATH_CALUDE_min_toothpicks_removal_l843_84322

/-- Represents a geometric figure made of toothpicks forming triangles -/
structure ToothpickFigure where
  total_toothpicks : ℕ
  upward_triangles : ℕ
  downward_triangles : ℕ

/-- The minimum number of toothpicks to remove to eliminate all triangles -/
def min_toothpicks_to_remove (figure : ToothpickFigure) : ℕ := sorry

/-- Theorem stating the minimum number of toothpicks to remove -/
theorem min_toothpicks_removal (figure : ToothpickFigure) 
  (h1 : figure.total_toothpicks = 40)
  (h2 : figure.upward_triangles = 15)
  (h3 : figure.downward_triangles = 10) :
  min_toothpicks_to_remove figure = 15 := by sorry

end NUMINAMATH_CALUDE_min_toothpicks_removal_l843_84322


namespace NUMINAMATH_CALUDE_triangle_area_l843_84335

/-- The area of a triangle with sides 10, 24, and 26 is 120 square units -/
theorem triangle_area : ∀ (a b c : ℝ),
  a = 10 ∧ b = 24 ∧ c = 26 →
  (∃ (s : ℝ), s = (a + b + c) / 2 ∧ 
   Real.sqrt (s * (s - a) * (s - b) * (s - c)) = 120) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l843_84335
