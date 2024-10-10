import Mathlib

namespace magic_square_solution_l1877_187743

/-- Represents a 3x3 magic square -/
structure MagicSquare :=
  (a11 a12 a13 a21 a22 a23 a31 a32 a33 : ℤ)

/-- The sum of any row, column, or diagonal in a magic square is the same -/
def is_magic (s : MagicSquare) : Prop :=
  let sum := s.a11 + s.a12 + s.a13
  sum = s.a21 + s.a22 + s.a23 ∧
  sum = s.a31 + s.a32 + s.a33 ∧
  sum = s.a11 + s.a21 + s.a31 ∧
  sum = s.a12 + s.a22 + s.a32 ∧
  sum = s.a13 + s.a23 + s.a33 ∧
  sum = s.a11 + s.a22 + s.a33 ∧
  sum = s.a13 + s.a22 + s.a31

theorem magic_square_solution :
  ∀ (s : MagicSquare),
    is_magic s →
    s.a11 = s.a11 ∧ s.a12 = 25 ∧ s.a13 = 75 ∧ s.a21 = 5 →
    s.a11 = 310 := by
  sorry

end magic_square_solution_l1877_187743


namespace nellie_gift_wrap_sales_l1877_187798

theorem nellie_gift_wrap_sales (total_goal : ℕ) (sold_to_uncle : ℕ) (sold_to_neighbor : ℕ) (remaining_to_sell : ℕ) :
  total_goal = 45 →
  sold_to_uncle = 10 →
  sold_to_neighbor = 6 →
  remaining_to_sell = 28 →
  total_goal - remaining_to_sell - (sold_to_uncle + sold_to_neighbor) = 1 :=
by sorry

end nellie_gift_wrap_sales_l1877_187798


namespace arithmetic_geometric_sequence_ratio_l1877_187718

theorem arithmetic_geometric_sequence_ratio (a : ℕ → ℝ) (d : ℝ) :
  d ≠ 0 ∧
  (∀ n, a (n + 1) = a n + d) ∧
  (a 3)^2 = a 1 * a 9 →
  (a 1 + a 3 + a 6) / (a 2 + a 4 + a 10) = 5/8 := by
  sorry

end arithmetic_geometric_sequence_ratio_l1877_187718


namespace quadratic_inequality_condition_l1877_187763

theorem quadratic_inequality_condition (k : ℝ) : 
  (∀ x : ℝ, x^2 - (k - 3)*x - k + 6 > 0) ↔ -3 < k ∧ k < 5 := by
  sorry

end quadratic_inequality_condition_l1877_187763


namespace chips_price_increase_l1877_187721

/-- The cost of a pack of pretzels in dollars -/
def pretzel_cost : ℝ := 4

/-- The number of packets of chips bought -/
def chips_bought : ℕ := 2

/-- The number of packets of pretzels bought -/
def pretzels_bought : ℕ := 2

/-- The total cost of the purchase in dollars -/
def total_cost : ℝ := 22

/-- The percentage increase in the price of chips compared to pretzels -/
def price_increase_percentage : ℝ := 75

theorem chips_price_increase :
  let chips_cost := pretzel_cost * (1 + price_increase_percentage / 100)
  chips_bought * chips_cost + pretzels_bought * pretzel_cost = total_cost :=
by sorry

end chips_price_increase_l1877_187721


namespace angle_sum_in_special_polygon_l1877_187708

theorem angle_sum_in_special_polygon (x y : ℝ) : 
  34 + 80 + 90 + (360 - x) + (360 - y) = 540 → x + y = 144 := by
  sorry

end angle_sum_in_special_polygon_l1877_187708


namespace unique_positive_solution_l1877_187740

theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ x ≤ 1 ∧ Real.cos (Real.arctan (Real.sin (Real.arccos x))) = x := by
  sorry

end unique_positive_solution_l1877_187740


namespace factorial_ratio_l1877_187723

theorem factorial_ratio : Nat.factorial 13 / Nat.factorial 12 = 13 := by
  sorry

end factorial_ratio_l1877_187723


namespace intersection_of_A_and_B_l1877_187716

def A : Set ℤ := {-1, 0, 1, 2, 3, 4, 5}
def B : Set ℤ := {2, 4, 6, 8}

theorem intersection_of_A_and_B : A ∩ B = {2, 4} := by
  sorry

end intersection_of_A_and_B_l1877_187716


namespace partial_pressure_of_compound_l1877_187786

/-- Represents the partial pressure of a compound in a gas mixture. -/
def partial_pressure (mole_fraction : ℝ) (total_pressure : ℝ) : ℝ :=
  mole_fraction * total_pressure

/-- Theorem stating that the partial pressure of a compound in a gas mixture
    is 0.375 atm, given specific conditions. -/
theorem partial_pressure_of_compound (mole_fraction : ℝ) (total_pressure : ℝ) 
  (h1 : mole_fraction = 0.15)
  (h2 : total_pressure = 2.5) :
  partial_pressure mole_fraction total_pressure = 0.375 := by
  sorry

end partial_pressure_of_compound_l1877_187786


namespace regression_line_not_necessarily_through_sample_point_l1877_187714

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a linear regression model -/
structure LinearRegression where
  a : ℝ  -- intercept
  b : ℝ  -- slope

/-- Calculates the y-value for a given x using the linear regression model -/
def predict (model : LinearRegression) (x : ℝ) : ℝ :=
  model.a + model.b * x

/-- Checks if a point lies on the regression line -/
def pointOnLine (model : LinearRegression) (p : Point) : Prop :=
  p.y = predict model p.x

/-- Theorem: The linear regression line does not necessarily pass through any sample point -/
theorem regression_line_not_necessarily_through_sample_point :
  ∃ (model : LinearRegression) (samples : List Point),
    samples.length > 0 ∧ ∀ p ∈ samples, ¬(pointOnLine model p) :=
by sorry

end regression_line_not_necessarily_through_sample_point_l1877_187714


namespace final_state_of_B_l1877_187762

/-- Represents a memory unit with a number of data pieces -/
structure MemoryUnit where
  data : ℕ

/-- Represents the state of all three memory units -/
structure MemoryState where
  A : MemoryUnit
  B : MemoryUnit
  C : MemoryUnit

/-- Performs the first operation: storing N data pieces in each unit -/
def firstOperation (N : ℕ) (state : MemoryState) : MemoryState :=
  { A := ⟨N⟩, B := ⟨N⟩, C := ⟨N⟩ }

/-- Performs the second operation: moving 2 data pieces from A to B -/
def secondOperation (state : MemoryState) : MemoryState :=
  { state with
    A := ⟨state.A.data - 2⟩
    B := ⟨state.B.data + 2⟩ }

/-- Performs the third operation: moving 2 data pieces from C to B -/
def thirdOperation (state : MemoryState) : MemoryState :=
  { state with
    B := ⟨state.B.data + 2⟩
    C := ⟨state.C.data - 2⟩ }

/-- Performs the fourth operation: moving N-2 data pieces from B to A -/
def fourthOperation (N : ℕ) (state : MemoryState) : MemoryState :=
  { state with
    A := ⟨state.A.data + (N - 2)⟩
    B := ⟨state.B.data - (N - 2)⟩ }

/-- The main theorem stating that after all operations, B has 6 data pieces -/
theorem final_state_of_B (N : ℕ) (h : N ≥ 3) :
  let initialState : MemoryState := ⟨⟨0⟩, ⟨0⟩, ⟨0⟩⟩
  let finalState := fourthOperation N (thirdOperation (secondOperation (firstOperation N initialState)))
  finalState.B.data = 6 := by sorry

end final_state_of_B_l1877_187762


namespace acute_angled_triangle_with_acute_pedals_l1877_187710

/-- Represents an angle in degrees, minutes, and seconds -/
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)
  (seconds : ℕ)

/-- Converts an Angle to seconds -/
def Angle.toSeconds (a : Angle) : ℕ :=
  a.degrees * 3600 + a.minutes * 60 + a.seconds

/-- Checks if an angle is acute (less than 90 degrees) -/
def Angle.isAcute (a : Angle) : Prop :=
  a.toSeconds < 90 * 3600

/-- Calculates the i-th pedal angle given an original angle -/
def pedalAngle (a : Angle) (i : ℕ) : Angle :=
  sorry -- Implementation not required for the statement

/-- Theorem statement for the acute-angled triangle problem -/
theorem acute_angled_triangle_with_acute_pedals :
  ∃ (α β γ : Angle),
    α.toSeconds < β.toSeconds ∧
    β.toSeconds < γ.toSeconds ∧
    Angle.isAcute α ∧
    Angle.isAcute β ∧
    Angle.isAcute γ ∧
    α.toSeconds + β.toSeconds + γ.toSeconds = 180 * 3600 ∧
    (∀ i : ℕ, i > 0 → i ≤ 15 →
      Angle.isAcute (pedalAngle α i) ∧
      Angle.isAcute (pedalAngle β i) ∧
      Angle.isAcute (pedalAngle γ i)) :=
by sorry

end acute_angled_triangle_with_acute_pedals_l1877_187710


namespace directional_vector_of_line_l1877_187756

/-- Given a line with equation 3x + 2y - 1 = 0, prove that (2, -3) is a directional vector --/
theorem directional_vector_of_line (x y : ℝ) :
  (3 * x + 2 * y - 1 = 0) → (2 * 3 + (-3) * 2 = 0) := by
  sorry

end directional_vector_of_line_l1877_187756


namespace ice_cream_sundaes_l1877_187769

/-- The number of unique two-scoop sundaes that can be made from n types of ice cream -/
def two_scoop_sundaes (n : ℕ) : ℕ := Nat.choose n 2

/-- Theorem: Given 6 types of ice cream, the number of unique two-scoop sundaes is 15 -/
theorem ice_cream_sundaes :
  two_scoop_sundaes 6 = 15 := by
  sorry

end ice_cream_sundaes_l1877_187769


namespace angle_measure_in_triangle_l1877_187787

theorem angle_measure_in_triangle (D E F : ℝ) : 
  D = 75 →
  E = 4 * F - 15 →
  D + E + F = 180 →
  F = 24 := by
sorry

end angle_measure_in_triangle_l1877_187787


namespace azure_valley_skirts_l1877_187704

/-- The number of skirts in Purple Valley -/
def purple_skirts : ℕ := 10

/-- The ratio of skirts in Purple Valley to Seafoam Valley -/
def purple_to_seafoam_ratio : ℚ := 1 / 4

/-- The ratio of skirts in Seafoam Valley to Azure Valley -/
def seafoam_to_azure_ratio : ℚ := 2 / 3

/-- The number of skirts in Azure Valley -/
def azure_skirts : ℕ := 60

theorem azure_valley_skirts :
  azure_skirts = (purple_skirts : ℚ) / (purple_to_seafoam_ratio * seafoam_to_azure_ratio) := by
  sorry

end azure_valley_skirts_l1877_187704


namespace triangle_identity_l1877_187770

/-- Operation △ between ordered pairs of real numbers -/
def triangle (a b c d : ℝ) : ℝ × ℝ := (a*c + b*d, a*d + b*c)

/-- Theorem: If (u,v) △ (x,y) = (u,v) for all real u and v, then (x,y) = (1,0) -/
theorem triangle_identity (x y : ℝ) :
  (∀ u v : ℝ, triangle u v x y = (u, v)) → (x = 1 ∧ y = 0) := by
  sorry

end triangle_identity_l1877_187770


namespace olivias_remaining_money_l1877_187717

def olivias_wallet (initial_amount : ℕ) (atm_amount : ℕ) (extra_spent : ℕ) : ℕ :=
  initial_amount + atm_amount - (atm_amount + extra_spent)

theorem olivias_remaining_money :
  olivias_wallet 53 91 39 = 14 :=
by sorry

end olivias_remaining_money_l1877_187717


namespace symmetric_with_x_minus_y_factor_implies_squared_factor_l1877_187789

-- Define a symmetric polynomial
def is_symmetric (p : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y, p x y = p y x

-- Define what it means for (x - y) to be a factor
def has_x_minus_y_factor (p : ℝ → ℝ → ℝ) : Prop :=
  ∃ q : ℝ → ℝ → ℝ, ∀ x y, p x y = (x - y) * q x y

-- Define what it means for (x - y)^2 to be a factor
def has_x_minus_y_squared_factor (p : ℝ → ℝ → ℝ) : Prop :=
  ∃ r : ℝ → ℝ → ℝ, ∀ x y, p x y = (x - y)^2 * r x y

-- The theorem to be proved
theorem symmetric_with_x_minus_y_factor_implies_squared_factor
  (p : ℝ → ℝ → ℝ)
  (h_sym : is_symmetric p)
  (h_factor : has_x_minus_y_factor p) :
  has_x_minus_y_squared_factor p :=
sorry

end symmetric_with_x_minus_y_factor_implies_squared_factor_l1877_187789


namespace polygon_ABCDE_perimeter_l1877_187741

/-- A point in a 2D coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The perimeter of a polygon given its vertices -/
def perimeter (vertices : List Point) : ℝ := sorry

/-- The distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

theorem polygon_ABCDE_perimeter :
  let A : Point := ⟨0, 8⟩
  let B : Point := ⟨4, 8⟩
  let C : Point := ⟨4, 4⟩
  let D : Point := ⟨8, 0⟩
  let E : Point := ⟨0, 0⟩
  perimeter [A, B, C, D, E] = 12 + 4 * Real.sqrt 5 := by
  sorry

end polygon_ABCDE_perimeter_l1877_187741


namespace externally_tangent_circle_radius_l1877_187781

/-- The radius of a circle externally tangent to three circles in a right triangle -/
theorem externally_tangent_circle_radius (A B C : ℝ × ℝ) (h_right_triangle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0) 
  (h_AB : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 3)
  (h_AC : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 6)
  (r_A : ℝ) (r_B : ℝ) (r_C : ℝ)
  (h_r_A : r_A = 1) (h_r_B : r_B = 2) (h_r_C : r_C = 3) :
  ∃ R : ℝ, R = (8 * Real.sqrt 11 - 19) / 7 ∧
    ∀ O : ℝ × ℝ, (Real.sqrt ((O.1 - A.1)^2 + (O.2 - A.2)^2) = R + r_A) ∧
                 (Real.sqrt ((O.1 - B.1)^2 + (O.2 - B.2)^2) = R + r_B) ∧
                 (Real.sqrt ((O.1 - C.1)^2 + (O.2 - C.2)^2) = R + r_C) :=
by sorry

end externally_tangent_circle_radius_l1877_187781


namespace orthocenter_of_triangle_l1877_187722

/-- The orthocenter of a triangle in 3D space -/
def orthocenter (A B C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  sorry

/-- Theorem: The orthocenter of triangle ABC is (4,3,2) -/
theorem orthocenter_of_triangle :
  let A : ℝ × ℝ × ℝ := (2, 3, 4)
  let B : ℝ × ℝ × ℝ := (6, 4, 2)
  let C : ℝ × ℝ × ℝ := (4, 5, 6)
  orthocenter A B C = (4, 3, 2) :=
by sorry

end orthocenter_of_triangle_l1877_187722


namespace house_number_theorem_l1877_187775

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that checks if a number is a perfect cube -/
def isPerfectCube (n : ℕ) : Prop := sorry

/-- A function that converts a pair of digits to a two-digit number -/
def twoDigitNumber (a b : ℕ) : ℕ := sorry

/-- The set of valid house numbers -/
def validHouseNumbers : Set (ℕ × ℕ × ℕ × ℕ) := sorry

/-- The set of valid prime pairs -/
def validPrimePairs : Set (ℕ × ℕ) := sorry

theorem house_number_theorem :
  ∃ (f : (ℕ × ℕ × ℕ × ℕ) → (ℕ × ℕ)), 
    Function.Bijective f ∧
    (∀ a b c d, (a, b, c, d) ∈ validHouseNumbers ↔ 
      (twoDigitNumber a b, twoDigitNumber c d) ∈ validPrimePairs) := by
  sorry

#check house_number_theorem

end house_number_theorem_l1877_187775


namespace team_E_not_played_B_l1877_187782

/-- Represents a soccer team in the tournament -/
inductive Team : Type
| A | B | C | D | E | F

/-- Represents the number of matches played by each team -/
def matches_played : Team → ℕ
| Team.A => 5
| Team.B => 4
| Team.C => 3
| Team.D => 2
| Team.E => 1
| Team.F => 0  -- We don't know F's matches, so we set it to 0

/-- Theorem stating that team E has not played against team B -/
theorem team_E_not_played_B :
  ∀ (t : Team), matches_played Team.E = 1 → matches_played Team.B = 4 →
  matches_played Team.A = 5 → t ≠ Team.B → t ≠ Team.E → 
  ∃ (opponent : Team), opponent ≠ Team.E ∧ opponent ≠ Team.B :=
by sorry

end team_E_not_played_B_l1877_187782


namespace digit_A_value_l1877_187761

theorem digit_A_value : ∃ (A : ℕ), A < 10 ∧ 2 * 1000000 * A + 299561 = (3 * (523 + A))^2 → A = 4 := by
  sorry

end digit_A_value_l1877_187761


namespace quadratic_inequality_solution_set_l1877_187757

-- Define the quadratic function
def f (a b c x : ℝ) := a * x^2 + b * x + c

-- Define the solution set of the first inequality
def S := Set.Ioo 1 2

-- Define the second inequality
def g (a b c x : ℝ) := a - c * (x^2 - x - 1) - b * x

-- Define the solution set of the second inequality
def T := {x : ℝ | x ≤ -3/2 ∨ x ≥ 1}

theorem quadratic_inequality_solution_set 
  (a b c : ℝ) (h : ∀ x, x ∈ S ↔ f a b c x > 0) :
  ∀ x, x ∈ T ↔ g a b c x ≥ 0 := by
sorry

end quadratic_inequality_solution_set_l1877_187757


namespace aarti_work_completion_l1877_187790

/-- Given that Aarti can complete a piece of work in 9 days, 
    this theorem proves that she will complete 3 times the same work in 27 days. -/
theorem aarti_work_completion :
  ∀ (work : ℕ) (days : ℕ),
    days = 9 →  -- Aarti can complete the work in 9 days
    (27 : ℚ) / days = 3 -- The ratio of 27 days to the original work duration is 3
    :=
by
  sorry

end aarti_work_completion_l1877_187790


namespace expression_value_l1877_187768

theorem expression_value (a b c : ℝ) (h1 : a * b * c > 0) (h2 : a * b < 0) :
  (|a| / a + 2 * b / |b| - b * c / |4 * b * c|) = -5/4 ∨
  (|a| / a + 2 * b / |b| - b * c / |4 * b * c|) = 5/4 :=
by sorry

end expression_value_l1877_187768


namespace pet_food_discount_l1877_187733

theorem pet_food_discount (msrp : ℝ) (regular_discount_max : ℝ) (additional_discount : ℝ)
  (h1 : msrp = 30)
  (h2 : regular_discount_max = 0.3)
  (h3 : additional_discount = 0.2) :
  msrp * (1 - regular_discount_max) * (1 - additional_discount) = 16.8 :=
by sorry

end pet_food_discount_l1877_187733


namespace max_angle_sum_l1877_187727

/-- Represents a quadrilateral ABCD with specific properties -/
structure Quadrilateral :=
  (x : ℝ)
  (d : ℝ)
  (angle_sum : x + (x + 2*d) + (x + d) = 180)
  (angle_progression : x ≤ x + d ∧ x + d ≤ x + 2*d)
  (similarity : x + d = 60)

/-- The maximum sum of the largest angles in triangles ABC and ACD is 180° -/
theorem max_angle_sum (q : Quadrilateral) :
  ∃ (max_sum : ℝ), max_sum = 180 ∧
  ∀ (sum : ℝ), sum = (q.x + 2*q.d) + (q.x + 2*q.d) → sum ≤ max_sum :=
sorry

end max_angle_sum_l1877_187727


namespace exists_a_for_min_g_zero_l1877_187764

-- Define the function f
def f (x : ℝ) : ℝ := x^(3/2)

-- Define the function g
def g (a x : ℝ) : ℝ := x + a * (f x)^(1/3)

-- State the theorem
theorem exists_a_for_min_g_zero :
  (∀ x₁ x₂, x₁ < x₂ → f x₁ < f x₂) →  -- f is increasing
  ∃ a : ℝ, (∀ x ∈ Set.Icc 1 9, g a x ≥ 0) ∧ 
           (∃ x ∈ Set.Icc 1 9, g a x = 0) ∧
           a = -1 :=
by sorry

end exists_a_for_min_g_zero_l1877_187764


namespace quadratic_equations_solutions_l1877_187737

theorem quadratic_equations_solutions :
  (∀ x : ℝ, (x + 4)^2 - 5*(x + 4) = 0 ↔ x = -4 ∨ x = 1) ∧
  (∀ x : ℝ, x^2 - 2*x - 15 = 0 ↔ x = -3 ∨ x = 5) :=
by sorry

end quadratic_equations_solutions_l1877_187737


namespace sasha_remaining_questions_l1877_187732

/-- Calculates the number of remaining questions given the completion rate, total questions, and work time. -/
def remaining_questions (completion_rate : ℕ) (total_questions : ℕ) (work_time : ℕ) : ℕ :=
  total_questions - completion_rate * work_time

/-- Proves that for Sasha's specific case, the number of remaining questions is 30. -/
theorem sasha_remaining_questions :
  remaining_questions 15 60 2 = 30 := by
  sorry

end sasha_remaining_questions_l1877_187732


namespace find_z_l1877_187729

/-- A structure representing the relationship between x, y, and z. -/
structure Relationship where
  x : ℝ
  y : ℝ
  z : ℝ
  k : ℝ
  prop : y = k * x^2 / z

/-- The theorem statement -/
theorem find_z (r : Relationship) (h1 : r.y = 8) (h2 : r.x = 2) (h3 : r.z = 4)
    (h4 : r.x = 4) (h5 : r.y = 72) : r.z = 16/9 := by
  sorry


end find_z_l1877_187729


namespace actual_quarterly_earnings_l1877_187780

/-- Calculates the actual quarterly earnings per share given the dividend paid for 400 shares -/
theorem actual_quarterly_earnings
  (expected_earnings : ℝ)
  (expected_dividend_ratio : ℝ)
  (additional_dividend_rate : ℝ)
  (additional_earnings_threshold : ℝ)
  (shares : ℕ)
  (total_dividend : ℝ)
  (h1 : expected_earnings = 0.80)
  (h2 : expected_dividend_ratio = 0.5)
  (h3 : additional_dividend_rate = 0.04)
  (h4 : additional_earnings_threshold = 0.10)
  (h5 : shares = 400)
  (h6 : total_dividend = 208) :
  ∃ (actual_earnings : ℝ), actual_earnings = 1.10 ∧
  total_dividend = shares * (expected_earnings * expected_dividend_ratio +
    (actual_earnings - expected_earnings) * (additional_dividend_rate / additional_earnings_threshold)) :=
by sorry

end actual_quarterly_earnings_l1877_187780


namespace triangle_rotation_path_length_l1877_187795

/-- The length of the path traversed by vertex C of an equilateral triangle rotating inside a square -/
theorem triangle_rotation_path_length :
  ∀ (triangle_side square_side : ℝ),
  triangle_side = 3 →
  square_side = 6 →
  ∃ (path_length : ℝ),
  path_length = 18 * Real.pi ∧
  path_length = 12 * (triangle_side * Real.pi / 2) :=
by sorry

end triangle_rotation_path_length_l1877_187795


namespace sister_packs_l1877_187707

def total_packs : ℕ := 13
def emily_packs : ℕ := 6

theorem sister_packs : total_packs - emily_packs = 7 := by
  sorry

end sister_packs_l1877_187707


namespace divisibility_by_eight_l1877_187772

theorem divisibility_by_eight (n : ℤ) (h : Even n) :
  ∃ k₁ k₂ k₃ k₄ : ℤ,
    n * (n^2 + 20) = 8 * k₁ ∧
    n * (n^2 - 20) = 8 * k₂ ∧
    n * (n^2 + 4) = 8 * k₃ ∧
    n * (n^2 - 4) = 8 * k₄ :=
by
  sorry

end divisibility_by_eight_l1877_187772


namespace smallest_visible_sum_l1877_187734

/-- Represents a die with opposite sides summing to 7 -/
structure Die where
  sides : Fin 6 → Nat
  opposite_sum : ∀ i : Fin 3, sides i + sides (i + 3) = 7

/-- Represents a 4x4x4 cube made of 64 dice -/
def LargeCube := Fin 4 → Fin 4 → Fin 4 → Die

/-- Function to calculate the sum of visible faces on the large cube -/
def visibleSum (cube : LargeCube) : Nat :=
  sorry

/-- Theorem stating the smallest possible sum of visible faces -/
theorem smallest_visible_sum (cube : LargeCube) : 
  visibleSum cube ≥ 144 := by
  sorry

end smallest_visible_sum_l1877_187734


namespace range_of_a_l1877_187791

-- Define propositions A and B
def propA (x : ℝ) : Prop := |x - 1| < 3
def propB (x a : ℝ) : Prop := (x + 2) * (x + a) < 0

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, propA x → propB x a) ∧ 
  (∃ x, propB x a ∧ ¬propA x)

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, sufficient_not_necessary a ↔ a < -4 :=
sorry

end range_of_a_l1877_187791


namespace three_numbers_sum_l1877_187713

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b → b ≤ c → 
  b = 10 → 
  (a + b + c) / 3 = a + 20 → 
  (a + b + c) / 3 = c - 25 → 
  a + b + c = 45 := by
  sorry

end three_numbers_sum_l1877_187713


namespace dance_circle_partition_l1877_187730

/-- The number of ways to partition n distinguishable objects into k indistinguishable,
    non-empty subsets, where rotations within subsets are considered identical. -/
def partition_count (n k : ℕ) : ℕ :=
  if k > n ∨ k = 0 then 0
  else
    (Finset.range (n - k + 1)).sum (λ i =>
      Nat.choose n (i + 1) * Nat.factorial i * Nat.factorial (n - i - 2))
    / 2

/-- Theorem stating that there are 50 ways to partition 5 children into 2 dance circles. -/
theorem dance_circle_partition :
  partition_count 5 2 = 50 := by
  sorry


end dance_circle_partition_l1877_187730


namespace diamond_value_in_treasure_l1877_187783

/-- Represents the treasure of precious stones -/
structure Treasure where
  diamond_masses : List ℝ
  crystal_mass : ℝ
  total_value : ℝ
  martin_value : ℝ

/-- Calculates the value of diamonds given their masses -/
def diamond_value (masses : List ℝ) : ℝ :=
  100 * (masses.map (λ m => m^2)).sum

/-- Calculates the value of crystals given their mass -/
def crystal_value (mass : ℝ) : ℝ :=
  3 * mass

/-- The main theorem about the value of diamonds in the treasure -/
theorem diamond_value_in_treasure (t : Treasure) : 
  t.total_value = 5000000 ∧ 
  t.martin_value = 2000000 ∧ 
  t.total_value = diamond_value t.diamond_masses + crystal_value t.crystal_mass ∧
  t.martin_value = diamond_value (t.diamond_masses.map (λ m => m/2)) + crystal_value (t.crystal_mass/2) →
  diamond_value t.diamond_masses = 2000000 := by
  sorry

end diamond_value_in_treasure_l1877_187783


namespace one_common_sale_day_in_july_l1877_187758

def is_bookstore_sale_day (day : Nat) : Prop :=
  day ≤ 31 ∧ day % 5 = 0

def is_shoe_store_sale_day (day : Nat) : Prop :=
  day ≤ 31 ∧ ∃ k : Nat, day = 3 + 7 * k

def both_stores_sale_day (day : Nat) : Prop :=
  is_bookstore_sale_day day ∧ is_shoe_store_sale_day day

theorem one_common_sale_day_in_july :
  ∃! day : Nat, both_stores_sale_day day :=
sorry

end one_common_sale_day_in_july_l1877_187758


namespace volleyball_lineup_count_l1877_187702

/-- The number of ways to choose a starting lineup from a volleyball team. -/
def starting_lineup_count (total_players : ℕ) (lineup_size : ℕ) (captain_count : ℕ) : ℕ :=
  total_players * (Nat.choose (total_players - 1) (lineup_size - 1))

/-- Theorem: The number of ways to choose a starting lineup of 8 players
    (including one captain) from a team of 18 players is 350,064. -/
theorem volleyball_lineup_count :
  starting_lineup_count 18 8 1 = 350064 := by
  sorry

end volleyball_lineup_count_l1877_187702


namespace cubic_roots_roots_product_l1877_187706

/-- Given a cubic equation x^3 - 7x^2 + 36 = 0 where the product of two of its roots is 18,
    prove that the roots are -2, 3, and 6. -/
theorem cubic_roots (x : ℝ) : 
  (∃ r₁ r₂ r₃ : ℝ, x^3 - 7*x^2 + 36 = 0 ∧ 
   r₁ * r₂ = 18 ∧
   (x = r₁ ∨ x = r₂ ∨ x = r₃)) →
  (x = -2 ∨ x = 3 ∨ x = 6) :=
by sorry

/-- The product of all three roots of the cubic equation x^3 - 7x^2 + 36 = 0 is -36. -/
theorem roots_product (r₁ r₂ r₃ : ℝ) :
  r₁^3 - 7*r₁^2 + 36 = 0 ∧ 
  r₂^3 - 7*r₂^2 + 36 = 0 ∧ 
  r₃^3 - 7*r₃^2 + 36 = 0 →
  r₁ * r₂ * r₃ = -36 :=
by sorry

end cubic_roots_roots_product_l1877_187706


namespace set_operations_and_range_l1877_187750

-- Define the sets A, B, and C
def A : Set ℝ := {x | 2 < x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | 5 - a < x ∧ x < a}

-- Theorem statement
theorem set_operations_and_range :
  (A ∪ B = {x : ℝ | 2 < x ∧ x < 10}) ∧
  ((Set.univ \ A) ∩ B = {x : ℝ | 7 ≤ x ∧ x < 10}) ∧
  (∀ a : ℝ, C a ⊆ B → a ≤ 3) := by
  sorry

end set_operations_and_range_l1877_187750


namespace tan_difference_inequality_l1877_187754

theorem tan_difference_inequality (x y n : ℝ) (hn : n > 0) (h : Real.tan x = n * Real.tan y) :
  Real.tan (x - y) ^ 2 ≤ (n - 1) ^ 2 / (4 * n) := by
  sorry

end tan_difference_inequality_l1877_187754


namespace shaded_area_ratio_l1877_187720

/-- The side length of square EFGH -/
def side_length : ℕ := 7

/-- The area of square EFGH -/
def total_area : ℕ := side_length ^ 2

/-- The area of the first shaded region (2x2 square) -/
def shaded_area_1 : ℕ := 2 ^ 2

/-- The area of the second shaded region (5x5 square minus 3x3 square) -/
def shaded_area_2 : ℕ := 5 ^ 2 - 3 ^ 2

/-- The area of the third shaded region (7x1 rectangle) -/
def shaded_area_3 : ℕ := 7 * 1

/-- The total shaded area -/
def total_shaded_area : ℕ := shaded_area_1 + shaded_area_2 + shaded_area_3

/-- Theorem: The ratio of shaded area to total area is 33/49 -/
theorem shaded_area_ratio :
  (total_shaded_area : ℚ) / total_area = 33 / 49 := by
  sorry

end shaded_area_ratio_l1877_187720


namespace three_digit_ending_l1877_187700

theorem three_digit_ending (N : ℕ) (h1 : N > 0) (h2 : N % 1000 = N^2 % 1000) 
  (h3 : N % 1000 ≥ 100) : N % 1000 = 127 := by
  sorry

end three_digit_ending_l1877_187700


namespace painting_price_increase_l1877_187774

theorem painting_price_increase (x : ℝ) : 
  (1 + x / 100) * (1 - 0.15) = 1.0625 → x = 25 := by
  sorry

end painting_price_increase_l1877_187774


namespace least_n_divisible_by_77_l1877_187709

theorem least_n_divisible_by_77 (n : ℕ) : 
  (n ≥ 100 ∧ 
   77 ∣ (2^(n+1) - 1) ∧ 
   ∀ m, m ≥ 100 ∧ m < n → ¬(77 ∣ (2^(m+1) - 1))) → 
  n = 119 :=
by sorry

end least_n_divisible_by_77_l1877_187709


namespace five_digit_number_puzzle_l1877_187788

theorem five_digit_number_puzzle :
  ∃! N : ℕ,
    10000 ≤ N ∧ N < 100000 ∧
    ∃ (x y : ℕ),
      0 ≤ x ∧ x < 10 ∧
      1000 ≤ y ∧ y < 10000 ∧
      N = 10 * y + x ∧
      N + y = 54321 :=
by sorry

end five_digit_number_puzzle_l1877_187788


namespace average_of_numbers_is_eleven_l1877_187773

theorem average_of_numbers_is_eleven : ∃ (M N : ℝ), 
  10 < N ∧ N < 20 ∧ 
  M = N - 4 ∧ 
  (8 + M + N) / 3 = 11 := by
  sorry

end average_of_numbers_is_eleven_l1877_187773


namespace expression_evaluation_l1877_187725

theorem expression_evaluation :
  (3^1006 + 7^1007)^2 - (3^1006 - 7^1007)^2 = 42 * 21^1006 :=
by sorry

end expression_evaluation_l1877_187725


namespace count_valid_pairs_l1877_187731

def is_odd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

def valid_pair (a b : ℕ) : Prop :=
  is_odd a ∧ is_odd b ∧ a > 1 ∧ b > 1 ∧ a * b = 315

theorem count_valid_pairs :
  ∃! (pairs : Finset (ℕ × ℕ)), 
    (∀ p ∈ pairs, valid_pair p.1 p.2) ∧ 
    pairs.card = 5 :=
sorry

end count_valid_pairs_l1877_187731


namespace tangent_property_reasoning_l1877_187703

-- Define the types of geometric objects
inductive GeometricObject
| Circle
| Line
| Sphere
| Plane

-- Define the types of reasoning
inductive ReasoningType
| Inductive
| Deductive
| Analogical
| Transitive

-- Define the property of perpendicularity for 2D and 3D cases
def isPerpendicular (obj1 obj2 : GeometricObject) : Prop :=
  match obj1, obj2 with
  | GeometricObject.Line, GeometricObject.Line => true
  | GeometricObject.Line, GeometricObject.Plane => true
  | _, _ => false

-- Define the tangent property for 2D case
def tangentProperty2D (circle : GeometricObject) (tangentLine : GeometricObject) (centerToTangentLine : GeometricObject) : Prop :=
  circle = GeometricObject.Circle ∧
  tangentLine = GeometricObject.Line ∧
  centerToTangentLine = GeometricObject.Line ∧
  isPerpendicular tangentLine centerToTangentLine

-- Define the tangent property for 3D case
def tangentProperty3D (sphere : GeometricObject) (tangentPlane : GeometricObject) (centerToTangentLine : GeometricObject) : Prop :=
  sphere = GeometricObject.Sphere ∧
  tangentPlane = GeometricObject.Plane ∧
  centerToTangentLine = GeometricObject.Line ∧
  isPerpendicular tangentPlane centerToTangentLine

-- Theorem statement
theorem tangent_property_reasoning :
  (∃ (circle tangentLine centerToTangentLine : GeometricObject),
    tangentProperty2D circle tangentLine centerToTangentLine) →
  (∃ (sphere tangentPlane centerToTangentLine : GeometricObject),
    tangentProperty3D sphere tangentPlane centerToTangentLine) →
  (∀ (r : ReasoningType), r = ReasoningType.Analogical) :=
by sorry

end tangent_property_reasoning_l1877_187703


namespace polynomial_divisibility_l1877_187776

-- Define the polynomial
def f (x m : ℝ) : ℝ := 3 * x^2 - 5 * x + m

-- Theorem statement
theorem polynomial_divisibility (m : ℝ) : 
  (∀ x : ℝ, f x m = 0 → x = 2) ↔ m = -2 := by
  sorry

end polynomial_divisibility_l1877_187776


namespace sufficient_but_not_necessary_l1877_187796

theorem sufficient_but_not_necessary (x : ℝ) :
  (∀ x, x^2 > 1 → 1/x < 1) ∧
  (∃ x, 1/x < 1 ∧ x^2 ≤ 1) :=
by sorry

end sufficient_but_not_necessary_l1877_187796


namespace fixed_point_and_equal_intercept_line_l1877_187724

/-- The fixed point through which all lines of the form ax + y - a - 2 = 0 pass -/
def fixed_point : ℝ × ℝ := (1, 2)

/-- The line equation with parameter a -/
def line_equation (a x y : ℝ) : Prop := a * x + y - a - 2 = 0

/-- A line with equal intercepts on both axes passing through a point -/
def equal_intercept_line (p : ℝ × ℝ) (x y : ℝ) : Prop := x + y = p.1 + p.2

theorem fixed_point_and_equal_intercept_line :
  (∀ a : ℝ, ∃ x y : ℝ, line_equation a x y ∧ (x, y) = fixed_point) ∧
  equal_intercept_line fixed_point = λ x y => x + y = 3 := by sorry

end fixed_point_and_equal_intercept_line_l1877_187724


namespace seating_arrangements_with_restriction_l1877_187747

def number_of_people : ℕ := 10

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem seating_arrangements_with_restriction :
  let total_arrangements := factorial (number_of_people - 1) * 7
  total_arrangements = 2540160 := by sorry

end seating_arrangements_with_restriction_l1877_187747


namespace housewife_spending_fraction_l1877_187799

theorem housewife_spending_fraction (initial_amount : ℝ) (remaining_amount : ℝ)
  (h1 : initial_amount = 150)
  (h2 : remaining_amount = 50) :
  (initial_amount - remaining_amount) / initial_amount = 2 / 3 := by
sorry

end housewife_spending_fraction_l1877_187799


namespace fruit_seller_loss_percentage_l1877_187779

/-- Calculates the percentage loss for a fruit seller given selling price, break-even price, and profit percentage. -/
def calculate_loss_percentage (selling_price profit_price profit_percentage : ℚ) : ℚ :=
  let cost_price := profit_price / (1 + profit_percentage / 100)
  let loss := cost_price - selling_price
  (loss / cost_price) * 100

/-- Theorem stating that under given conditions, the fruit seller's loss percentage is 15%. -/
theorem fruit_seller_loss_percentage :
  let selling_price : ℚ := 12
  let profit_price : ℚ := 14823529411764707 / 1000000000000000
  let profit_percentage : ℚ := 5
  calculate_loss_percentage selling_price profit_price profit_percentage = 15 := by
  sorry

#eval calculate_loss_percentage 12 (14823529411764707 / 1000000000000000) 5

end fruit_seller_loss_percentage_l1877_187779


namespace opposite_of_2023_l1877_187712

theorem opposite_of_2023 : -(2023 : ℤ) = -2023 := by
  sorry

end opposite_of_2023_l1877_187712


namespace ellipse_properties_l1877_187745

-- Define the ellipse C
def ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the line l
def line (x y k m : ℝ) : Prop := y = k * x + m

-- Define the conditions
def conditions (a b k m : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ 2 * b = 2 ∧ (a^2 - b^2) / a^2 = 1/2

-- Define the perpendicular bisector condition
def perp_bisector_condition (k m : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse x₁ y₁ (Real.sqrt 2) 1 ∧
    ellipse x₂ y₂ (Real.sqrt 2) 1 ∧
    line x₁ y₁ k m ∧
    line x₂ y₂ k m ∧
    (y₁ + y₂) / 2 + 1/2 = -1/k * ((x₁ + x₂) / 2)

-- Define the theorem
theorem ellipse_properties (a b k m : ℝ) :
  conditions a b k m →
  (∀ x y, ellipse x y a b ↔ ellipse x y (Real.sqrt 2) 1) ∧
  (perp_bisector_condition k m → 2 * k^2 + 1 = 2 * m) ∧
  (∃ (S : ℝ → ℝ), (∀ k m, perp_bisector_condition k m → S m ≤ Real.sqrt 2 / 2) ∧
                  (∃ k₀ m₀, perp_bisector_condition k₀ m₀ ∧ S m₀ = Real.sqrt 2 / 2)) :=
sorry

end ellipse_properties_l1877_187745


namespace brendas_age_l1877_187726

/-- Given the ages of Addison, Brenda, and Janet, prove that Brenda is 7/3 years old. -/
theorem brendas_age (A B J : ℚ) 
  (h1 : A = 4 * B)  -- Addison's age is four times Brenda's age
  (h2 : J = B + 7)  -- Janet is seven years older than Brenda
  (h3 : A = J)      -- Addison and Janet are twins
  : B = 7 / 3 := by
  sorry

end brendas_age_l1877_187726


namespace min_value_theorem_l1877_187794

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_mn : m + n = 1) :
  (1 / m + 2 / n) ≥ 3 + 2 * Real.sqrt 2 :=
by sorry

end min_value_theorem_l1877_187794


namespace power_multiplication_l1877_187738

theorem power_multiplication (a : ℝ) : a^2 * a^4 = a^6 := by
  sorry

end power_multiplication_l1877_187738


namespace journey_speed_journey_speed_theorem_l1877_187746

/-- 
Given a journey of 24 km completed in 8 hours, where the first 4 hours are
traveled at speed v km/hr and the last 4 hours at 2 km/hr, prove that v = 4.
-/
theorem journey_speed : ℝ → Prop :=
  fun v : ℝ =>
    (4 * v + 4 * 2 = 24) →
    v = 4

-- The proof is omitted
axiom journey_speed_proof : journey_speed 4

#check journey_speed_proof

-- Proof
theorem journey_speed_theorem : ∃ v : ℝ, journey_speed v := by
  sorry

end journey_speed_journey_speed_theorem_l1877_187746


namespace inverse_proportion_problem_l1877_187705

theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) (h1 : x * y = k) 
  (h2 : x + y = 30) (h3 : x - y = 10) : 
  x = 8 → y = 25 := by
  sorry

end inverse_proportion_problem_l1877_187705


namespace largest_prime_divisor_for_primality_test_l1877_187771

theorem largest_prime_divisor_for_primality_test :
  ∀ n : ℕ, 950 ≤ n → n ≤ 1000 →
  (∀ p : ℕ, p ≤ 31 → Nat.Prime p → ¬(p ∣ n)) →
  Nat.Prime n :=
by sorry

end largest_prime_divisor_for_primality_test_l1877_187771


namespace tetrahedron_volume_bound_l1877_187777

/-- A tetrahedron is represented by its six edge lengths -/
structure Tetrahedron where
  edge1 : ℝ
  edge2 : ℝ
  edge3 : ℝ
  edge4 : ℝ
  edge5 : ℝ
  edge6 : ℝ

/-- The volume of a tetrahedron -/
noncomputable def volume (t : Tetrahedron) : ℝ := sorry

/-- A tetrahedron with five edges not exceeding 1 -/
def FiveEdgesLimitedTetrahedron : Type :=
  { t : Tetrahedron // t.edge1 ≤ 1 ∧ t.edge2 ≤ 1 ∧ t.edge3 ≤ 1 ∧ t.edge4 ≤ 1 ∧ t.edge5 ≤ 1 }

theorem tetrahedron_volume_bound (t : FiveEdgesLimitedTetrahedron) :
  volume t.val ≤ 1/8 := by sorry

end tetrahedron_volume_bound_l1877_187777


namespace markers_per_box_l1877_187728

theorem markers_per_box (total_students : ℕ) (boxes : ℕ) 
  (group1_students : ℕ) (group1_markers : ℕ)
  (group2_students : ℕ) (group2_markers : ℕ)
  (group3_markers : ℕ) :
  total_students = 30 →
  boxes = 22 →
  group1_students = 10 →
  group1_markers = 2 →
  group2_students = 15 →
  group2_markers = 4 →
  group3_markers = 6 →
  (boxes : ℚ) * ((group1_students * group1_markers + 
                  group2_students * group2_markers + 
                  (total_students - group1_students - group2_students) * group3_markers) / boxes : ℚ) = 
  (boxes : ℚ) * (5 : ℚ) :=
by sorry

end markers_per_box_l1877_187728


namespace parking_lot_ratio_l1877_187748

/-- Given the initial number of cars in the front parking lot, the total number of cars at the end,
    and the number of cars added during the play, prove the ratio of cars in the back to front parking lot. -/
theorem parking_lot_ratio
  (front_initial : ℕ)
  (total_end : ℕ)
  (added_during : ℕ)
  (h1 : front_initial = 100)
  (h2 : total_end = 700)
  (h3 : added_during = 300) :
  (total_end - added_during - front_initial) / front_initial = 3 := by
  sorry

#check parking_lot_ratio

end parking_lot_ratio_l1877_187748


namespace sum_of_roots_greater_than_four_l1877_187719

/-- Given a function f(x) = x - 1 + a*exp(x), prove that the sum of its roots is greater than 4 -/
theorem sum_of_roots_greater_than_four (a : ℝ) :
  let f := λ x : ℝ => x - 1 + a * Real.exp x
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ + x₂ > 4 := by
  sorry

end sum_of_roots_greater_than_four_l1877_187719


namespace not_sufficient_not_necessary_l1877_187711

theorem not_sufficient_not_necessary (a b : ℝ) : 
  (∃ x y : ℝ, x > y ∧ x^2 ≤ y^2) ∧ (∃ u v : ℝ, u^2 > v^2 ∧ u ≤ v) := by sorry

end not_sufficient_not_necessary_l1877_187711


namespace skyscraper_anniversary_l1877_187752

theorem skyscraper_anniversary (years_since_built : ℕ) (years_to_anniversary : ℕ) (years_before_anniversary : ℕ) : 
  years_since_built = 100 →
  years_to_anniversary = 200 →
  years_before_anniversary = 5 →
  years_to_anniversary - years_before_anniversary - years_since_built = 95 :=
by sorry

end skyscraper_anniversary_l1877_187752


namespace floor_sum_example_l1877_187755

theorem floor_sum_example : ⌊(23.7 : ℝ)⌋ + ⌊(-23.7 : ℝ)⌋ = -1 := by
  sorry

end floor_sum_example_l1877_187755


namespace exists_m_even_function_l1877_187742

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x

-- State the theorem
theorem exists_m_even_function :
  ∃ m : ℝ, ∀ x : ℝ, f m x = f m (-x) :=
sorry

end exists_m_even_function_l1877_187742


namespace benny_spent_amount_l1877_187778

/-- Represents the total amount spent in US dollars -/
def total_spent (initial_amount : ℝ) (remaining_amount : ℝ) : ℝ :=
  initial_amount - remaining_amount

/-- Theorem stating that given the initial amount of 200 US dollars and
    the remaining amount of 45 US dollars, the total amount spent is 155 US dollars -/
theorem benny_spent_amount :
  total_spent 200 45 = 155 := by sorry

end benny_spent_amount_l1877_187778


namespace log_inequality_l1877_187759

theorem log_inequality (a b c : ℝ) 
  (ha : a = Real.log 3 / Real.log (1/2))
  (hb : b = Real.log 5 / Real.log (1/2))
  (hc : c = Real.log (1/2) / Real.log 3) :
  b < a ∧ a < c := by
sorry

end log_inequality_l1877_187759


namespace roots_transformation_l1877_187785

theorem roots_transformation (s₁ s₂ s₃ : ℂ) : 
  (s₁^3 - 4*s₁^2 + 5*s₁ - 1 = 0) ∧ 
  (s₂^3 - 4*s₂^2 + 5*s₂ - 1 = 0) ∧ 
  (s₃^3 - 4*s₃^2 + 5*s₃ - 1 = 0) →
  ((3*s₁)^3 - 12*(3*s₁)^2 + 135*(3*s₁) - 27 = 0) ∧
  ((3*s₂)^3 - 12*(3*s₂)^2 + 135*(3*s₂) - 27 = 0) ∧
  ((3*s₃)^3 - 12*(3*s₃)^2 + 135*(3*s₃) - 27 = 0) :=
by sorry

end roots_transformation_l1877_187785


namespace red_yellow_peach_difference_l1877_187793

theorem red_yellow_peach_difference (red_peaches yellow_peaches : ℕ) 
  (h1 : red_peaches = 19) 
  (h2 : yellow_peaches = 11) : 
  red_peaches - yellow_peaches = 8 := by
sorry

end red_yellow_peach_difference_l1877_187793


namespace function_inequality_implies_m_bound_l1877_187767

/-- Given functions f and g, prove that if for any x₁ in [0, 2], 
    there exists x₂ in [1, 2] such that f(x₁) ≥ g(x₂), then m ≥ 1/4 -/
theorem function_inequality_implies_m_bound 
  (f : ℝ → ℝ) (g : ℝ → ℝ) (m : ℝ)
  (hf : ∀ x, f x = x^2)
  (hg : ∀ x, g x = (1/2)^x - m)
  (h : ∀ x₁ ∈ Set.Icc 0 2, ∃ x₂ ∈ Set.Icc 1 2, f x₁ ≥ g x₂) :
  m ≥ 1/4 :=
by sorry

end function_inequality_implies_m_bound_l1877_187767


namespace quadratic_root_in_unit_interval_l1877_187797

/-- Given a quadratic polynomial P(x) = x^2 + px + q where P(q) < 0, 
    exactly one root of P(x) lies in the interval (0, 1) -/
theorem quadratic_root_in_unit_interval (p q : ℝ) :
  let P : ℝ → ℝ := λ x => x^2 + p*x + q
  (P q < 0) →
  ∃! x : ℝ, P x = 0 ∧ 0 < x ∧ x < 1 :=
by sorry

end quadratic_root_in_unit_interval_l1877_187797


namespace meal_combinations_count_l1877_187749

/-- The number of items on the menu -/
def menu_items : ℕ := 15

/-- The index of the restricted item -/
def restricted_item : ℕ := 10

/-- A function that calculates the number of valid meal combinations -/
def valid_combinations (n : ℕ) (r : ℕ) : ℕ :=
  n * n - 1

/-- Theorem stating that the number of valid meal combinations is 224 -/
theorem meal_combinations_count :
  valid_combinations menu_items restricted_item = 224 := by
  sorry

#eval valid_combinations menu_items restricted_item

end meal_combinations_count_l1877_187749


namespace sum_of_digits_of_square_of_ones_l1877_187701

/-- Given a natural number n, construct a number consisting of n ones -/
def numberWithOnes (n : ℕ) : ℕ :=
  (10^n - 1) / 9

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  sorry

theorem sum_of_digits_of_square_of_ones (n : ℕ) :
  sumOfDigits ((numberWithOnes n)^2) = n^2 :=
sorry

end sum_of_digits_of_square_of_ones_l1877_187701


namespace function_inequality_and_range_l1877_187744

/-- Given functions f and g, prove that if |f(x)| ≤ |g(x)| for all x, then f = g/2 - 4.
    Also prove that if f(x) ≥ (m + 2)x - m - 15 for all x > 2, then m ≤ 2. -/
theorem function_inequality_and_range (a b m : ℝ) : 
  let f := fun (x : ℝ) => x^2 + a*x + b
  let g := fun (x : ℝ) => 2*x^2 - 4*x - 16
  (∀ x, |f x| ≤ |g x|) →
  (a = -2 ∧ b = -8) ∧
  ((∀ x > 2, f x ≥ (m + 2)*x - m - 15) → m ≤ 2) :=
by sorry

end function_inequality_and_range_l1877_187744


namespace undefined_values_count_l1877_187766

theorem undefined_values_count : ∃ (S : Finset ℝ), 
  (∀ x ∈ S, (x^2 + 2*x - 3) * (x - 3) * (x + 1) = 0) ∧ 
  (∀ x ∉ S, (x^2 + 2*x - 3) * (x - 3) * (x + 1) ≠ 0) ∧ 
  Finset.card S = 4 := by
sorry

end undefined_values_count_l1877_187766


namespace chicken_purchase_equation_l1877_187765

/-- Represents a group purchase scenario -/
structure GroupPurchase where
  numPeople : ℕ
  itemPrice : ℕ

/-- Calculates the surplus or shortage in a group purchase -/
def calculateDifference (g : GroupPurchase) (contribution : ℕ) : ℤ :=
  (g.numPeople * contribution : ℤ) - g.itemPrice

/-- Theorem stating the correct equation for the chicken purchase problem -/
theorem chicken_purchase_equation (g : GroupPurchase) :
  calculateDifference g 9 = 11 ∧ calculateDifference g 6 = -16 →
  9 * g.numPeople - 11 = 6 * g.numPeople + 16 := by
  sorry


end chicken_purchase_equation_l1877_187765


namespace sin_sqrt3_over_2_solution_set_l1877_187739

theorem sin_sqrt3_over_2_solution_set (θ : ℝ) : 
  Real.sin θ = (Real.sqrt 3) / 2 ↔ 
  ∃ k : ℤ, θ = π / 3 + 2 * k * π ∨ θ = 2 * π / 3 + 2 * k * π :=
sorry

end sin_sqrt3_over_2_solution_set_l1877_187739


namespace sum_of_roots_eq_twelve_l1877_187792

theorem sum_of_roots_eq_twelve : ∃ (x₁ x₂ : ℝ), 
  (x₁ - 6)^2 = 16 ∧ 
  (x₂ - 6)^2 = 16 ∧ 
  x₁ + x₂ = 12 := by
sorry

end sum_of_roots_eq_twelve_l1877_187792


namespace ten_mile_taxi_cost_l1877_187784

/-- The cost of a taxi ride given the base fare, cost per mile, and distance traveled. -/
def taxi_cost (base_fare : ℝ) (cost_per_mile : ℝ) (distance : ℝ) : ℝ :=
  base_fare + cost_per_mile * distance

/-- Theorem: The cost of a 10-mile taxi ride with a $2.00 base fare and $0.30 per mile is $5.00. -/
theorem ten_mile_taxi_cost :
  taxi_cost 2 0.3 10 = 5 := by
  sorry

end ten_mile_taxi_cost_l1877_187784


namespace zero_of_f_necessary_not_sufficient_for_decreasing_g_l1877_187753

noncomputable def f (m : ℝ) (x : ℝ) := 2^x + m - 1
noncomputable def g (m : ℝ) (x : ℝ) := Real.log x / Real.log m

theorem zero_of_f_necessary_not_sufficient_for_decreasing_g :
  (∀ m : ℝ, (∃ x : ℝ, f m x = 0) → 
    (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → g m x₁ > g m x₂)) ∧
  (∃ m : ℝ, (∃ x : ℝ, f m x = 0) ∧ 
    ¬(∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → g m x₁ > g m x₂)) :=
by sorry

end zero_of_f_necessary_not_sufficient_for_decreasing_g_l1877_187753


namespace neither_necessary_nor_sufficient_l1877_187715

open Real

/-- A function f is increasing on (0,∞) -/
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x ∧ x < y → f x < f y

theorem neither_necessary_nor_sufficient :
  ∃ (f₁ f₂ : ℝ → ℝ),
    (∀ x, x > 0 → f₁ x ≠ 0 ∧ f₂ x ≠ 0) ∧
    IsIncreasing f₁ ∧
    ¬IsIncreasing (fun x ↦ x * f₁ x) ∧
    ¬IsIncreasing f₂ ∧
    IsIncreasing (fun x ↦ x * f₂ x) :=
by sorry

end neither_necessary_nor_sufficient_l1877_187715


namespace head_start_is_90_meters_l1877_187736

/-- The head start distance in a race between Cristina and Nicky -/
def head_start_distance (cristina_speed nicky_speed : ℝ) (catch_up_time : ℝ) : ℝ :=
  nicky_speed * catch_up_time

/-- Theorem: Given Cristina's speed of 5 m/s, Nicky's speed of 3 m/s, 
    and a catch-up time of 30 seconds, the head start distance is 90 meters -/
theorem head_start_is_90_meters :
  head_start_distance 5 3 30 = 90 := by sorry

end head_start_is_90_meters_l1877_187736


namespace platform_length_l1877_187760

/-- The length of a platform given train parameters -/
theorem platform_length (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ) :
  train_length = 120 →
  train_speed_kmph = 60 →
  crossing_time = 15 →
  ∃ (platform_length : ℝ), abs (platform_length - 130.05) < 0.01 :=
by
  sorry


end platform_length_l1877_187760


namespace solve_euro_equation_l1877_187751

-- Define the € operation
def euro (x y : ℝ) := 3 * x * y

-- State the theorem
theorem solve_euro_equation (y : ℝ) (h1 : euro y (euro x 5) = 540) (h2 : y = 3) : x = 4 := by
  sorry

end solve_euro_equation_l1877_187751


namespace bottom_right_value_mod_2011_l1877_187735

/-- Represents a cell on the board -/
structure Cell where
  row : Nat
  col : Nat

/-- Represents the board configuration -/
structure Board where
  size : Nat
  markedCells : List Cell

/-- The value of a cell on the board -/
def cellValue (board : Board) (cell : Cell) : ℕ :=
  sorry

/-- Theorem stating that the bottom-right corner value is congruent to 2 modulo 2011 -/
theorem bottom_right_value_mod_2011 (board : Board) 
  (h1 : board.size = 2012)
  (h2 : ∀ c ∈ board.markedCells, c.row + c.col = 2011 ∧ c.row ≠ 1 ∧ c.col ≠ 1)
  (h3 : ∀ c, c.row = 1 ∨ c.col = 1 → cellValue board c = 1)
  (h4 : ∀ c ∈ board.markedCells, cellValue board c = 0)
  (h5 : ∀ c, c.row > 1 ∧ c.col > 1 ∧ c ∉ board.markedCells → 
    cellValue board c = cellValue board {row := c.row - 1, col := c.col} + 
                        cellValue board {row := c.row, col := c.col - 1}) :
  cellValue board {row := 2012, col := 2012} ≡ 2 [MOD 2011] :=
sorry

end bottom_right_value_mod_2011_l1877_187735
