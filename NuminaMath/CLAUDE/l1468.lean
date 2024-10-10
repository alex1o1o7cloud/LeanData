import Mathlib

namespace max_oranges_removal_l1468_146821

/-- A triangular grid of length n -/
structure TriangularGrid (n : ℕ) where
  (n_pos : 0 < n)
  (n_not_div_3 : ¬ 3 ∣ n)

/-- The total number of oranges in a triangular grid -/
def totalOranges (n : ℕ) : ℕ := (n + 1) * (n + 2) / 2

/-- A good triple of oranges -/
structure GoodTriple (n : ℕ) where
  (isValid : Bool)

/-- The maximum number of oranges that can be removed -/
def maxRemovableOranges (n : ℕ) : ℕ := totalOranges n - 3

theorem max_oranges_removal (n : ℕ) (grid : TriangularGrid n) :
  maxRemovableOranges n = totalOranges n - 3 := by sorry

end max_oranges_removal_l1468_146821


namespace ryan_bus_trips_l1468_146867

/-- Represents Ryan's commuting schedule and times --/
structure CommuteSchedule where
  bike_time : ℕ
  bus_time : ℕ
  friend_time : ℕ
  bike_frequency : ℕ
  friend_frequency : ℕ
  total_time : ℕ

/-- Calculates the number of bus trips given a CommuteSchedule --/
def calculate_bus_trips (schedule : CommuteSchedule) : ℕ :=
  (schedule.total_time - 
   (schedule.bike_time * schedule.bike_frequency + 
    schedule.friend_time * schedule.friend_frequency)) / 
  schedule.bus_time

/-- Ryan's actual commute schedule --/
def ryan_schedule : CommuteSchedule :=
  { bike_time := 30
  , bus_time := 40
  , friend_time := 10
  , bike_frequency := 1
  , friend_frequency := 1
  , total_time := 160 }

/-- Theorem stating that Ryan takes the bus 3 times a week --/
theorem ryan_bus_trips : calculate_bus_trips ryan_schedule = 3 := by
  sorry

end ryan_bus_trips_l1468_146867


namespace complex_magnitude_from_equation_l1468_146898

theorem complex_magnitude_from_equation (z : ℂ) : 
  Complex.I * (1 - z) = 1 → Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_magnitude_from_equation_l1468_146898


namespace power_inequality_l1468_146874

theorem power_inequality (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 2) : a^b < b^a := by
  sorry

end power_inequality_l1468_146874


namespace unique_real_root_of_equation_l1468_146844

theorem unique_real_root_of_equation :
  ∃! x : ℝ, 2 * Real.sqrt (x - 3) + 6 = x :=
by sorry

end unique_real_root_of_equation_l1468_146844


namespace donation_is_45_l1468_146828

/-- The total donation to the class funds given the number of stuffed animals and selling prices -/
def total_donation (barbara_stuffed_animals : ℕ) (barbara_price : ℚ) (trish_price : ℚ) : ℚ :=
  let trish_stuffed_animals := 2 * barbara_stuffed_animals
  barbara_stuffed_animals * barbara_price + trish_stuffed_animals * trish_price

/-- Proof that the total donation is $45 given the specific conditions -/
theorem donation_is_45 :
  total_donation 9 2 (3/2) = 45 := by
  sorry

#eval total_donation 9 2 (3/2)

end donation_is_45_l1468_146828


namespace exists_circumcircle_equation_l1468_146809

/-- Triangle with side lengths 6, 8, and 10 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 6
  hb : b = 8
  hc : c = 10
  right_angle : a^2 + b^2 = c^2

/-- Circumcircle of a triangle -/
structure Circumcircle (t : RightTriangle) where
  center : ℝ × ℝ
  radius : ℝ
  is_valid : radius^2 = (t.c / 2)^2

theorem exists_circumcircle_equation (t : RightTriangle) :
  ∃ (cc : Circumcircle t), ∃ (x y : ℝ), (x - cc.center.1)^2 + (y - cc.center.2)^2 = cc.radius^2 ∧
  cc.center = (0, 0) ∧ cc.radius = 5 := by
  sorry

end exists_circumcircle_equation_l1468_146809


namespace dereks_age_l1468_146806

/-- Given that Charlie's age is four times Derek's age, Emily is five years older than Derek,
    and Charlie and Emily are twins, prove that Derek is 5/3 years old. -/
theorem dereks_age (charlie emily derek : ℝ)
    (h1 : charlie = 4 * derek)
    (h2 : emily = derek + 5)
    (h3 : charlie = emily) :
    derek = 5 / 3 := by
  sorry

end dereks_age_l1468_146806


namespace percentage_of_percentage_l1468_146875

theorem percentage_of_percentage (x : ℝ) (h : x ≠ 0) :
  (60 / 100) * (30 / 100) * x = (18 / 100) * x := by
  sorry

end percentage_of_percentage_l1468_146875


namespace pure_imaginary_iff_a_eq_one_l1468_146894

theorem pure_imaginary_iff_a_eq_one (a : ℝ) :
  (∃ b : ℝ, Complex.mk (a^2 - 1) (a + 1) = Complex.I * b) ↔ a = 1 := by
  sorry

end pure_imaginary_iff_a_eq_one_l1468_146894


namespace odd_shift_three_l1468_146824

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

theorem odd_shift_three (f : ℝ → ℝ) 
  (h1 : is_odd (λ x => f (x + 1))) 
  (h2 : is_odd (λ x => f (x - 1))) : 
  is_odd (λ x => f (x + 3)) := by
sorry

end odd_shift_three_l1468_146824


namespace determine_c_l1468_146882

/-- Given integers a and b, where there exist unique x, y, z satisfying the LCM conditions,
    the value of c can be uniquely determined. -/
theorem determine_c (a b : ℕ) 
  (h_exists : ∃! (x y z : ℕ), a = Nat.lcm y z ∧ b = Nat.lcm x z ∧ ∃ c, c = Nat.lcm x y) :
  ∃! c, ∀ (x y z : ℕ), 
    (a = Nat.lcm y z ∧ b = Nat.lcm x z ∧ c = Nat.lcm x y) → 
    (∀ (x' y' z' : ℕ), a = Nat.lcm y' z' ∧ b = Nat.lcm x' z' → (x = x' ∧ y = y' ∧ z = z')) :=
by sorry


end determine_c_l1468_146882


namespace unique_P_value_l1468_146890

theorem unique_P_value (x y P : ℤ) : 
  x > 0 → y > 0 → x + y = P → 3 * x + 5 * y = 13 → P = 3 := by
  sorry

end unique_P_value_l1468_146890


namespace binomial_10_choose_3_l1468_146803

theorem binomial_10_choose_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_choose_3_l1468_146803


namespace geometric_sequence_a6_l1468_146832

/-- Given a geometric sequence {a_n} where a₄ = 7 and a₈ = 63, prove that a₆ = 21 -/
theorem geometric_sequence_a6 (a : ℕ → ℝ) :
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) →  -- geometric sequence condition
  a 4 = 7 →                                  -- given a₄ = 7
  a 8 = 63 →                                 -- given a₈ = 63
  a 6 = 21 :=                                -- prove a₆ = 21
by sorry

end geometric_sequence_a6_l1468_146832


namespace equation_equality_l1468_146851

theorem equation_equality (a b c : ℕ) 
  (ha : 0 < a ∧ a < 10) 
  (hb : 0 < b ∧ b < 10) 
  (hc : 0 < c ∧ c < 10) 
  (hac : a + c = 10) : 
  (10 * b + a) * (10 * b + c) = 100 * b * (b + 1) + a * c := by
  sorry

end equation_equality_l1468_146851


namespace surface_area_of_problem_structure_l1468_146859

/-- Represents a solid formed by unit cubes -/
structure CubeStructure where
  base_layer : Nat
  middle_layer : Nat
  top_layer : Nat
  base_width : Nat
  base_length : Nat

/-- Calculates the surface area of the cube structure -/
def surface_area (c : CubeStructure) : Nat :=
  sorry

/-- The specific cube structure described in the problem -/
def problem_structure : CubeStructure :=
  { base_layer := 6
  , middle_layer := 4
  , top_layer := 2
  , base_width := 2
  , base_length := 3 }

theorem surface_area_of_problem_structure :
  surface_area problem_structure = 36 :=
sorry

end surface_area_of_problem_structure_l1468_146859


namespace line_circle_intersection_l1468_146876

-- Define the line l and circle C
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x - y - k + 1 = 0
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*y = 0

-- Theorem statement
theorem line_circle_intersection :
  ∀ (k : ℝ) (A B : ℝ × ℝ),
  (∃ (x y : ℝ), line_l k x y ∧ circle_C x y) →
  (∀ (x y : ℝ), line_l k x y → x = 1 ∧ y = 1) ∧
  (∃ (chord_length : ℝ), chord_length = Real.sqrt 8 ∧ 
    ∀ (x y : ℝ), line_l k x y ∧ circle_C x y → 
      (x - A.1)^2 + (y - A.2)^2 ≥ chord_length^2) ∧
  (∃ (max_chord : ℝ), max_chord = 4 ∧
    ∀ (x y : ℝ), line_l k x y ∧ circle_C x y → 
      (x - A.1)^2 + (y - A.2)^2 ≤ max_chord^2) :=
by sorry

end line_circle_intersection_l1468_146876


namespace continuity_at_3_l1468_146808

def f (x : ℝ) : ℝ := -2 * x^2 - 4

theorem continuity_at_3 :
  ∀ ε > 0, ∃ δ > 0, δ = ε / 12 ∧
  ∀ x : ℝ, |x - 3| < δ → |f x - f 3| < ε := by
  sorry

end continuity_at_3_l1468_146808


namespace no_integer_pairs_with_square_diff_150_l1468_146896

theorem no_integer_pairs_with_square_diff_150 :
  ¬ ∃ (m n : ℕ), m ≥ n ∧ m^2 - n^2 = 150 := by
  sorry

end no_integer_pairs_with_square_diff_150_l1468_146896


namespace solution_set_when_a_is_one_range_of_a_for_existence_condition_l1468_146856

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 2| + |2*x + a|

-- Part I
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≥ 5} = {x : ℝ | x ≤ -4/3 ∨ x ≥ 2} :=
sorry

-- Part II
theorem range_of_a_for_existence_condition :
  (∃ x₀ : ℝ, f a x₀ + |x₀ - 2| < 3) ↔ (-7 < a ∧ a < -1) :=
sorry

end solution_set_when_a_is_one_range_of_a_for_existence_condition_l1468_146856


namespace crushing_load_calculation_l1468_146837

theorem crushing_load_calculation (T H K : ℝ) (hT : T = 3) (hH : H = 9) (hK : K = 2) :
  (50 * T^5) / (K * H^3) = 25/3 := by
  sorry

end crushing_load_calculation_l1468_146837


namespace polynomial_root_relation_l1468_146880

-- Define the polynomials h(x) and k(x)
def h (p : ℝ) (x : ℝ) : ℝ := x^3 + p*x^2 + x + 15
def k (q s : ℝ) (x : ℝ) : ℝ := x^4 + x^3 + q*x^2 + 150*x + s

-- State the theorem
theorem polynomial_root_relation (p q s : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    h p x = 0 ∧ h p y = 0 ∧ h p z = 0) →
  (∀ x : ℝ, h p x = 0 → k q s x = 0) →
  k q s 1 = -16048 :=
sorry

end polynomial_root_relation_l1468_146880


namespace chess_competition_result_l1468_146833

/-- Represents the number of 8th-grade students in the chess competition. -/
def n : ℕ := sorry

/-- Represents the score of each 8th-grade student. -/
def σ : ℚ := sorry

/-- The total points scored by the two 7th-grade students. -/
def seventh_grade_total : ℕ := 8

/-- The theorem stating the conditions and the result of the chess competition. -/
theorem chess_competition_result :
  (∃ (n : ℕ) (σ : ℚ),
    n > 0 ∧
    σ = (2 * n - 7 : ℚ) / n ∧
    σ = 2 - 7 / n ∧
    (σ = 1 ∨ σ = (3 : ℚ) / 2) ∧
    n = 7) :=
sorry

end chess_competition_result_l1468_146833


namespace correct_calculation_l1468_146863

theorem correct_calculation (a b : ℝ) : 6 * a^2 * b - b * a^2 = 5 * a^2 * b := by
  sorry

end correct_calculation_l1468_146863


namespace shirt_discount_percentage_l1468_146840

/-- Calculates the discount percentage on a shirt given the original prices,
    total paid, and discount on the jacket. -/
theorem shirt_discount_percentage
  (jacket_price : ℝ)
  (shirt_price : ℝ)
  (total_paid : ℝ)
  (jacket_discount : ℝ)
  (h1 : jacket_price = 100)
  (h2 : shirt_price = 60)
  (h3 : total_paid = 110)
  (h4 : jacket_discount = 0.3)
  : (1 - (total_paid - jacket_price * (1 - jacket_discount)) / shirt_price) * 100 = 100 / 3 := by
  sorry

#eval (1 - (110 - 100 * (1 - 0.3)) / 60) * 100

end shirt_discount_percentage_l1468_146840


namespace vector_relation_in_right_triangular_prism_l1468_146836

/-- A right triangular prism with vertices A, B, C, A₁, B₁, C₁ -/
structure RightTriangularPrism (V : Type*) [AddCommGroup V] :=
  (A B C A₁ B₁ C₁ : V)

/-- The theorem stating the relation between vectors in a right triangular prism -/
theorem vector_relation_in_right_triangular_prism
  {V : Type*} [AddCommGroup V] (prism : RightTriangularPrism V)
  (a b c : V)
  (h1 : prism.C - prism.A = a)
  (h2 : prism.C - prism.B = b)
  (h3 : prism.C - prism.C₁ = c) :
  prism.A₁ - prism.B = -a - c + b := by
  sorry

end vector_relation_in_right_triangular_prism_l1468_146836


namespace unknown_interest_rate_l1468_146822

/-- Proves that given the conditions of the problem, the unknown interest rate is 6% -/
theorem unknown_interest_rate (total : ℚ) (part1 : ℚ) (part2 : ℚ) (rate1 : ℚ) (rate2 : ℚ) (yearly_income : ℚ) :
  total = 2600 →
  part1 = 1600 →
  part2 = total - part1 →
  rate1 = 5 / 100 →
  yearly_income = 140 →
  yearly_income = part1 * rate1 + part2 * rate2 →
  rate2 = 6 / 100 := by
sorry

#eval (6 : ℚ) / 100

end unknown_interest_rate_l1468_146822


namespace reflection_coordinates_sum_l1468_146818

/-- Given a point C at (3, y+4) and its reflection D over the y-axis, with y = 2,
    the sum of all four coordinates of C and D is equal to 12. -/
theorem reflection_coordinates_sum :
  let y : ℝ := 2
  let C : ℝ × ℝ := (3, y + 4)
  let D : ℝ × ℝ := (-C.1, C.2)  -- Reflection over y-axis
  C.1 + C.2 + D.1 + D.2 = 12 := by
sorry

end reflection_coordinates_sum_l1468_146818


namespace fraction_value_l1468_146897

theorem fraction_value (a b : ℝ) (h1 : |a| = 3) (h2 : |b| = 1/2) (h3 : a > b) :
  a / b = 6 ∨ a / b = -6 := by
  sorry

end fraction_value_l1468_146897


namespace wolf_prize_laureates_l1468_146858

theorem wolf_prize_laureates (total_scientists : ℕ) 
                              (both_wolf_and_nobel : ℕ) 
                              (total_nobel : ℕ) 
                              (h1 : total_scientists = 50)
                              (h2 : both_wolf_and_nobel = 16)
                              (h3 : total_nobel = 27)
                              (h4 : total_nobel - both_wolf_and_nobel = 
                                    (total_scientists - wolf_laureates - (total_nobel - both_wolf_and_nobel)) + 3) :
  wolf_laureates = 31 :=
by
  sorry

end wolf_prize_laureates_l1468_146858


namespace no_four_consecutive_lucky_numbers_l1468_146881

/-- A function that checks if a number is lucky -/
def is_lucky (n : ℕ) : Prop :=
  1000000 ≤ n ∧ n ≤ 9999999 ∧ 
  ∃ (a b c d e f g : ℕ),
    n = a * 1000000 + b * 100000 + c * 10000 + d * 1000 + e * 100 + f * 10 + g ∧
    a ≠ 0 ∧ (n % (a * b * c * d * e * f * g) = 0)

/-- Theorem stating that four consecutive lucky numbers do not exist -/
theorem no_four_consecutive_lucky_numbers :
  ¬ ∃ (n : ℕ), is_lucky n ∧ is_lucky (n + 1) ∧ is_lucky (n + 2) ∧ is_lucky (n + 3) :=
sorry

end no_four_consecutive_lucky_numbers_l1468_146881


namespace intersection_A_B_l1468_146884

def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B : Set ℝ := {2, 3, 4, 5}

theorem intersection_A_B : A ∩ B = {2, 3} := by
  sorry

end intersection_A_B_l1468_146884


namespace prob_reroll_two_dice_l1468_146848

-- Define a die as a natural number between 1 and 6
def Die := {n : ℕ | 1 ≤ n ∧ n ≤ 6}

-- Define a roll of four dice
def FourDiceRoll := Die × Die × Die × Die

-- Function to calculate the sum of four dice
def diceSum (roll : FourDiceRoll) : ℕ := roll.1 + roll.2.1 + roll.2.2.1 + roll.2.2.2

-- Function to determine if a roll is a win (sum is 9)
def isWin (roll : FourDiceRoll) : Prop := diceSum roll = 9

-- Function to calculate the probability of winning by rerolling all four dice
def probWinRerollAll : ℚ := 56 / 1296

-- Function to calculate the probability of winning by rerolling two dice
def probWinRerollTwo (keptSum : ℕ) : ℚ :=
  if keptSum ≤ 7 then (9 - keptSum - 1) / 36
  else (13 - (9 - keptSum)) / 36

-- Theorem: The probability of Jason choosing to reroll exactly two dice is 1/18
theorem prob_reroll_two_dice : 
  (∀ roll : FourDiceRoll, ∃ (keptSum : ℕ), 
    (keptSum ≤ 8 ∧ probWinRerollTwo keptSum > probWinRerollAll) ∨
    (keptSum > 8 ∧ probWinRerollAll ≥ probWinRerollTwo keptSum)) →
  (2 : ℚ) / 36 = 1 / 18 := by
  sorry

end prob_reroll_two_dice_l1468_146848


namespace rectangle_tiling_l1468_146838

/-- A rectangle can be perfectly tiled by unit-width strips if and only if
    at least one of its dimensions is an integer. -/
theorem rectangle_tiling (a b : ℝ) :
  (∃ (n : ℕ), a * b = n) →
  (∃ (k : ℕ), a = k ∨ b = k) := by sorry

end rectangle_tiling_l1468_146838


namespace quadratic_inequality_implication_l1468_146815

theorem quadratic_inequality_implication (x : ℝ) :
  x^2 - 5*x + 6 < 0 → 20 < x^2 + 5*x + 6 ∧ x^2 + 5*x + 6 < 30 :=
by
  sorry

end quadratic_inequality_implication_l1468_146815


namespace sqrt_450_simplification_l1468_146819

theorem sqrt_450_simplification : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end sqrt_450_simplification_l1468_146819


namespace P_has_negative_and_positive_roots_l1468_146843

def P (x : ℝ) : ℝ := x^7 - 2*x^6 - 7*x^4 - x^2 + 9

theorem P_has_negative_and_positive_roots :
  (∃ (a : ℝ), a < 0 ∧ P a = 0) ∧ (∃ (b : ℝ), b > 0 ∧ P b = 0) := by sorry

end P_has_negative_and_positive_roots_l1468_146843


namespace expand_and_simplify_l1468_146816

theorem expand_and_simplify (x y : ℝ) : (-x + y) * (-x - y) = x^2 - y^2 := by
  sorry

end expand_and_simplify_l1468_146816


namespace inverse_not_in_M_exponential_in_M_logarithmic_in_M_l1468_146855

-- Define set M
def M (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, f (x₀ + 1) = f x₀ + f 1

-- Problem 1
theorem inverse_not_in_M :
  ¬ M (fun x => 1 / x) := by sorry

-- Problem 2
theorem exponential_in_M (k b : ℝ) :
  M (fun x => k * 2^x + b) ↔ (k = 0 ∧ b = 0) ∨ (k ≠ 0 ∧ (2 * k + b) / k > 0) := by sorry

-- Problem 3
theorem logarithmic_in_M :
  ∀ a : ℝ, M (fun x => Real.log (a / (x^2 + 2))) ↔ 
    (a ≥ 3/2 ∧ a ≤ 6 ∧ a ≠ 3) := by sorry

end inverse_not_in_M_exponential_in_M_logarithmic_in_M_l1468_146855


namespace lottery_winnings_l1468_146825

theorem lottery_winnings 
  (num_tickets : ℕ) 
  (winning_numbers_per_ticket : ℕ) 
  (total_winnings : ℕ) 
  (h1 : num_tickets = 3)
  (h2 : winning_numbers_per_ticket = 5)
  (h3 : total_winnings = 300) :
  total_winnings / (num_tickets * winning_numbers_per_ticket) = 20 :=
by sorry

end lottery_winnings_l1468_146825


namespace inequality_solution_set_l1468_146865

theorem inequality_solution_set :
  ∀ x : ℝ, (7 / 30 + |x - 7 / 60| < 11 / 20) ↔ (-1 / 5 < x ∧ x < 13 / 30) := by
  sorry

end inequality_solution_set_l1468_146865


namespace point_transformation_theorem_l1468_146872

/-- Rotation of a point (x, y) by 180° around (h, k) -/
def rotate180 (x y h k : ℝ) : ℝ × ℝ :=
  (2 * h - x, 2 * k - y)

/-- Reflection of a point (x, y) about the line y = x -/
def reflectAboutYEqualX (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem point_transformation_theorem (a b : ℝ) :
  let P := (a, b)
  let rotated := rotate180 a b 2 3
  let final := reflectAboutYEqualX rotated.1 rotated.2
  final = (5, -4) →
  a - b = 7 := by
sorry

end point_transformation_theorem_l1468_146872


namespace solve_equation_l1468_146879

/-- Given the equation 19(x + y) + 17 = 19(-x + y) - z, where x = 1, prove that z = -55 -/
theorem solve_equation (y : ℝ) : 
  ∃ (z : ℝ), 19 * (1 + y) + 17 = 19 * (-1 + y) - z ∧ z = -55 := by
  sorry

end solve_equation_l1468_146879


namespace rhombus_area_l1468_146870

/-- The area of a rhombus with side length 4 cm and an acute angle of 45° is 8 cm². -/
theorem rhombus_area (side_length : ℝ) (acute_angle : ℝ) : 
  side_length = 4 → acute_angle = π / 4 → 
  (side_length * side_length * Real.sin acute_angle) = 8 := by
  sorry

end rhombus_area_l1468_146870


namespace total_lunch_is_fifteen_l1468_146846

/-- The total amount spent on lunch given the conditions -/
def total_lunch_amount (your_amount : ℕ) (friend_amount : ℕ) : ℕ :=
  your_amount + friend_amount

/-- Theorem: The total amount spent on lunch is $15 -/
theorem total_lunch_is_fifteen :
  ∃ (your_amount : ℕ),
    (your_amount + 1 = 8) →
    (total_lunch_amount your_amount 8 = 15) :=
by
  sorry

end total_lunch_is_fifteen_l1468_146846


namespace proposition_truth_values_l1468_146839

theorem proposition_truth_values (p q : Prop) (hp : p) (hq : ¬q) :
  (p ∨ q) ∧ ¬(¬p) ∧ ¬(p ∧ q) ∧ ¬(¬p ∨ q) := by
  sorry

end proposition_truth_values_l1468_146839


namespace perpendicular_line_through_point_l1468_146813

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a line in the form ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Function to check if a point is on a line
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if two lines are perpendicular
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

-- Theorem statement
theorem perpendicular_line_through_point :
  let givenLine : Line := { a := 2, b := 1, c := -5 }
  let point : Point := { x := 3, y := 0 }
  let perpendicularLine : Line := { a := 1, b := -2, c := 3 }
  perpendicular givenLine perpendicularLine ∧ 
  pointOnLine point perpendicularLine := by sorry

end perpendicular_line_through_point_l1468_146813


namespace plane_equation_from_perpendicular_foot_l1468_146891

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the coefficients of a plane equation Ax + By + Cz + D = 0 -/
structure PlaneCoefficients where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- Check if a point lies on a plane -/
def pointOnPlane (p : Point3D) (c : PlaneCoefficients) : Prop :=
  c.A * p.x + c.B * p.y + c.C * p.z + c.D = 0

/-- Check if a vector is perpendicular to a plane -/
def vectorPerpendicularToPlane (v : Point3D) (c : PlaneCoefficients) : Prop :=
  ∃ (k : ℝ), v.x = k * c.A ∧ v.y = k * c.B ∧ v.z = k * c.C

/-- The main theorem -/
theorem plane_equation_from_perpendicular_foot : 
  ∃ (c : PlaneCoefficients),
    c.A = 4 ∧ c.B = -3 ∧ c.C = 1 ∧ c.D = -52 ∧
    pointOnPlane ⟨8, -6, 2⟩ c ∧
    vectorPerpendicularToPlane ⟨8, -6, 2⟩ c ∧
    c.A > 0 ∧
    Nat.gcd (Nat.gcd (Nat.gcd (Int.natAbs c.A) (Int.natAbs c.B)) (Int.natAbs c.C)) (Int.natAbs c.D) = 1 :=
by sorry

end plane_equation_from_perpendicular_foot_l1468_146891


namespace lunks_needed_for_apples_l1468_146807

/-- Exchange rate between lunks and kunks -/
def lunks_to_kunks (l : ℚ) : ℚ := (4 / 7) * l

/-- Exchange rate between kunks and apples -/
def kunks_to_apples (k : ℚ) : ℚ := (5 / 3) * k

/-- Number of apples to be purchased -/
def apples_to_buy : ℕ := 24

/-- Theorem stating that at least 27 lunks are needed to buy 24 apples -/
theorem lunks_needed_for_apples :
  ∀ l : ℚ, kunks_to_apples (lunks_to_kunks l) ≥ apples_to_buy → l ≥ 27 := by
  sorry

end lunks_needed_for_apples_l1468_146807


namespace missing_ratio_l1468_146831

theorem missing_ratio (x y : ℚ) (h : x / y * (6 / 11) * (11 / 2) = 2) : x / y = 2 / 3 := by
  sorry

end missing_ratio_l1468_146831


namespace least_possible_value_of_x_l1468_146854

theorem least_possible_value_of_x : 
  ∃ (x y z : ℤ), 
    (∃ k : ℤ, x = 2 * k) ∧ 
    (∃ m n : ℤ, y = 2 * m + 1 ∧ z = 2 * n + 1) ∧ 
    y - x > 5 ∧ 
    (∀ w : ℤ, w - x ≥ 9 → w ≥ z) ∧ 
    (∀ v : ℤ, (∃ j : ℤ, v = 2 * j) → v ≥ x) → 
    x = 0 :=
by sorry

end least_possible_value_of_x_l1468_146854


namespace valid_pairs_l1468_146835

def is_valid_pair (a b : ℕ) : Prop :=
  (a^2 + b) % (b^2 - a) = 0 ∧ (b^2 + a) % (a^2 - b) = 0

theorem valid_pairs :
  ∀ a b : ℕ, is_valid_pair a b ↔ 
    ((a = 1 ∧ b = 2) ∨ 
     (a = 2 ∧ b = 1) ∨ 
     (a = 2 ∧ b = 2) ∨ 
     (a = 2 ∧ b = 3) ∨ 
     (a = 3 ∧ b = 2) ∨ 
     (a = 3 ∧ b = 3)) :=
by sorry

end valid_pairs_l1468_146835


namespace regular_ngon_inscribed_circle_l1468_146860

theorem regular_ngon_inscribed_circle (n : ℕ) (R : ℝ) (h : R > 0) :
  (n : ℝ) / 2 * R^2 * Real.sin (2 * Real.pi / n) = 3 * R^2 → n = 12 := by
  sorry

end regular_ngon_inscribed_circle_l1468_146860


namespace rectangle_areas_sum_l1468_146834

theorem rectangle_areas_sum : 
  let rectangles : List (ℕ × ℕ) := [(2, 1), (2, 9), (2, 25), (2, 49), (2, 81), (2, 121)]
  let areas := rectangles.map (fun (w, l) => w * l)
  areas.sum = 572 := by
  sorry

end rectangle_areas_sum_l1468_146834


namespace true_discount_example_l1468_146814

/-- Given a banker's discount and face value, calculate the true discount -/
def true_discount (bankers_discount : ℚ) (face_value : ℚ) : ℚ :=
  (face_value * bankers_discount) / (face_value + bankers_discount)

/-- Theorem stating that for given values, the true discount is 480 -/
theorem true_discount_example : true_discount 576 2880 = 480 := by
  sorry

end true_discount_example_l1468_146814


namespace largest_non_sum_of_100_composites_l1468_146853

/-- A number is composite if it's the product of two integers greater than 1 -/
def IsComposite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

/-- A number can be expressed as the sum of 100 composite numbers -/
def IsSumOf100Composites (n : ℕ) : Prop :=
  ∃ (f : Fin 100 → ℕ), (∀ i, IsComposite (f i)) ∧ n = (Finset.univ.sum f)

/-- 403 is the largest integer that cannot be expressed as the sum of 100 composites -/
theorem largest_non_sum_of_100_composites :
  (¬ IsSumOf100Composites 403) ∧ (∀ n > 403, IsSumOf100Composites n) :=
sorry

end largest_non_sum_of_100_composites_l1468_146853


namespace girls_to_boys_ratio_l1468_146864

theorem girls_to_boys_ratio (total : ℕ) (girl_boy_diff : ℕ) : 
  total = 25 → girl_boy_diff = 3 → 
  ∃ (girls boys : ℕ), 
    girls + boys = total ∧ 
    girls = boys + girl_boy_diff ∧ 
    (girls : ℚ) / (boys : ℚ) = 14 / 11 := by
  sorry

#check girls_to_boys_ratio

end girls_to_boys_ratio_l1468_146864


namespace regular_icosahedron_edges_l1468_146857

/-- A regular icosahedron is a convex polyhedron with 20 faces, each of which is an equilateral triangle. -/
structure RegularIcosahedron :=
  (faces : Nat)
  (face_shape : String)
  (is_convex : Bool)
  (h_faces : faces = 20)
  (h_face_shape : face_shape = "equilateral triangle")
  (h_convex : is_convex = true)

/-- The number of edges in a regular icosahedron -/
def num_edges (i : RegularIcosahedron) : Nat := 30

/-- Theorem: A regular icosahedron has 30 edges -/
theorem regular_icosahedron_edges (i : RegularIcosahedron) : num_edges i = 30 := by
  sorry

end regular_icosahedron_edges_l1468_146857


namespace floor_sqrt_eight_count_l1468_146886

theorem floor_sqrt_eight_count : 
  (Finset.range 81 \ Finset.range 64).card = 17 := by sorry

end floor_sqrt_eight_count_l1468_146886


namespace symmetry_classification_l1468_146887

-- Define the shape type
inductive Shape
  | Parallelogram
  | Rectangle
  | RightTrapezoid
  | Square
  | EquilateralTriangle
  | LineSegment

-- Define properties
def isAxiallySymmetric (s : Shape) : Prop :=
  match s with
  | Shape.Rectangle => True
  | Shape.Square => True
  | Shape.EquilateralTriangle => True
  | Shape.LineSegment => True
  | _ => False

def isCentrallySymmetric (s : Shape) : Prop :=
  match s with
  | Shape.Parallelogram => True
  | Shape.Rectangle => True
  | Shape.Square => True
  | Shape.LineSegment => True
  | _ => False

-- Theorem statement
theorem symmetry_classification (s : Shape) :
  (isAxiallySymmetric s ∧ isCentrallySymmetric s) ↔
  (s = Shape.Rectangle ∨ s = Shape.Square ∨ s = Shape.LineSegment) :=
by sorry

end symmetry_classification_l1468_146887


namespace first_term_value_l1468_146802

/-- A geometric sequence with five terms -/
structure GeometricSequence :=
  (a b c d e : ℝ)
  (is_geometric : ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r ∧ d = c * r ∧ e = d * r)

/-- Theorem: In a geometric sequence where the fourth term is 81 and the fifth term is 243, the first term is 3 -/
theorem first_term_value (seq : GeometricSequence) 
  (h1 : seq.d = 81)
  (h2 : seq.e = 243) : 
  seq.a = 3 := by
sorry

end first_term_value_l1468_146802


namespace cycles_alignment_min_cycles_alignment_l1468_146899

/-- The length of the letter cycle -/
def letter_cycle_length : ℕ := 6

/-- The length of the digit cycle -/
def digit_cycle_length : ℕ := 4

/-- The theorem stating when both cycles will simultaneously return to their original state -/
theorem cycles_alignment (m : ℕ) (h1 : m > 0) (h2 : m % letter_cycle_length = 0) (h3 : m % digit_cycle_length = 0) :
  m ≥ 12 :=
sorry

/-- The theorem stating that 12 is the least number satisfying the conditions -/
theorem min_cycles_alignment :
  12 % letter_cycle_length = 0 ∧ 12 % digit_cycle_length = 0 ∧
  ∀ (k : ℕ), k > 0 → k % letter_cycle_length = 0 → k % digit_cycle_length = 0 → k ≥ 12 :=
sorry

end cycles_alignment_min_cycles_alignment_l1468_146899


namespace union_complement_equals_B_l1468_146827

def U : Set ℕ := {x | x < 4}
def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {2, 3}

theorem union_complement_equals_B : B ∪ (U \ A) = B := by sorry

end union_complement_equals_B_l1468_146827


namespace smallest_d_for_divisibility_l1468_146826

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def digit_sum (d : ℕ) : ℕ := 5 + 4 + 7 + d + 0 + 6

def number (d : ℕ) : ℕ := 547000 + d * 1000 + 6

theorem smallest_d_for_divisibility :
  ∃ (d : ℕ), d = 2 ∧ 
  is_divisible_by_3 (number d) ∧ 
  ∀ (k : ℕ), k < d → ¬is_divisible_by_3 (number k) := by
  sorry

#check smallest_d_for_divisibility

end smallest_d_for_divisibility_l1468_146826


namespace spaceship_travel_distance_l1468_146849

def earth_to_x : ℝ := 0.5
def x_to_y : ℝ := 0.1
def y_to_earth : ℝ := 0.1

theorem spaceship_travel_distance :
  earth_to_x + x_to_y + y_to_earth = 0.7 := by
  sorry

end spaceship_travel_distance_l1468_146849


namespace prime_sum_product_l1468_146817

theorem prime_sum_product : ∃ p q : ℕ, 
  Nat.Prime p ∧ Nat.Prime q ∧ p + q = 97 ∧ p * q = 190 := by
  sorry

end prime_sum_product_l1468_146817


namespace race_permutations_l1468_146812

theorem race_permutations (n : ℕ) (h : n = 4) : Nat.factorial n = 24 := by
  sorry

end race_permutations_l1468_146812


namespace assignment_n_plus_one_increases_by_one_l1468_146830

/-- Represents a variable in a programming language -/
structure Variable where
  name : String
  value : Int

/-- Represents an expression in a programming language -/
inductive Expression where
  | Const : Int → Expression
  | Var : Variable → Expression
  | Add : Expression → Expression → Expression

/-- Represents an assignment statement in a programming language -/
structure AssignmentStatement where
  lhs : Variable
  rhs : Expression

/-- Evaluates an expression given the current state of variables -/
def evalExpression (expr : Expression) (state : List Variable) : Int :=
  match expr with
  | Expression.Const n => n
  | Expression.Var v => v.value
  | Expression.Add e1 e2 => evalExpression e1 state + evalExpression e2 state

/-- Executes an assignment statement and returns the updated state -/
def executeAssignment (stmt : AssignmentStatement) (state : List Variable) : List Variable :=
  let newValue := evalExpression stmt.rhs state
  state.map fun v => if v.name = stmt.lhs.name then { v with value := newValue } else v

/-- Theorem: N=N+1 increases the value of N by 1 -/
theorem assignment_n_plus_one_increases_by_one (n : Variable) (state : List Variable) :
  let stmt : AssignmentStatement := { lhs := n, rhs := Expression.Add (Expression.Var n) (Expression.Const 1) }
  let newState := executeAssignment stmt state
  let oldValue := (state.find? fun v => v.name = n.name).map (fun v => v.value)
  let newValue := (newState.find? fun v => v.name = n.name).map (fun v => v.value)
  (oldValue.isSome ∧ newValue.isSome) →
  newValue = oldValue.map (fun v => v + 1) :=
by
  sorry

end assignment_n_plus_one_increases_by_one_l1468_146830


namespace parallel_vectors_m_value_l1468_146889

/-- Two vectors in R² are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value (m : ℝ) :
  let a : ℝ × ℝ := (m + 1, -2)
  let b : ℝ × ℝ := (-3, 3)
  parallel a b → m = -3 := by
  sorry

end parallel_vectors_m_value_l1468_146889


namespace eighth_term_value_l1468_146805

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- First term of the sequence
  a : ℚ
  -- Common difference of the sequence
  d : ℚ
  -- Sum of first six terms is 21
  sum_first_six : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) + (a + 5*d) = 21
  -- Seventh term is 8
  seventh_term : a + 6*d = 8

/-- The eighth term of the arithmetic sequence is 65/7 -/
theorem eighth_term_value (seq : ArithmeticSequence) : seq.a + 7*seq.d = 65/7 := by
  sorry

end eighth_term_value_l1468_146805


namespace book_pages_calculation_l1468_146847

theorem book_pages_calculation (num_books : ℕ) (pages_per_side : ℕ) (sides_per_sheet : ℕ) (num_sheets : ℕ) : 
  num_books = 2 → 
  pages_per_side = 4 → 
  sides_per_sheet = 2 → 
  num_sheets = 150 → 
  (num_sheets * pages_per_side * sides_per_sheet) / num_books = 600 :=
by
  sorry

end book_pages_calculation_l1468_146847


namespace problem_solution_l1468_146861

def p (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0

def q (x m : ℝ) : Prop := x^2 + 2*x + 1 - m^4 ≤ 0

theorem problem_solution (m : ℝ) :
  (∀ x, q x m → p x) → (m ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3)) ∧
  (∀ x, ¬(q x m) → ¬(p x)) → (m ≥ 3 ∨ m ≤ -3) ∧
  ((∀ x, ¬(q x m) → ¬(p x)) ∧ ¬(∀ x, ¬(p x) → ¬(q x m))) → (m ≥ 3 ∨ m ≤ -3) :=
by sorry

#check problem_solution

end problem_solution_l1468_146861


namespace function_period_l1468_146877

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
def condition1 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (2 + x) = f (2 - x)
def condition2 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (5 + x) = f (5 - x)

-- Define the period
def is_period (T : ℝ) (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (x + T) = f x

-- Theorem statement
theorem function_period (f : ℝ → ℝ) 
  (h1 : condition1 f) (h2 : condition2 f) : 
  (∃ T : ℝ, T > 0 ∧ is_period T f ∧ ∀ S : ℝ, S > 0 ∧ is_period S f → T ≤ S) ∧
  (∀ T : ℝ, T > 0 ∧ is_period T f ∧ (∀ S : ℝ, S > 0 ∧ is_period S f → T ≤ S) → T = 6) :=
sorry

end function_period_l1468_146877


namespace video_subscription_duration_l1468_146841

theorem video_subscription_duration (monthly_cost : ℚ) (total_paid : ℚ) : 
  monthly_cost = 14 →
  total_paid = 84 →
  (total_paid / (monthly_cost / 2)) = 12 :=
by
  sorry

end video_subscription_duration_l1468_146841


namespace numerator_increase_percentage_numerator_increase_proof_l1468_146893

theorem numerator_increase_percentage (original_fraction : ℚ) 
  (denominator_decrease : ℚ) (resulting_fraction : ℚ) : ℚ :=
  let numerator_increase := 
    (resulting_fraction * (1 - denominator_decrease / 100) / original_fraction - 1) * 100
  numerator_increase

#check numerator_increase_percentage (3/4) 8 (15/16) = 15

theorem numerator_increase_proof :
  numerator_increase_percentage (3/4) 8 (15/16) = 15 := by sorry

end numerator_increase_percentage_numerator_increase_proof_l1468_146893


namespace shifted_function_sum_l1468_146883

-- Define the original function
def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 5

-- Define the shifted function
def g (x : ℝ) : ℝ := f (x + 6)

-- Define a, b, and c
def a : ℝ := 3
def b : ℝ := 38
def c : ℝ := 115

theorem shifted_function_sum (x : ℝ) : g x = a * x^2 + b * x + c ∧ a + b + c = 156 := by
  sorry

end shifted_function_sum_l1468_146883


namespace parallel_tangents_and_function_inequality_l1468_146801

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (2*a + 1) * x + 2 * Real.log x

/-- The function g(x) defined in the problem -/
def g (x : ℝ) : ℝ := x^2 - 2*x

/-- The derivative of f(x) with respect to x -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 2*a*x - (2*a + 1) + 2/x

theorem parallel_tangents_and_function_inequality (a : ℝ) (h_a : a > 0) :
  (f_deriv a 1 = f_deriv a 3 → a = 1/12) ∧
  (∀ x₁ ∈ Set.Ioc 0 2, ∃ x₂ ∈ Set.Ioc 0 2, f a x₁ < g x₂) ↔ 
  (0 < a ∧ a ≤ 1/4) :=
sorry

end parallel_tangents_and_function_inequality_l1468_146801


namespace max_value_implies_a_l1468_146823

def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

theorem max_value_implies_a (a : ℝ) :
  (∀ x, a - 2 ≤ x ∧ x ≤ a + 1 → f x ≤ 3) ∧
  (∃ x, a - 2 ≤ x ∧ x ≤ a + 1 ∧ f x = 3) →
  a = 0 ∨ a = -1 := by
sorry

end max_value_implies_a_l1468_146823


namespace last_two_digits_squares_l1468_146862

theorem last_two_digits_squares (a b : ℕ) :
  (50 ∣ (a + b) ∨ 50 ∣ (a - b)) → a^2 ≡ b^2 [ZMOD 100] := by
  sorry

end last_two_digits_squares_l1468_146862


namespace unique_two_digit_prime_sum_reverse_l1468_146845

def reverse_digits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem unique_two_digit_prime_sum_reverse : 
  ∃! n : ℕ, is_two_digit n ∧ Nat.Prime (n + reverse_digits n) :=
sorry

end unique_two_digit_prime_sum_reverse_l1468_146845


namespace evaluate_expression_l1468_146885

theorem evaluate_expression : 3000 * (3000 ^ 2999) = 3000 ^ 3000 := by
  sorry

end evaluate_expression_l1468_146885


namespace cookie_count_indeterminate_l1468_146868

theorem cookie_count_indeterminate (total_bananas : ℕ) (num_boxes : ℕ) (bananas_per_box : ℕ) 
  (h1 : total_bananas = 40)
  (h2 : num_boxes = 8)
  (h3 : bananas_per_box = 5)
  (h4 : total_bananas = num_boxes * bananas_per_box) :
  ¬ ∃ (cookie_count : ℕ), ∀ (n : ℕ), cookie_count = n :=
by
  sorry

end cookie_count_indeterminate_l1468_146868


namespace complex_imaginary_part_eq_two_l1468_146800

theorem complex_imaginary_part_eq_two (a : ℝ) :
  let z : ℂ := (1 + a * Complex.I) / (1 - Complex.I)
  Complex.im z = 2 → a = 3 := by
sorry

end complex_imaginary_part_eq_two_l1468_146800


namespace water_added_to_reach_new_ratio_l1468_146852

-- Define the initial mixture volume
def initial_volume : ℝ := 80

-- Define the initial ratio of milk to water
def initial_milk_ratio : ℝ := 7
def initial_water_ratio : ℝ := 3

-- Define the amount of water evaporated
def evaporated_water : ℝ := 8

-- Define the new ratio of milk to water
def new_milk_ratio : ℝ := 5
def new_water_ratio : ℝ := 4

-- Theorem to prove
theorem water_added_to_reach_new_ratio :
  let initial_milk := (initial_milk_ratio / (initial_milk_ratio + initial_water_ratio)) * initial_volume
  let initial_water := (initial_water_ratio / (initial_milk_ratio + initial_water_ratio)) * initial_volume
  let water_after_evaporation := initial_water - evaporated_water
  let x := (((new_water_ratio / new_milk_ratio) * initial_milk) - water_after_evaporation)
  x = 28.8 := by sorry

end water_added_to_reach_new_ratio_l1468_146852


namespace bonus_distribution_l1468_146895

theorem bonus_distribution (total_amount : ℕ) (total_notes : ℕ) 
  (h1 : total_amount = 160) 
  (h2 : total_notes = 25) : 
  ∃ (x y z : ℕ), 
    x + y + z = total_notes ∧ 
    2*x + 5*y + 10*z = total_amount ∧ 
    y = z ∧ 
    x = 5 ∧ y = 10 ∧ z = 10 := by
  sorry

end bonus_distribution_l1468_146895


namespace set_intersection_theorem_l1468_146869

def A : Set ℝ := {x | 2 * x - x^2 > 0}
def B : Set ℝ := {x | x > 1}

theorem set_intersection_theorem : A ∩ B = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end set_intersection_theorem_l1468_146869


namespace expression_value_l1468_146878

theorem expression_value (a b c k : ℤ) 
  (ha : a = 10) (hb : b = 15) (hc : c = 3) (hk : k = 2) :
  (a - (b - k * c)) - ((a - b) - k * c) = 12 := by
  sorry

end expression_value_l1468_146878


namespace quadratic_equation_solutions_l1468_146888

theorem quadratic_equation_solutions :
  ∀ x : ℝ, x^2 = x ↔ x = 0 ∨ x = 1 := by sorry

end quadratic_equation_solutions_l1468_146888


namespace rational_numbers_include_zero_not_only_positive_and_negative_rationals_l1468_146829

theorem rational_numbers_include_zero : ∃ (x : ℚ), x ≠ 0 ∧ x ≥ 0 ∧ x ≤ 0 := by
  sorry

theorem not_only_positive_and_negative_rationals : 
  ¬(∀ (x : ℚ), x > 0 ∨ x < 0) := by
  sorry

end rational_numbers_include_zero_not_only_positive_and_negative_rationals_l1468_146829


namespace cos_negative_120_degrees_l1468_146850

theorem cos_negative_120_degrees : Real.cos (-(120 * Real.pi / 180)) = -1/2 := by
  sorry

end cos_negative_120_degrees_l1468_146850


namespace triangle_cosine_sum_l1468_146810

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  -- Add conditions for a valid triangle here
  True

-- Define the angle C to be 60°
def angle_C_60 (A B C : ℝ × ℝ) : Prop :=
  -- Add condition for ∠C = 60° here
  True

-- Define D as the point where altitude from C meets AB
def altitude_C_D (A B C D : ℝ × ℝ) : Prop :=
  -- Add condition for D being on altitude from C here
  True

-- Define that the sides of triangle ABC are integers
def integer_sides (A B C : ℝ × ℝ) : Prop :=
  -- Add conditions for integer sides here
  True

-- Define BD = 17³
def BD_17_cubed (B D : ℝ × ℝ) : Prop :=
  -- Add condition for BD = 17³ here
  True

-- Define cos B = m/n where m and n are relatively prime positive integers
def cos_B_frac (B : ℝ × ℝ) (m n : ℕ) : Prop :=
  -- Add conditions for cos B = m/n and m, n coprime here
  True

theorem triangle_cosine_sum (A B C D : ℝ × ℝ) (m n : ℕ) :
  triangle_ABC A B C →
  angle_C_60 A B C →
  altitude_C_D A B C D →
  integer_sides A B C →
  BD_17_cubed B D →
  cos_B_frac B m n →
  m + n = 18 :=
by
  sorry

end triangle_cosine_sum_l1468_146810


namespace max_area_inscribed_equilateral_triangle_l1468_146873

/-- The maximum area of an equilateral triangle inscribed in a 13x14 rectangle --/
theorem max_area_inscribed_equilateral_triangle :
  ∃ (A : ℝ),
    A = (183 : ℝ) * Real.sqrt 3 ∧
    ∀ (s : ℝ),
      0 ≤ s →
      s ≤ 13 →
      s * Real.sqrt 3 / 2 ≤ 14 →
      s^2 * Real.sqrt 3 / 4 ≤ A :=
by sorry

#eval (183 : Nat) + 3 + 0

end max_area_inscribed_equilateral_triangle_l1468_146873


namespace garment_costs_l1468_146866

/-- The cost of garment A -/
def cost_A : ℝ := 300

/-- The cost of garment B -/
def cost_B : ℝ := 200

/-- The total cost of garments A and B -/
def total_cost : ℝ := 500

/-- The profit margin for garment A -/
def profit_margin_A : ℝ := 0.3

/-- The profit margin for garment B -/
def profit_margin_B : ℝ := 0.2

/-- The total profit -/
def total_profit : ℝ := 130

/-- Theorem: Given the conditions, the costs of garments A and B are 300 yuan and 200 yuan respectively -/
theorem garment_costs : 
  cost_A + cost_B = total_cost ∧ 
  profit_margin_A * cost_A + profit_margin_B * cost_B = total_profit := by
  sorry

end garment_costs_l1468_146866


namespace tens_digit_of_square_even_for_odd_numbers_up_to_99_l1468_146820

/-- The tens digit of a natural number -/
def tensDigit (n : ℕ) : ℕ :=
  (n / 10) % 10

/-- Predicate for odd numbers -/
def isOdd (n : ℕ) : Prop := n % 2 = 1

theorem tens_digit_of_square_even_for_odd_numbers_up_to_99 :
  ∀ n : ℕ, n ≤ 99 → isOdd n → Even (tensDigit (n^2)) := by sorry

end tens_digit_of_square_even_for_odd_numbers_up_to_99_l1468_146820


namespace line_slope_through_point_with_x_intercept_l1468_146804

/-- Given a line passing through the point (3, 4) with an x-intercept of 1, 
    its slope is 2. -/
theorem line_slope_through_point_with_x_intercept : 
  ∀ (f : ℝ → ℝ), 
    (∃ m b : ℝ, ∀ x, f x = m * x + b) →  -- f is a linear function
    f 3 = 4 →                           -- f passes through (3, 4)
    f 1 = 0 →                           -- x-intercept is 1
    ∃ m : ℝ, (∀ x, f x = m * x + b) ∧ m = 2 :=
by
  sorry


end line_slope_through_point_with_x_intercept_l1468_146804


namespace smallest_n_value_l1468_146871

def count_factors_of_five (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625)

theorem smallest_n_value (a b c m n : ℕ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 2010 →
  c = 710 →
  a.factorial * b.factorial * c.factorial = m * (10 ^ n) →
  ¬(10 ∣ m) →
  (∀ k, k < n → ∃ p, p > 0 ∧ ¬(10 ∣ p) ∧ 
    a.factorial * b.factorial * c.factorial ≠ p * (10 ^ k)) →
  n = 500 := by
  sorry

end smallest_n_value_l1468_146871


namespace perfect_squares_and_multiple_of_40_l1468_146811

theorem perfect_squares_and_multiple_of_40 :
  ∃ n : ℤ, ∃ a b : ℤ,
    (2 * n + 1 = a^2) ∧
    (3 * n + 1 = b^2) ∧
    (∃ k : ℤ, n = 40 * k) :=
sorry

end perfect_squares_and_multiple_of_40_l1468_146811


namespace product_repeating_nine_and_nine_l1468_146892

/-- The repeating decimal 0.999... -/
def repeating_decimal_nine : ℚ := 1

theorem product_repeating_nine_and_nine :
  repeating_decimal_nine * 9 = 9 := by sorry

end product_repeating_nine_and_nine_l1468_146892


namespace product_sum_of_three_numbers_l1468_146842

theorem product_sum_of_three_numbers 
  (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 99) 
  (h2 : a + b + c = 19) : 
  a*b + b*c + a*c = 131 := by
  sorry

end product_sum_of_three_numbers_l1468_146842
