import Mathlib

namespace wrapping_paper_area_formula_l2828_282814

/-- The area of square wrapping paper required to wrap a rectangular box -/
def wrapping_paper_area (l w h : ℝ) : ℝ :=
  (l + 4 + 2 * h) ^ 2

/-- Theorem stating the formula for the area of wrapping paper -/
theorem wrapping_paper_area_formula (l w h : ℝ) (hl : l > 0) (hw : w > 0) (hh : h > 0) :
  wrapping_paper_area l w h = l^2 + 8*l + 16 + 4*l*h + 16*h + 4*h^2 := by
  sorry

#check wrapping_paper_area_formula

end wrapping_paper_area_formula_l2828_282814


namespace star_property_counterexample_l2828_282844

/-- Definition of the star operation -/
def star (x y : ℝ) : ℝ := x^2 - 2*x*y + y^2

/-- Theorem stating that 2(x ★ y) ≠ (2x) ★ (2y) for some real x and y -/
theorem star_property_counterexample : ∃ x y : ℝ, 2 * (star x y) ≠ star (2*x) (2*y) := by
  sorry

end star_property_counterexample_l2828_282844


namespace systematic_sampling_l2828_282898

theorem systematic_sampling (total_students : ℕ) (sample_size : ℕ) (interval : ℕ) (start : ℕ) :
  total_students = 800 →
  sample_size = 50 →
  interval = 16 →
  start = 7 →
  ∃ (n : ℕ), n ≤ 4 ∧ 
    (start + (n - 1) * interval ≥ 49) ∧ 
    (start + (n - 1) * interval ≤ 64) ∧
    (start + (n - 1) * interval = 55) :=
by sorry

end systematic_sampling_l2828_282898


namespace min_winning_set_size_l2828_282864

/-- The set of allowed digits -/
def AllowedDigits : Finset Nat := {1, 2, 3, 4}

/-- A type representing a three-digit number using only allowed digits -/
structure ThreeDigitNumber where
  d1 : Nat
  d2 : Nat
  d3 : Nat
  h1 : d1 ∈ AllowedDigits
  h2 : d2 ∈ AllowedDigits
  h3 : d3 ∈ AllowedDigits

/-- Function to count how many digits differ between two ThreeDigitNumbers -/
def diffCount (n1 n2 : ThreeDigitNumber) : Nat :=
  (if n1.d1 ≠ n2.d1 then 1 else 0) +
  (if n1.d2 ≠ n2.d2 then 1 else 0) +
  (if n1.d3 ≠ n2.d3 then 1 else 0)

/-- A set of ThreeDigitNumbers is winning if for any other ThreeDigitNumber,
    at least one number in the set differs from it by at most one digit -/
def isWinningSet (s : Finset ThreeDigitNumber) : Prop :=
  ∀ n : ThreeDigitNumber, ∃ m ∈ s, diffCount n m ≤ 1

/-- The main theorem: The minimum size of a winning set is 8 -/
theorem min_winning_set_size :
  (∃ s : Finset ThreeDigitNumber, isWinningSet s ∧ s.card = 8) ∧
  (∀ s : Finset ThreeDigitNumber, isWinningSet s → s.card ≥ 8) :=
sorry

end min_winning_set_size_l2828_282864


namespace income_ratio_proof_l2828_282806

def uma_income : ℕ := 20000
def bala_income : ℕ := 15000
def uma_savings : ℕ := 5000
def bala_savings : ℕ := 5000
def expenditure_ratio : Rat := 3 / 2

theorem income_ratio_proof :
  let uma_expenditure := uma_income - uma_savings
  let bala_expenditure := bala_income - bala_savings
  (uma_expenditure : Rat) / bala_expenditure = expenditure_ratio →
  (uma_income : Rat) / bala_income = 4 / 3 := by
  sorry

end income_ratio_proof_l2828_282806


namespace current_speed_l2828_282888

theorem current_speed (boat_speed : ℝ) (upstream_time : ℝ) (downstream_time : ℝ) 
  (h1 : boat_speed = 16)
  (h2 : upstream_time = 20 / 60)
  (h3 : downstream_time = 15 / 60) :
  ∃ c : ℝ, c = 16 / 7 ∧ 
    (boat_speed - c) * upstream_time = (boat_speed + c) * downstream_time :=
by sorry

end current_speed_l2828_282888


namespace complete_set_is_reals_l2828_282854

def is_complete (A : Set ℝ) : Prop :=
  A.Nonempty ∧ ∀ a b : ℝ, (a + b) ∈ A → (a * b) ∈ A

theorem complete_set_is_reals (A : Set ℝ) : is_complete A → A = Set.univ := by
  sorry

end complete_set_is_reals_l2828_282854


namespace sin_equation_condition_l2828_282841

theorem sin_equation_condition (α β : Real) :
  (7 * 15 * Real.sin α + Real.sin β = Real.sin (α + β)) ↔
  (∃ k : ℤ, α = 2 * k * Real.pi ∨ β = 2 * k * Real.pi ∨ α + β = 2 * k * Real.pi) :=
by sorry

end sin_equation_condition_l2828_282841


namespace units_digit_theorem_l2828_282837

-- Define a function to get the units digit of a number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the property we want to prove
def propertyHolds (n : ℕ) : Prop :=
  n > 0 → unitsDigit ((35 ^ n) + (93 ^ 45)) = 8

-- The theorem statement
theorem units_digit_theorem :
  ∀ n : ℕ, propertyHolds n :=
sorry

end units_digit_theorem_l2828_282837


namespace rectangle_90_42_cut_result_l2828_282830

/-- Represents the dimensions of a rectangle in centimeters -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Represents the result of cutting a rectangle into squares -/
structure CutResult where
  squareCount : ℕ
  totalPerimeter : ℕ

/-- Cuts a rectangle into the maximum number of equal-sized squares -/
def cutIntoSquares (rect : Rectangle) : CutResult :=
  sorry

/-- Theorem stating the correct result for a 90cm × 42cm rectangle -/
theorem rectangle_90_42_cut_result :
  let rect : Rectangle := { length := 90, width := 42 }
  let result : CutResult := cutIntoSquares rect
  result.squareCount = 105 ∧ result.totalPerimeter = 2520 := by
  sorry

end rectangle_90_42_cut_result_l2828_282830


namespace marbles_per_pack_l2828_282871

theorem marbles_per_pack (total_marbles : ℕ) (total_packs : ℕ) 
  (leo_packs manny_packs neil_packs : ℕ) : 
  total_marbles = 400 →
  leo_packs = 25 →
  manny_packs = total_packs / 4 →
  neil_packs = total_packs / 8 →
  leo_packs + manny_packs + neil_packs = total_packs →
  total_marbles / total_packs = 10 := by
  sorry

end marbles_per_pack_l2828_282871


namespace unique_integer_proof_l2828_282851

theorem unique_integer_proof : ∃! n : ℕ+, 
  (24 ∣ n) ∧ 
  (8 < (n : ℝ) ^ (1/3)) ∧ 
  ((n : ℝ) ^ (1/3) < 8.2) ∧ 
  n = 528 := by
sorry

end unique_integer_proof_l2828_282851


namespace decimal_between_996_998_l2828_282801

theorem decimal_between_996_998 :
  ∃ x y : ℝ, x ≠ y ∧ 0.996 < x ∧ x < 0.998 ∧ 0.996 < y ∧ y < 0.998 :=
sorry

end decimal_between_996_998_l2828_282801


namespace inequality_solution_range_l2828_282842

theorem inequality_solution_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x + 3 > a^2 - 2*a - 1) ↔ (-1 < a ∧ a < 3) := by
  sorry

end inequality_solution_range_l2828_282842


namespace money_distribution_l2828_282886

theorem money_distribution (x : ℚ) : 
  x > 0 →
  let moe_initial := 6 * x
  let loki_initial := 5 * x
  let nick_initial := 4 * x
  let ott_received := 3 * x
  let total_money := moe_initial + loki_initial + nick_initial
  ott_received / total_money = 1 / 5 := by
  sorry

end money_distribution_l2828_282886


namespace sqrt_difference_equals_two_sqrt_three_l2828_282880

theorem sqrt_difference_equals_two_sqrt_three :
  Real.sqrt (7 + 4 * Real.sqrt 3) - Real.sqrt (7 - 4 * Real.sqrt 3) = 2 * Real.sqrt 3 := by
  sorry

end sqrt_difference_equals_two_sqrt_three_l2828_282880


namespace hyperbola_asymptotes_l2828_282857

/-- The asymptotes of the hyperbola x²/4 - y² = 1 are y = ±(1/2)x -/
theorem hyperbola_asymptotes (x y : ℝ) : 
  (x^2 / 4 - y^2 = 1) → 
  (∃ (k : ℝ), k = 1/2 ∧ (y = k*x ∨ y = -k*x)) :=
by sorry

end hyperbola_asymptotes_l2828_282857


namespace block_stacks_ratio_l2828_282819

theorem block_stacks_ratio : 
  ∀ (stack1 stack2 stack3 stack4 stack5 : ℕ),
  stack1 = 7 →
  stack2 = stack1 + 3 →
  stack3 = stack2 - 6 →
  stack4 = stack3 + 10 →
  stack1 + stack2 + stack3 + stack4 + stack5 = 55 →
  stack5 / stack2 = 2 :=
by sorry

end block_stacks_ratio_l2828_282819


namespace square_equation_solution_l2828_282836

theorem square_equation_solution : 
  ∃! y : ℤ, (2010 + y)^2 = y^2 :=
by
  -- The unique solution is y = -1005
  use -1005
  -- Proof goes here
  sorry

end square_equation_solution_l2828_282836


namespace toms_floor_replacement_cost_l2828_282802

/-- The total cost to replace a floor given room dimensions, removal cost, and new floor cost per square foot. -/
def total_floor_replacement_cost (length width removal_cost cost_per_sqft : ℝ) : ℝ :=
  removal_cost + length * width * cost_per_sqft

/-- Theorem stating that the total cost to replace the floor in Tom's room is $120. -/
theorem toms_floor_replacement_cost :
  total_floor_replacement_cost 8 7 50 1.25 = 120 := by
  sorry

#eval total_floor_replacement_cost 8 7 50 1.25

end toms_floor_replacement_cost_l2828_282802


namespace norma_cards_l2828_282853

/-- The number of cards Norma has after losing some -/
def cards_remaining (initial : ℕ) (lost : ℕ) : ℕ :=
  initial - lost

/-- Theorem: Norma has 18 cards remaining -/
theorem norma_cards : cards_remaining 88 70 = 18 := by
  sorry

end norma_cards_l2828_282853


namespace abs_neg_four_minus_six_l2828_282860

theorem abs_neg_four_minus_six : |-4 - 6| = 10 := by
  sorry

end abs_neg_four_minus_six_l2828_282860


namespace f_range_upper_bound_l2828_282800

def f (x : ℝ) : ℝ := -x^2 + 2*x

theorem f_range_upper_bound (a : ℝ) : 
  (∀ x ∈ Set.Ioo 0 2, a < f x) → a < 0 := by
  sorry

end f_range_upper_bound_l2828_282800


namespace triangle_angle_bisector_theorem_l2828_282873

/-- Given a triangle ABC with AB = 16 and AC = 5, where the angle bisectors of ∠ABC and ∠BCA 
    meet at point P inside the triangle such that AP = 4, prove that BC = 14. -/
theorem triangle_angle_bisector_theorem (A B C P : ℝ × ℝ) : 
  let d (X Y : ℝ × ℝ) := Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2)
  -- AB = 16
  d A B = 16 →
  -- AC = 5
  d A C = 5 →
  -- P is on the angle bisector of ∠ABC
  (d A P / d B P = d A C / d B C) →
  -- P is on the angle bisector of ∠BCA
  (d C P / d A P = d C B / d A B) →
  -- P is inside the triangle
  (0 < d A P ∧ d A P < d A B ∧ d A P < d A C) →
  -- AP = 4
  d A P = 4 →
  -- BC = 14
  d B C = 14 := by
sorry

end triangle_angle_bisector_theorem_l2828_282873


namespace min_value_and_inequality_l2828_282899

theorem min_value_and_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∃ m : ℝ, m = 6 ∧
    (∀ a' b' c' : ℝ, a' > 0 → b' > 0 → c' > 0 →
      1 / a'^3 + 1 / b'^3 + 1 / c'^3 + 3 * a' * b' * c' ≥ m) ∧
    (∀ a' b' c' : ℝ, a' > 0 → b' > 0 → c' > 0 →
      1 / a'^3 + 1 / b'^3 + 1 / c'^3 + 3 * a' * b' * c' = m →
      a' = 1 ∧ b' = 1 ∧ c' = 1)) ∧
  (∀ x : ℝ, abs (x + 1) - 2 * x < 6 ↔ x > -7/3) :=
sorry

end min_value_and_inequality_l2828_282899


namespace largest_four_digit_divisible_by_six_l2828_282809

theorem largest_four_digit_divisible_by_six : 
  ∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ n % 6 = 0 → n ≤ 9996 :=
by
  sorry

end largest_four_digit_divisible_by_six_l2828_282809


namespace original_number_l2828_282859

theorem original_number (x : ℚ) : (1 / x) - 2 = 5 / 4 → x = 4 / 13 := by
  sorry

end original_number_l2828_282859


namespace parabola_intersection_count_l2828_282815

/-- The parabola is defined by the function f(x) = 2x^2 - 4x + 1 --/
def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 1

/-- The number of intersection points between the parabola and the coordinate axes --/
def intersection_count : ℕ := 3

/-- Theorem stating that the parabola intersects the coordinate axes at exactly 3 points --/
theorem parabola_intersection_count :
  (∃! y, y = f 0) ∧ 
  (∃! x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) ∧
  intersection_count = 3 :=
sorry

end parabola_intersection_count_l2828_282815


namespace tangent_length_equals_hypotenuse_leg_l2828_282840

-- Define the triangle DEF
structure RightTriangle where
  DE : ℝ
  DF : ℝ
  rightAngleAtE : True

-- Define the circle
structure TangentCircle where
  centerOnDE : True
  tangentToDF : True
  tangentToEF : True

-- Define the theorem
theorem tangent_length_equals_hypotenuse_leg 
  (triangle : RightTriangle) 
  (circle : TangentCircle) 
  (h1 : triangle.DE = 7) 
  (h2 : triangle.DF = Real.sqrt 85) : 
  ∃ Q : ℝ × ℝ, ∃ FQ : ℝ, FQ = 6 :=
sorry

end tangent_length_equals_hypotenuse_leg_l2828_282840


namespace quadratic_one_solution_l2828_282826

theorem quadratic_one_solution (k : ℝ) : 
  (∃! x : ℝ, x^2 + 2*k*x + 1 = 0) ↔ k = 1 ∨ k = -1 := by
  sorry

end quadratic_one_solution_l2828_282826


namespace marked_price_calculation_l2828_282893

theorem marked_price_calculation (purchase_price : ℝ) (discount_percentage : ℝ) : 
  purchase_price = 50 ∧ discount_percentage = 60 → 
  (purchase_price / ((100 - discount_percentage) / 100)) / 2 = 62.50 := by
sorry

end marked_price_calculation_l2828_282893


namespace smallest_cube_root_plus_small_fraction_l2828_282875

theorem smallest_cube_root_plus_small_fraction (m n : ℕ) (r : ℝ) : 
  (m > 0) →
  (n > 0) →
  (r > 0) →
  (r < 1/500) →
  (m : ℝ)^(1/3) = n + r →
  (∀ m' n' r', m' > 0 → n' > 0 → r' > 0 → r' < 1/500 → (m' : ℝ)^(1/3) = n' + r' → m' ≥ m) →
  n = 13 := by
sorry

end smallest_cube_root_plus_small_fraction_l2828_282875


namespace debt_payment_calculation_l2828_282824

theorem debt_payment_calculation (total_installments : Nat) 
  (first_payments : Nat) (remaining_payments : Nat) (average_payment : ℚ) :
  total_installments = 52 →
  first_payments = 25 →
  remaining_payments = 27 →
  average_payment = 551.9230769230769 →
  ∃ (x : ℚ), 
    (x * first_payments + (x + 100) * remaining_payments) / total_installments = average_payment ∧
    x = 500 := by
  sorry

end debt_payment_calculation_l2828_282824


namespace voting_theorem_l2828_282818

/-- Represents the number of students voting for each issue and against all issues -/
structure VotingData where
  total : ℕ
  issueA : ℕ
  issueB : ℕ
  issueC : ℕ
  againstAll : ℕ

/-- Calculates the number of students voting for all three issues -/
def studentsVotingForAll (data : VotingData) : ℕ :=
  data.issueA + data.issueB + data.issueC - data.total + data.againstAll

/-- Theorem stating the number of students voting for all three issues -/
theorem voting_theorem (data : VotingData) 
    (h1 : data.total = 300)
    (h2 : data.issueA = 210)
    (h3 : data.issueB = 190)
    (h4 : data.issueC = 160)
    (h5 : data.againstAll = 40) :
  studentsVotingForAll data = 80 := by
  sorry

#eval studentsVotingForAll { total := 300, issueA := 210, issueB := 190, issueC := 160, againstAll := 40 }

end voting_theorem_l2828_282818


namespace S_n_min_l2828_282839

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  first_term : a 1 = -11
  sum_5_6 : a 5 + a 6 = -4

/-- The sum of the first n terms of the arithmetic sequence -/
def S_n (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n^2 - 12*n

/-- The theorem stating that S_n reaches its minimum when n = 6 -/
theorem S_n_min (seq : ArithmeticSequence) :
  ∀ n : ℕ, n ≠ 0 → S_n seq 6 ≤ S_n seq n :=
sorry

end S_n_min_l2828_282839


namespace percentage_in_quarters_l2828_282896

def dimes : ℕ := 80
def quarters : ℕ := 30
def nickels : ℕ := 40

def dime_value : ℕ := 10
def quarter_value : ℕ := 25
def nickel_value : ℕ := 5

def total_value : ℕ := dimes * dime_value + quarters * quarter_value + nickels * nickel_value
def quarters_value : ℕ := quarters * quarter_value

theorem percentage_in_quarters : 
  (quarters_value : ℚ) / total_value * 100 = 3/7 * 100 := by sorry

end percentage_in_quarters_l2828_282896


namespace sample_first_year_300_l2828_282828

/-- Represents the ratio of students in each grade -/
structure GradeRatio :=
  (first second third fourth : ℕ)

/-- Calculates the number of first-year students to be sampled given the total sample size and grade ratio -/
def sampleFirstYear (totalSample : ℕ) (ratio : GradeRatio) : ℕ :=
  let totalRatio := ratio.first + ratio.second + ratio.third + ratio.fourth
  (totalSample * ratio.first) / totalRatio

/-- Theorem stating that for a sample size of 300 and ratio 4:5:5:6, the number of first-year students sampled is 60 -/
theorem sample_first_year_300 :
  sampleFirstYear 300 ⟨4, 5, 5, 6⟩ = 60 := by
  sorry

#eval sampleFirstYear 300 ⟨4, 5, 5, 6⟩

end sample_first_year_300_l2828_282828


namespace inequality_solution_set_l2828_282866

theorem inequality_solution_set :
  {x : ℝ | x * (x - 1) > 0} = {x : ℝ | x < 0 ∨ x > 1} := by
sorry

end inequality_solution_set_l2828_282866


namespace associate_prof_charts_l2828_282869

theorem associate_prof_charts (total_people : ℕ) (total_pencils : ℕ) (total_charts : ℕ)
  (h1 : total_people = 8)
  (h2 : total_pencils = 10)
  (h3 : total_charts = 14) :
  ∃ (assoc_prof : ℕ) (asst_prof : ℕ) (charts_per_assoc : ℕ),
    assoc_prof + asst_prof = total_people ∧
    2 * assoc_prof + asst_prof = total_pencils ∧
    charts_per_assoc * assoc_prof + 2 * asst_prof = total_charts ∧
    charts_per_assoc = 1 :=
by sorry

end associate_prof_charts_l2828_282869


namespace midpoint_property_l2828_282872

/-- Given two points A and B in the plane, if C is their midpoint,
    then 3 times the x-coordinate of C minus 5 times the y-coordinate of C equals 6. -/
theorem midpoint_property (A B : ℝ × ℝ) (h : A = (20, 10) ∧ B = (4, 2)) :
  let C : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  3 * C.1 - 5 * C.2 = 6 := by sorry

end midpoint_property_l2828_282872


namespace cookie_earnings_proof_l2828_282813

/-- The amount earned by girl scouts from selling cookies -/
def cookie_earnings : ℝ := 30

/-- The cost per person to go to the pool -/
def pool_cost_per_person : ℝ := 2.5

/-- The number of people going to the pool -/
def number_of_people : ℕ := 10

/-- The amount left after paying for the pool -/
def amount_left : ℝ := 5

/-- Theorem stating that the cookie earnings equal $30 -/
theorem cookie_earnings_proof :
  cookie_earnings = pool_cost_per_person * number_of_people + amount_left :=
by sorry

end cookie_earnings_proof_l2828_282813


namespace intersection_A_B_l2828_282822

open Set Real

-- Define sets A and B
def A : Set ℝ := {x | x^2 - x - 2 > 0}
def B : Set ℝ := Ioo 0 3

-- State the theorem
theorem intersection_A_B : A ∩ B = Ioo 2 3 := by sorry

end intersection_A_B_l2828_282822


namespace tablet_cash_savings_l2828_282833

/-- Represents the savings when buying a tablet in cash versus installments -/
def tablet_savings (cash_price : ℕ) (down_payment : ℕ) 
  (first_4_months : ℕ) (next_4_months : ℕ) (last_4_months : ℕ) : ℕ :=
  (down_payment + 4 * first_4_months + 4 * next_4_months + 4 * last_4_months) - cash_price

/-- Theorem stating the savings when buying the tablet in cash -/
theorem tablet_cash_savings : 
  tablet_savings 450 100 40 35 30 = 70 := by
  sorry

end tablet_cash_savings_l2828_282833


namespace f_properties_l2828_282887

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (2 * x + 1) + 2 * a * x - 4 * a * Real.exp x + 4

theorem f_properties (a : ℝ) (h : a > 0) :
  (∃ x, f 1 x ≤ f 1 0) ∧
  ((0 < a ∧ a < 1 → ∃ x₁ x₂, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) ∧
   (a = 1 → ∃! x, f a x = 0) ∧
   (a > 1 → ∀ x, f a x ≠ 0)) :=
by sorry

end f_properties_l2828_282887


namespace singing_competition_average_age_l2828_282855

theorem singing_competition_average_age 
  (num_females : Nat) 
  (num_males : Nat)
  (avg_age_females : ℝ) 
  (avg_age_males : ℝ) :
  num_females = 12 →
  num_males = 18 →
  avg_age_females = 25 →
  avg_age_males = 40 →
  (num_females * avg_age_females + num_males * avg_age_males) / (num_females + num_males) = 34 := by
sorry

end singing_competition_average_age_l2828_282855


namespace muscovy_duck_percentage_l2828_282845

theorem muscovy_duck_percentage (total_ducks : ℕ) (female_muscovy : ℕ) 
  (h1 : total_ducks = 40)
  (h2 : female_muscovy = 6)
  (h3 : (female_muscovy : ℝ) / ((total_ducks : ℝ) * 0.5) = 0.3) :
  (total_ducks : ℝ) * 0.5 = (total_ducks : ℝ) * 0.5 := by
  sorry

#check muscovy_duck_percentage

end muscovy_duck_percentage_l2828_282845


namespace sphere_radius_in_cone_l2828_282810

/-- A right circular cone with four congruent spheres inside -/
structure ConeSpheres where
  base_radius : ℝ
  height : ℝ
  sphere_radius : ℝ
  is_right_circular : base_radius > 0 ∧ height > 0
  spheres_tangent : sphere_radius > 0
  spheres_fit : sphere_radius < base_radius ∧ sphere_radius < height

/-- The theorem stating the radius of each sphere in the specific configuration -/
theorem sphere_radius_in_cone (cs : ConeSpheres) 
  (h1 : cs.base_radius = 8)
  (h2 : cs.height = 15) :
  cs.sphere_radius = 8 * Real.sqrt 3 / 17 := by
  sorry


end sphere_radius_in_cone_l2828_282810


namespace election_winner_votes_l2828_282808

theorem election_winner_votes (total_votes : ℕ) (winner_percentage : ℚ) (vote_difference : ℕ) :
  winner_percentage = 60 / 100 →
  vote_difference = 288 →
  winner_percentage * total_votes - (1 - winner_percentage) * total_votes = vote_difference →
  winner_percentage * total_votes = 864 :=
by
  sorry

end election_winner_votes_l2828_282808


namespace position_selection_count_l2828_282804

/-- The number of people in the group --/
def group_size : ℕ := 6

/-- The number of positions to be filled --/
def num_positions : ℕ := 3

/-- Theorem: The number of ways to choose a President, Vice-President, and Secretary
    from a group of 6 people, where all positions must be held by different individuals,
    is equal to 120. --/
theorem position_selection_count :
  (group_size.factorial) / ((group_size - num_positions).factorial) = 120 := by
  sorry

end position_selection_count_l2828_282804


namespace chess_pieces_present_l2828_282812

/-- The number of pieces in a standard chess set -/
def standard_chess_pieces : ℕ := 32

/-- The number of missing chess pieces -/
def missing_pieces : ℕ := 8

/-- Theorem: The number of chess pieces present is 24 -/
theorem chess_pieces_present : 
  standard_chess_pieces - missing_pieces = 24 := by
  sorry

end chess_pieces_present_l2828_282812


namespace volume_of_CO2_released_l2828_282856

/-- The volume of CO₂ gas released in a chemical reaction --/
theorem volume_of_CO2_released (n : ℝ) (Vₘ : ℝ) (h1 : n = 2.4) (h2 : Vₘ = 22.4) :
  n * Vₘ = 53.76 := by
  sorry

end volume_of_CO2_released_l2828_282856


namespace circle_properties_l2828_282850

-- Define the circle C
def circle_C (center : ℝ × ℝ) (radius : ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the conditions
def center_on_line (a : ℝ) : ℝ × ℝ := (a, -2*a)
def point_A : ℝ × ℝ := (2, -1)
def tangent_line (p : ℝ × ℝ) : Prop := p.1 + p.2 = 1

-- Theorem statement
theorem circle_properties (a : ℝ) :
  let center := center_on_line a
  let C := circle_C center (|a - 2*a - 1| / Real.sqrt 2)
  point_A ∈ C ∧ (∃ p, p ∈ C ∧ tangent_line p) →
  (C = circle_C (1, -2) (Real.sqrt 2)) ∧
  (Set.Icc (-3 : ℝ) (-1) ⊆ {y | (0, y) ∈ C}) ∧
  (Set.Ioo (-3 : ℝ) (-1) ⊆ {y | (0, y) ∉ C}) :=
sorry

end circle_properties_l2828_282850


namespace total_savings_after_three_months_l2828_282821

def savings (n : ℕ) : ℕ := 10 + 30 * n

theorem total_savings_after_three_months : 
  savings 0 + savings 1 + savings 2 = 70 := by
  sorry

end total_savings_after_three_months_l2828_282821


namespace inverse_proportion_l2828_282867

theorem inverse_proportion (a b : ℝ → ℝ) (k : ℝ) :
  (∀ x, a x * b x = k) →  -- a and b are inversely proportional
  (a 5 = 40) →            -- a = 40 when b = 5
  (b 5 = 5) →             -- explicitly stating b = 5
  (b 10 = 10) →           -- explicitly stating b = 10
  (a 10 = 20) :=          -- a = 20 when b = 10
by
  sorry


end inverse_proportion_l2828_282867


namespace equation_solutions_l2828_282889

theorem equation_solutions : 
  {x : ℝ | (1 / (x^2 + 13*x - 8) + 1 / (x^2 + 3*x - 8) + 1 / (x^2 - 15*x - 8) = 0)} = {8, 1, -1, -8} := by
  sorry

end equation_solutions_l2828_282889


namespace cubic_minimum_condition_l2828_282877

/-- A cubic function with parameters p, q, and r -/
def cubic_function (p q r x : ℝ) : ℝ := x^3 + 3*p*x^2 + 3*q*x + r

/-- The derivative of the cubic function with respect to x -/
def cubic_derivative (p q x : ℝ) : ℝ := 3*x^2 + 6*p*x + 3*q

theorem cubic_minimum_condition (p q r : ℝ) :
  (∀ x : ℝ, cubic_function p q r x ≥ cubic_function p q r (-p)) ∧
  cubic_function p q r (-p) = -27 →
  r = -27 - 2*p^3 + 3*p*q :=
by sorry

end cubic_minimum_condition_l2828_282877


namespace robotics_club_enrollment_l2828_282847

theorem robotics_club_enrollment (total : ℕ) (cs : ℕ) (electronics : ℕ) (both : ℕ) 
  (h1 : total = 60)
  (h2 : cs = 45)
  (h3 : electronics = 33)
  (h4 : both = 25) :
  total - (cs + electronics - both) = 7 := by
  sorry

end robotics_club_enrollment_l2828_282847


namespace mark_reading_time_l2828_282897

/-- Calculates Mark's total weekly reading time given his daily reading time and weekly increase. -/
def weekly_reading_time (x : ℝ) (y : ℝ) : ℝ :=
  7 * x + y

/-- Theorem stating that Mark's total weekly reading time is 7x + y hours -/
theorem mark_reading_time (x y : ℝ) :
  weekly_reading_time x y = 7 * x + y := by
  sorry

end mark_reading_time_l2828_282897


namespace infinitely_many_square_averages_l2828_282884

theorem infinitely_many_square_averages :
  ∃ f : ℕ → ℕ, 
    (f 0 = 1) ∧ 
    (∀ k : ℕ, f k < f (k + 1)) ∧
    (∀ k : ℕ, ∃ m : ℕ, (f k * (f k + 1) * (2 * f k + 1)) / 6 = m^2 * f k) :=
sorry

end infinitely_many_square_averages_l2828_282884


namespace contrapositive_equivalence_l2828_282834

theorem contrapositive_equivalence (a b : ℝ) :
  (¬(a = 0 ∧ b = 0) → a^2 + b^2 ≠ 0) ↔ (a^2 + b^2 = 0 → a = 0 ∧ b = 0) :=
by sorry

end contrapositive_equivalence_l2828_282834


namespace min_value_of_D_l2828_282891

noncomputable def D (x a : ℝ) : ℝ :=
  Real.sqrt ((x - a^2) + (Real.log x - a^2 / 4)^2) + a^2 / 4 + 1

theorem min_value_of_D :
  ∃ (m : ℝ), ∀ (x a : ℝ), D x a ≥ m ∧ ∃ (x₀ a₀ : ℝ), D x₀ a₀ = m ∧ m = Real.sqrt 2 :=
sorry

end min_value_of_D_l2828_282891


namespace base_8_4512_equals_2378_l2828_282881

def base_8_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

theorem base_8_4512_equals_2378 :
  base_8_to_10 [2, 1, 5, 4] = 2378 := by
  sorry

end base_8_4512_equals_2378_l2828_282881


namespace quadratic_root_transformations_l2828_282846

/-- Given a quadratic equation x^2 + px + q = 0, this theorem proves the equations
    with roots differing by sign and reciprocal roots. -/
theorem quadratic_root_transformations (p q : ℝ) :
  let original := fun x : ℝ => x^2 + p*x + q
  let opposite_sign := fun x : ℝ => x^2 - p*x + q
  let reciprocal := fun x : ℝ => q*x^2 + p*x + 1
  (∀ x, original x = 0 → ∃ y, opposite_sign y = 0 ∧ y = -x) ∧
  (∀ x, original x = 0 → x ≠ 0 → ∃ y, reciprocal y = 0 ∧ y = 1/x) :=
by sorry

end quadratic_root_transformations_l2828_282846


namespace smallest_number_with_given_remainders_l2828_282827

theorem smallest_number_with_given_remainders : ∃! n : ℕ, 
  (n % 6 = 2) ∧ (n % 5 = 3) ∧ (n % 7 = 1) ∧
  (∀ m : ℕ, m < n → ¬((m % 6 = 2) ∧ (m % 5 = 3) ∧ (m % 7 = 1))) :=
by
  -- The proof goes here
  sorry

end smallest_number_with_given_remainders_l2828_282827


namespace jane_has_66_robots_l2828_282848

/-- The number of car robots each person has -/
structure CarRobots where
  tom : ℕ
  michael : ℕ
  bob : ℕ
  sarah : ℕ
  jane : ℕ

/-- The conditions of the car robot collections -/
def satisfiesConditions (c : CarRobots) : Prop :=
  c.tom = 15 ∧
  c.michael = 3 * c.tom - 5 ∧
  c.bob = 8 * (c.tom + c.michael) ∧
  c.sarah = c.bob / 2 - 7 ∧
  c.jane = (c.sarah - c.tom) / 3

/-- Theorem stating that Jane has 66 car robots -/
theorem jane_has_66_robots (c : CarRobots) (h : satisfiesConditions c) : c.jane = 66 := by
  sorry

end jane_has_66_robots_l2828_282848


namespace parabola_triangle_circumradius_range_l2828_282883

/-- A point on a parabola y = x^2 -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y = x^2

/-- Triangle on a parabola y = x^2 -/
structure ParabolaTriangle where
  A : ParabolaPoint
  B : ParabolaPoint
  C : ParabolaPoint
  distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C

/-- The circumradius of a triangle -/
def circumradius (t : ParabolaTriangle) : ℝ :=
  sorry  -- Definition of circumradius

theorem parabola_triangle_circumradius_range :
  ∀ (t : ParabolaTriangle), circumradius t > 1/2 ∧
  ∀ (r : ℝ), r > 1/2 → ∃ (t : ParabolaTriangle), circumradius t = r :=
by sorry

end parabola_triangle_circumradius_range_l2828_282883


namespace exists_large_configuration_l2828_282843

/-- A configuration in the plane is a finite set of points where each point
    has at least k other points at a distance of exactly 1 unit. -/
def IsConfiguration (S : Set (ℝ × ℝ)) (k : ℕ) : Prop :=
  S.Finite ∧ 
  ∀ P ∈ S, ∃ T ⊆ S, T.ncard ≥ k ∧ ∀ Q ∈ T, Q ≠ P ∧ dist P Q = 1

/-- There exists a configuration of 3^1000 points where each point
    has at least 2000 other points at a distance of 1 unit. -/
theorem exists_large_configuration :
  ∃ S : Set (ℝ × ℝ), IsConfiguration S 2000 ∧ S.ncard = 3^1000 := by
  sorry


end exists_large_configuration_l2828_282843


namespace age_difference_l2828_282885

theorem age_difference (A B : ℕ) : B = 35 → A + 10 = 2 * (B - 10) → A - B = 5 := by
  sorry

end age_difference_l2828_282885


namespace intersection_of_lines_l2828_282816

theorem intersection_of_lines : ∃! p : ℚ × ℚ, 
  8 * p.1 - 3 * p.2 = 24 ∧ 5 * p.1 + 2 * p.2 = 17 ∧ p = (99/31, 16/31) := by
  sorry

end intersection_of_lines_l2828_282816


namespace frank_candy_count_l2828_282835

/-- Given a number of bags, pieces per bag, and leftover pieces, 
    calculates the total number of candy pieces. -/
def total_candy (bags : ℕ) (pieces_per_bag : ℕ) (leftover : ℕ) : ℕ :=
  bags * pieces_per_bag + leftover

/-- Proves that with 37 bags of 46 pieces each and 5 leftover pieces, 
    the total number of candy pieces is 1707. -/
theorem frank_candy_count : total_candy 37 46 5 = 1707 := by
  sorry

end frank_candy_count_l2828_282835


namespace savings_over_three_years_l2828_282820

def multi_tariff_meter_cost : ℕ := 3500
def installation_cost : ℕ := 1100
def monthly_consumption : ℕ := 300
def night_consumption : ℕ := 230
def day_consumption : ℕ := monthly_consumption - night_consumption
def multi_tariff_day_rate : ℚ := 52/10
def multi_tariff_night_rate : ℚ := 34/10
def standard_rate : ℚ := 46/10

def monthly_cost_multi_tariff : ℚ :=
  (night_consumption : ℚ) * multi_tariff_night_rate + (day_consumption : ℚ) * multi_tariff_day_rate

def monthly_cost_standard : ℚ :=
  (monthly_consumption : ℚ) * standard_rate

def total_cost_multi_tariff (months : ℕ) : ℚ :=
  (multi_tariff_meter_cost : ℚ) + (installation_cost : ℚ) + monthly_cost_multi_tariff * (months : ℚ)

def total_cost_standard (months : ℕ) : ℚ :=
  monthly_cost_standard * (months : ℚ)

theorem savings_over_three_years :
  total_cost_standard 36 - total_cost_multi_tariff 36 = 3824 := by sorry

end savings_over_three_years_l2828_282820


namespace y_range_l2828_282890

theorem y_range (a b y : ℝ) (h1 : a + b = 2) (h2 : b ≤ 2) (h3 : y - a^2 - 2*a + 2 = 0) :
  y ≥ -2 := by
sorry

end y_range_l2828_282890


namespace current_rate_calculation_l2828_282858

/-- Given a boat with speed in still water and its downstream travel details, 
    calculate the rate of the current. -/
theorem current_rate_calculation 
  (boat_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (h1 : boat_speed = 42)
  (h2 : downstream_distance = 33)
  (h3 : downstream_time = 44 / 60) : 
  (downstream_distance / downstream_time) - boat_speed = 3 := by
  sorry

end current_rate_calculation_l2828_282858


namespace apples_to_eat_raw_l2828_282868

theorem apples_to_eat_raw (total : ℕ) (wormy : ℕ) (bruised : ℕ) : 
  total = 85 →
  wormy = total / 5 →
  bruised = wormy + 9 →
  total - bruised - wormy = 42 :=
by
  sorry

end apples_to_eat_raw_l2828_282868


namespace min_jumps_to_visit_all_l2828_282823

/-- Represents a jump on the circle -/
inductive Jump
| Two  : Jump  -- Jump of 2 points
| Three : Jump -- Jump of 3 points

/-- The number of points on the circle -/
def numPoints : ℕ := 2016

/-- Function to calculate the total distance covered by a sequence of jumps -/
def totalDistance (jumps : List Jump) : ℕ :=
  jumps.foldl (fun acc jump => acc + match jump with
    | Jump.Two => 2
    | Jump.Three => 3) 0

/-- Predicate to check if a sequence of jumps visits all points -/
def visitsAllPoints (jumps : List Jump) : Prop :=
  totalDistance jumps % numPoints = 0 ∧ 
  jumps.length ≥ numPoints

/-- The main theorem stating the minimum number of jumps required -/
theorem min_jumps_to_visit_all : 
  ∃ (jumps : List Jump), visitsAllPoints jumps ∧ 
    jumps.length = 2017 ∧ 
    (∀ (other_jumps : List Jump), visitsAllPoints other_jumps → 
      other_jumps.length ≥ 2017) := by
  sorry

end min_jumps_to_visit_all_l2828_282823


namespace compare_negative_fractions_l2828_282838

theorem compare_negative_fractions : -4/3 < -5/4 := by
  sorry

end compare_negative_fractions_l2828_282838


namespace pythagorean_numbers_l2828_282825

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem pythagorean_numbers : 
  (is_pythagorean_triple 9 12 15) ∧ 
  (¬ is_pythagorean_triple 3 4 5) ∧ 
  (¬ is_pythagorean_triple 1 1 2) :=
by
  sorry

end pythagorean_numbers_l2828_282825


namespace limit_of_sequence_l2828_282863

/-- The sum of the first n multiples of 3 -/
def S (n : ℕ) : ℕ := 3 * n * (n + 1) / 2

/-- The sequence we're interested in -/
def a (n : ℕ) : ℚ := (S n : ℚ) / (n^2 + 4 : ℚ)

theorem limit_of_sequence :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |a n - 3/2| < ε :=
sorry

end limit_of_sequence_l2828_282863


namespace simplify_sqrt_500_l2828_282862

theorem simplify_sqrt_500 : Real.sqrt 500 = 10 * Real.sqrt 5 := by
  sorry

end simplify_sqrt_500_l2828_282862


namespace ab_range_l2828_282865

theorem ab_range (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 4 * a * b = a + b) : a * b ≥ 1/4 := by
  sorry

end ab_range_l2828_282865


namespace contradictory_implies_mutually_exclusive_but_not_conversely_l2828_282892

/-- Two events are mutually exclusive if they cannot occur simultaneously -/
def mutually_exclusive (A B : Set Ω) : Prop := A ∩ B = ∅

/-- Two events are contradictory if one event is the complement of the other -/
def contradictory (A B : Set Ω) : Prop := A = Bᶜ

/-- Theorem: Contradictory events are mutually exclusive, but mutually exclusive events are not necessarily contradictory -/
theorem contradictory_implies_mutually_exclusive_but_not_conversely :
  (∀ A B : Set Ω, contradictory A B → mutually_exclusive A B) ∧
  ¬(∀ A B : Set Ω, mutually_exclusive A B → contradictory A B) := by
  sorry

end contradictory_implies_mutually_exclusive_but_not_conversely_l2828_282892


namespace radical_axes_theorem_l2828_282876

/-- A circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The radical axis of two circles --/
def radicalAxis (c1 c2 : Circle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - c1.center.1)^2 + (p.2 - c1.center.2)^2 - c1.radius^2 = 
               (p.1 - c2.center.1)^2 + (p.2 - c2.center.2)^2 - c2.radius^2}

/-- Three lines are either coincident, parallel, or concurrent --/
def linesCoincidentParallelOrConcurrent (l1 l2 l3 : Set (ℝ × ℝ)) : Prop :=
  (l1 = l2 ∧ l2 = l3) ∨ 
  (∀ p1 ∈ l1, ∀ p2 ∈ l2, ∀ p3 ∈ l3, 
    (p1.1 - p2.1) * (p3.2 - p2.2) = (p3.1 - p2.1) * (p1.2 - p2.2)) ∨
  (∃ p : ℝ × ℝ, p ∈ l1 ∧ p ∈ l2 ∧ p ∈ l3)

/-- The Theorem of Radical Axes --/
theorem radical_axes_theorem (c1 c2 c3 : Circle) :
  linesCoincidentParallelOrConcurrent 
    (radicalAxis c1 c2) 
    (radicalAxis c2 c3) 
    (radicalAxis c3 c1) :=
sorry

end radical_axes_theorem_l2828_282876


namespace julia_tag_game_l2828_282829

theorem julia_tag_game (total : ℕ) (monday : ℕ) (tuesday : ℕ) 
  (h1 : total = 18) 
  (h2 : monday = 4) 
  (h3 : total = monday + tuesday) : 
  tuesday = 14 := by
  sorry

end julia_tag_game_l2828_282829


namespace verandah_area_is_124_l2828_282811

/-- Calculates the area of a verandah surrounding a rectangular room. -/
def verandahArea (roomLength : ℝ) (roomWidth : ℝ) (verandahWidth : ℝ) : ℝ :=
  (roomLength + 2 * verandahWidth) * (roomWidth + 2 * verandahWidth) - roomLength * roomWidth

/-- Theorem: The area of the verandah is 124 square meters. -/
theorem verandah_area_is_124 :
  verandahArea 15 12 2 = 124 := by
  sorry

#eval verandahArea 15 12 2

end verandah_area_is_124_l2828_282811


namespace line_slope_l2828_282874

theorem line_slope (x y : ℝ) : 4 * y + 2 * x = 10 → (y - 2.5) / x = -1 / 2 := by sorry

end line_slope_l2828_282874


namespace existence_of_finite_sequence_no_infinite_sequence_l2828_282882

/-- S(k) denotes the sum of all digits of a positive integer k in its decimal representation. -/
def S (k : ℕ+) : ℕ :=
  sorry

/-- For any positive integer n, there exists an arithmetic sequence of positive integers
    a₁, a₂, ..., aₙ such that S(a₁) < S(a₂) < ... < S(aₙ). -/
theorem existence_of_finite_sequence (n : ℕ+) :
  ∃ (a : ℕ+ → ℕ+) (d : ℕ+), (∀ i : ℕ+, i ≤ n → a i = a 1 + (i - 1) * d) ∧
    (∀ i : ℕ+, i < n → S (a i) < S (a (i + 1))) :=
  sorry

/-- There does not exist an infinite arithmetic sequence of positive integers {aₙ}
    such that S(a₁) < S(a₂) < ... < S(aₙ) < ... -/
theorem no_infinite_sequence :
  ¬ ∃ (a : ℕ+ → ℕ+) (d : ℕ+), (∀ i : ℕ+, a i = a 1 + (i - 1) * d) ∧
    (∀ i j : ℕ+, i < j → S (a i) < S (a j)) :=
  sorry

end existence_of_finite_sequence_no_infinite_sequence_l2828_282882


namespace faster_train_speed_l2828_282832

/-- Calculates the speed of a faster train given the conditions of the problem -/
theorem faster_train_speed
  (train_length : ℝ)
  (slower_speed : ℝ)
  (passing_time : ℝ)
  (h1 : train_length = 100)  -- Length of each train in meters
  (h2 : slower_speed = 36)   -- Speed of slower train in km/hr
  (h3 : passing_time = 72)   -- Time to pass in seconds
  : ∃ (faster_speed : ℝ), faster_speed = 86 :=
by
  sorry

#check faster_train_speed

end faster_train_speed_l2828_282832


namespace point_in_fourth_quadrant_l2828_282807

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def is_in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The theorem to prove -/
theorem point_in_fourth_quadrant :
  let P : Point := ⟨3, -3⟩
  is_in_fourth_quadrant P := by
  sorry

end point_in_fourth_quadrant_l2828_282807


namespace chessboard_separating_edges_l2828_282895

/-- Represents a square on the chessboard -/
inductive Square
| White
| Black

/-- Represents the chessboard -/
def Chessboard (n : ℕ) := Fin n → Fin n → Square

/-- Counts the number of white squares on the border of the chessboard -/
def countWhiteBorderSquares (n : ℕ) (board : Chessboard n) : ℕ := sorry

/-- Counts the number of black squares on the border of the chessboard -/
def countBlackBorderSquares (n : ℕ) (board : Chessboard n) : ℕ := sorry

/-- Counts the number of edges inside the board that separate squares of different colors -/
def countSeparatingEdges (n : ℕ) (board : Chessboard n) : ℕ := sorry

/-- Main theorem: If a chessboard has at least n white and n black squares on its border,
    then there are at least n edges inside the board separating different colors -/
theorem chessboard_separating_edges (n : ℕ) (board : Chessboard n) :
  countWhiteBorderSquares n board ≥ n →
  countBlackBorderSquares n board ≥ n →
  countSeparatingEdges n board ≥ n := by sorry

end chessboard_separating_edges_l2828_282895


namespace smallest_a_l2828_282803

/-- The polynomial x³ - ax² + bx - 2010 with three positive integer zeros -/
def polynomial (a b x : ℤ) : ℤ := x^3 - a*x^2 + b*x - 2010

/-- The polynomial has three positive integer zeros -/
def has_three_positive_integer_zeros (a b : ℤ) : Prop :=
  ∃ (x y z : ℤ), x > 0 ∧ y > 0 ∧ z > 0 ∧
    polynomial a b x = 0 ∧ polynomial a b y = 0 ∧ polynomial a b z = 0

/-- The smallest possible value of a is 78 -/
theorem smallest_a (a b : ℤ) :
  has_three_positive_integer_zeros a b → a ≥ 78 :=
sorry

end smallest_a_l2828_282803


namespace system1_solution_system2_solution_l2828_282849

-- System 1
theorem system1_solution :
  ∃ (x y : ℝ), 2*x + 3*y = -1 ∧ y = 4*x - 5 ∧ x = 1 ∧ y = -1 := by
  sorry

-- System 2
theorem system2_solution :
  ∃ (x y : ℝ), 3*x + 2*y = 20 ∧ 4*x - 5*y = 19 ∧ x = 6 ∧ y = 1 := by
  sorry

end system1_solution_system2_solution_l2828_282849


namespace no_lattice_polygon1994_l2828_282894

/-- A polygon with 1994 sides where side lengths are √(i^2 + 4) -/
def Polygon1994 : Type :=
  { vertices : Fin 1995 → ℤ × ℤ // 
    ∀ i : Fin 1994, 
      let (x₁, y₁) := vertices i
      let (x₂, y₂) := vertices (i + 1)
      (x₂ - x₁)^2 + (y₂ - y₁)^2 = i^2 + 4 ∧
    vertices 0 = vertices 1994 }

/-- Theorem stating that such a polygon cannot exist with all vertices on lattice points -/
theorem no_lattice_polygon1994 : ¬ ∃ (p : Polygon1994), True := by
  sorry

end no_lattice_polygon1994_l2828_282894


namespace arithmetic_sequence_properties_l2828_282805

/-- An arithmetic sequence with given conditions -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  a_4 : a 4 = 70
  a_21 : a 21 = -100

/-- Main theorem about the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (seq.a 1 = 100) ∧ 
  (∀ n, seq.a (n + 1) - seq.a n = -10) ∧
  (∀ n, seq.a n = -10 * n + 110) ∧
  (Finset.filter (fun n => -18 ≤ seq.a n ∧ seq.a n ≤ 18) (Finset.range 100)).card = 3 := by
  sorry


end arithmetic_sequence_properties_l2828_282805


namespace initial_leaves_count_l2828_282831

/-- The number of leaves Mikey had initially -/
def initial_leaves : ℕ := sorry

/-- The number of leaves that blew away -/
def blown_leaves : ℕ := 244

/-- The number of leaves left -/
def remaining_leaves : ℕ := 112

/-- Theorem stating that the initial number of leaves is 356 -/
theorem initial_leaves_count : initial_leaves = 356 := by
  sorry

end initial_leaves_count_l2828_282831


namespace vertical_dominoes_even_l2828_282870

/-- A grid with even rows colored white and odd rows colored black -/
structure ColoredGrid where
  rows : ℕ
  cols : ℕ

/-- A domino placement on a colored grid -/
structure DominoPlacement (grid : ColoredGrid) where
  horizontal : Finset (ℕ × ℕ)  -- Set of starting positions for horizontal dominoes
  vertical : Finset (ℕ × ℕ)    -- Set of starting positions for vertical dominoes

/-- Predicate to check if a domino placement is valid -/
def is_valid_placement (grid : ColoredGrid) (placement : DominoPlacement grid) : Prop :=
  ∀ (i j : ℕ), i < grid.rows ∧ j < grid.cols →
    ((i, j) ∈ placement.horizontal → j + 1 < grid.cols) ∧
    ((i, j) ∈ placement.vertical → i + 1 < grid.rows)

/-- The main theorem: The number of vertically placed dominoes is even -/
theorem vertical_dominoes_even (grid : ColoredGrid) (placement : DominoPlacement grid)
  (h_valid : is_valid_placement grid placement) :
  Even placement.vertical.card :=
sorry

end vertical_dominoes_even_l2828_282870


namespace length_AG_is_3_sqrt_10_over_2_l2828_282879

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define properties of the triangle
def isRightAngled (t : Triangle) : Prop :=
  -- Right angle at A
  sorry

def hasGivenSides (t : Triangle) : Prop :=
  -- AB = 3 and AC = 3√5
  sorry

-- Define altitude AD
def altitudeAD (t : Triangle) : ℝ × ℝ :=
  sorry

-- Define median BE
def medianBE (t : Triangle) : ℝ × ℝ :=
  sorry

-- Define intersection point G
def intersectionG (t : Triangle) : ℝ × ℝ :=
  sorry

-- Define length of AG
def lengthAG (t : Triangle) : ℝ :=
  sorry

-- Theorem statement
theorem length_AG_is_3_sqrt_10_over_2 (t : Triangle) :
  isRightAngled t → hasGivenSides t →
  lengthAG t = (3 * Real.sqrt 10) / 2 :=
by
  sorry

end length_AG_is_3_sqrt_10_over_2_l2828_282879


namespace smallest_multiple_l2828_282852

theorem smallest_multiple (n : ℕ) : n = 544 ↔ 
  (∃ k : ℕ, n = 17 * k) ∧ 
  (∃ m : ℕ, n = 53 * m + 7) ∧ 
  (∀ x : ℕ, x < n → ¬(∃ k m : ℕ, x = 17 * k ∧ x = 53 * m + 7)) :=
by sorry

end smallest_multiple_l2828_282852


namespace radius_of_combined_lead_spheres_l2828_282878

/-- The radius of a sphere formed by combining the volume of multiple smaller spheres -/
def radiusOfCombinedSphere (n : ℕ) (r : ℝ) : ℝ :=
  ((n : ℝ) * r^3)^(1/3)

/-- Theorem: The radius of a sphere formed by combining 12 spheres of radius 2 cm is ∛96 cm -/
theorem radius_of_combined_lead_spheres :
  radiusOfCombinedSphere 12 2 = (96 : ℝ)^(1/3) := by
  sorry

end radius_of_combined_lead_spheres_l2828_282878


namespace election_win_percentage_l2828_282817

theorem election_win_percentage 
  (total_votes : ℕ) 
  (geoff_percentage : ℚ) 
  (additional_votes_needed : ℕ) 
  (h1 : total_votes = 6000)
  (h2 : geoff_percentage = 1 / 100)
  (h3 : additional_votes_needed = 3000) : 
  ∃ (x : ℚ), x > 51 / 100 ∧ 
    x * total_votes ≤ (geoff_percentage * total_votes + additional_votes_needed) ∧ 
    ∀ (y : ℚ), y < x → y * total_votes < (geoff_percentage * total_votes + additional_votes_needed) :=
by
  sorry

end election_win_percentage_l2828_282817


namespace number_of_ways_to_choose_officials_l2828_282861

-- Define the number of people in the group
def group_size : ℕ := 8

-- Define the number of positions to be filled
def num_positions : ℕ := 3

-- Theorem stating the number of ways to choose the officials
theorem number_of_ways_to_choose_officials :
  (group_size * (group_size - 1) * (group_size - 2)) = 336 := by
  sorry

end number_of_ways_to_choose_officials_l2828_282861
