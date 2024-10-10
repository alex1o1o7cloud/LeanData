import Mathlib

namespace mikey_has_56_jelly_beans_l1513_151394

def napoleon_jelly_beans : ℕ := 34

def sedrich_jelly_beans (napoleon : ℕ) : ℕ := napoleon + 7

def daphne_jelly_beans (sedrich : ℕ) : ℕ := sedrich - 4

def mikey_jelly_beans (napoleon sedrich daphne : ℕ) : ℕ :=
  (3 * (napoleon + sedrich + daphne)) / 6

theorem mikey_has_56_jelly_beans :
  mikey_jelly_beans napoleon_jelly_beans 
    (sedrich_jelly_beans napoleon_jelly_beans) 
    (daphne_jelly_beans (sedrich_jelly_beans napoleon_jelly_beans)) = 56 := by
  sorry

end mikey_has_56_jelly_beans_l1513_151394


namespace enchilada_cost_l1513_151383

theorem enchilada_cost (T E : ℝ) 
  (h1 : 2 * T + 3 * E = 7.80)
  (h2 : 3 * T + 5 * E = 12.70) : 
  E = 2.00 := by
sorry

end enchilada_cost_l1513_151383


namespace max_difference_is_61_l1513_151386

def digits : List Nat := [2, 4, 5, 8]

def two_digit_number (d1 d2 : Nat) : Nat := 10 * d1 + d2

def valid_two_digit_number (n : Nat) : Prop :=
  ∃ d1 d2, d1 ∈ digits ∧ d2 ∈ digits ∧ d1 ≠ d2 ∧ n = two_digit_number d1 d2

theorem max_difference_is_61 :
  ∃ a b, valid_two_digit_number a ∧ valid_two_digit_number b ∧
    (∀ x y, valid_two_digit_number x → valid_two_digit_number y →
      x - y ≤ a - b) ∧
    a - b = 61 := by sorry

end max_difference_is_61_l1513_151386


namespace tempo_original_value_l1513_151353

/-- Given a tempo insured to 5/7 of its original value, with a 3% premium rate
resulting in a $300 premium, prove that the original value of the tempo is $14,000. -/
theorem tempo_original_value (insurance_ratio : ℚ) (premium_rate : ℚ) (premium_amount : ℚ) :
  insurance_ratio = 5 / 7 →
  premium_rate = 3 / 100 →
  premium_amount = 300 →
  premium_rate * (insurance_ratio * 14000) = premium_amount :=
by sorry

end tempo_original_value_l1513_151353


namespace archer_weekly_expenditure_l1513_151361

def archer_expenditure (shots_per_day : ℕ) (days_per_week : ℕ) (recovery_rate : ℚ) 
  (arrow_cost : ℚ) (team_contribution_rate : ℚ) : ℚ :=
  let total_shots := shots_per_day * days_per_week
  let recovered_arrows := total_shots * recovery_rate
  let net_arrows_used := total_shots - recovered_arrows
  let total_cost := net_arrows_used * arrow_cost
  let archer_cost := total_cost * (1 - team_contribution_rate)
  archer_cost

theorem archer_weekly_expenditure :
  archer_expenditure 200 4 (1/5) (11/2) (7/10) = 1056 := by
  sorry

end archer_weekly_expenditure_l1513_151361


namespace ellipse_equation_l1513_151312

/-- The equation √(x² + (y-3)²) + √(x² + (y+3)²) = 10 represents an ellipse. -/
theorem ellipse_equation (x y : ℝ) :
  (Real.sqrt (x^2 + (y - 3)^2) + Real.sqrt (x^2 + (y + 3)^2) = 10) ↔
  (y^2 / 25 + x^2 / 16 = 1) :=
sorry

end ellipse_equation_l1513_151312


namespace even_painted_faces_count_l1513_151371

/-- Represents a rectangular block with given dimensions -/
structure Block where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Counts the number of cubes with an even number of painted faces -/
def countEvenPaintedFaces (b : Block) : ℕ :=
  -- We don't implement the actual counting logic here
  sorry

/-- Theorem stating that a 3x4x2 block has 12 cubes with even number of painted faces -/
theorem even_painted_faces_count (b : Block) 
  (h1 : b.length = 3) 
  (h2 : b.width = 4) 
  (h3 : b.height = 2) : 
  countEvenPaintedFaces b = 12 := by
  sorry

end even_painted_faces_count_l1513_151371


namespace chocolate_bars_distribution_l1513_151389

theorem chocolate_bars_distribution (total_bars : ℕ) (num_people : ℕ) (num_combined : ℕ) :
  total_bars = 12 →
  num_people = 3 →
  num_combined = 2 →
  (total_bars / num_people) * num_combined = 8 :=
by
  sorry

end chocolate_bars_distribution_l1513_151389


namespace trigonometric_identity_l1513_151332

theorem trigonometric_identity (α β : Real) 
  (h1 : Real.tan (α + β) = 2/5)
  (h2 : Real.tan (β - π/4) = 1/4) : 
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 3/22 := by
  sorry

end trigonometric_identity_l1513_151332


namespace six_digit_divisibility_l1513_151303

theorem six_digit_divisibility (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) :
  let n := 100000 * a + 10000 * b + 1000 * c + 100 * a + 10 * b + c
  n % 7 = 0 ∧ n % 11 = 0 ∧ n % 13 = 0 :=
by sorry

end six_digit_divisibility_l1513_151303


namespace denarii_problem_l1513_151344

theorem denarii_problem (x y : ℚ) : 
  x + 7 = 5 * (y - 7) ∧ 
  y + 5 = 7 * (x - 5) → 
  x = 121 / 17 ∧ y = 167 / 17 := by
sorry

end denarii_problem_l1513_151344


namespace milkman_profit_is_90_l1513_151310

/-- Calculates the profit of a milkman selling a milk-water mixture --/
def milkman_profit (total_milk : ℕ) (milk_in_mixture : ℕ) (water_in_mixture : ℕ) (cost_per_liter : ℕ) : ℕ :=
  let total_mixture := milk_in_mixture + water_in_mixture
  let selling_price := total_mixture * cost_per_liter
  let cost_of_milk_used := milk_in_mixture * cost_per_liter
  selling_price - cost_of_milk_used

/-- Proves that the milkman's profit is 90 under given conditions --/
theorem milkman_profit_is_90 :
  milkman_profit 30 20 5 18 = 90 := by
  sorry

#eval milkman_profit 30 20 5 18

end milkman_profit_is_90_l1513_151310


namespace duty_arrangements_l1513_151334

/-- Represents the number of teachers -/
def num_teachers : ℕ := 3

/-- Represents the number of days in a week -/
def num_days : ℕ := 5

/-- Represents the number of teachers required on Monday -/
def teachers_on_monday : ℕ := 2

/-- Represents the number of duty days per teacher -/
def duty_days_per_teacher : ℕ := 2

/-- Theorem stating the number of possible duty arrangements -/
theorem duty_arrangements :
  (num_teachers.choose teachers_on_monday) * ((num_days - 1).choose (num_teachers - 1)) = 12 := by
  sorry

end duty_arrangements_l1513_151334


namespace cos_2alpha_plus_2beta_l1513_151360

theorem cos_2alpha_plus_2beta (α β : ℝ) 
  (h1 : Real.sin (α - β) = 1/3) 
  (h2 : Real.cos α * Real.sin β = 1/6) : 
  Real.cos (2*α + 2*β) = 1/9 := by
  sorry

end cos_2alpha_plus_2beta_l1513_151360


namespace estimation_greater_than_exact_l1513_151391

theorem estimation_greater_than_exact 
  (a b d : ℕ+) 
  (a' b' d' : ℝ)
  (h_a : a' > a ∧ a' < a + 1)
  (h_b : b' < b ∧ b' > b - 1)
  (h_d : d' < d ∧ d' > d - 1) :
  Real.sqrt (a' / b') - Real.sqrt d' > Real.sqrt (a / b) - Real.sqrt d :=
sorry

end estimation_greater_than_exact_l1513_151391


namespace yellow_balls_count_l1513_151301

def total_balls : ℕ := 60
def white_balls : ℕ := 22
def green_balls : ℕ := 10
def red_balls : ℕ := 15
def purple_balls : ℕ := 6
def prob_not_red_or_purple : ℚ := 65/100

theorem yellow_balls_count :
  ∃ (y : ℕ), y = total_balls - (white_balls + green_balls + red_balls + purple_balls) ∧
  (white_balls + green_balls + y : ℚ) / total_balls = prob_not_red_or_purple :=
by sorry

end yellow_balls_count_l1513_151301


namespace arithmetic_sequence_sum_l1513_151350

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  (a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 420) →
  (a 2 + a 10 = 120) := by
  sorry

end arithmetic_sequence_sum_l1513_151350


namespace sum_mod_ten_l1513_151315

theorem sum_mod_ten : (17145 + 17146 + 17147 + 17148 + 17149) % 10 = 5 := by
  sorry

end sum_mod_ten_l1513_151315


namespace pizza_remainder_l1513_151363

theorem pizza_remainder (john_portion emma_fraction : ℚ) : 
  john_portion = 4/5 →
  emma_fraction = 1/4 →
  (1 - john_portion) * (1 - emma_fraction) = 3/20 :=
by sorry

end pizza_remainder_l1513_151363


namespace cube_volume_from_surface_area_l1513_151328

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 600 → s^3 = 1000 := by
  sorry

end cube_volume_from_surface_area_l1513_151328


namespace min_value_of_expression_l1513_151308

theorem min_value_of_expression (x y : ℝ) :
  2 * x^2 + 2 * x * y + y^2 - 2 * x - 1 ≥ -2 ∧
  ∃ (a b : ℝ), 2 * a^2 + 2 * a * b + b^2 - 2 * a - 1 = -2 :=
sorry

end min_value_of_expression_l1513_151308


namespace most_likely_parent_genotypes_l1513_151368

/-- Represents the alleles for rabbit fur type -/
inductive Allele
| H  -- Dominant hairy
| h  -- Recessive hairy
| S  -- Dominant smooth
| s  -- Recessive smooth

/-- Represents the genotype of a rabbit -/
structure Genotype :=
(allele1 : Allele)
(allele2 : Allele)

/-- Determines if a rabbit has hairy fur based on its genotype -/
def isHairy (g : Genotype) : Bool :=
  match g.allele1, g.allele2 with
  | Allele.H, _ => true
  | _, Allele.H => true
  | Allele.h, Allele.h => true
  | _, _ => false

/-- The probability of the hairy allele in the population -/
def p : ℝ := 0.1

/-- Theorem: The most likely genotype combination for parents resulting in all hairy offspring -/
theorem most_likely_parent_genotypes :
  ∃ (parent1 parent2 : Genotype),
    isHairy parent1 ∧
    ¬isHairy parent2 ∧
    (∀ (offspring : Genotype),
      (offspring.allele1 = parent1.allele1 ∨ offspring.allele1 = parent1.allele2) ∧
      (offspring.allele2 = parent2.allele1 ∨ offspring.allele2 = parent2.allele2) →
      isHairy offspring) ∧
    parent1 = ⟨Allele.H, Allele.H⟩ ∧
    parent2 = ⟨Allele.S, Allele.h⟩ :=
by sorry


end most_likely_parent_genotypes_l1513_151368


namespace parabola_equation_l1513_151379

/-- The equation of a parabola with given focus and directrix -/
theorem parabola_equation (x y : ℝ) : 
  let focus : ℝ × ℝ := (4, 4)
  let directrix : ℝ → ℝ → ℝ := λ x y => 4*x + 8*y - 32
  let parabola : ℝ → ℝ → ℝ := λ x y => 64*x^2 - 128*x*y + 64*y^2 - 512*x - 512*y + 1024
  (∀ (p : ℝ × ℝ), p ∈ {p | parabola p.1 p.2 = 0} ↔ 
    (p.1 - focus.1)^2 + (p.2 - focus.2)^2 = (directrix p.1 p.2 / (4 * Real.sqrt 5))^2) :=
by sorry

#check parabola_equation

end parabola_equation_l1513_151379


namespace no_seven_flip_l1513_151357

/-- A function that returns the reverse of the digits of a natural number -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Definition of a k-flip number -/
def isKFlip (k : ℕ) (n : ℕ) : Prop :=
  k * n = reverseDigits n

/-- Theorem: There is no 7-flip integer -/
theorem no_seven_flip : ¬∃ (n : ℕ), n > 0 ∧ isKFlip 7 n := by sorry

end no_seven_flip_l1513_151357


namespace typists_calculation_l1513_151378

/-- The number of typists in the initial group -/
def initial_typists : ℕ := 10

/-- The number of letters typed by the initial group in 20 minutes -/
def initial_letters : ℕ := 20

/-- The time taken by the initial group to type the initial letters (in minutes) -/
def initial_time : ℕ := 20

/-- The number of typists in the second group -/
def second_typists : ℕ := 40

/-- The number of letters typed by the second group in 1 hour -/
def second_letters : ℕ := 240

/-- The time taken by the second group to type the second letters (in minutes) -/
def second_time : ℕ := 60

theorem typists_calculation :
  initial_typists * second_typists * second_time * initial_letters =
  initial_time * second_typists * second_letters * initial_typists :=
by sorry

end typists_calculation_l1513_151378


namespace nested_fraction_evaluation_l1513_151390

theorem nested_fraction_evaluation : 
  (1 : ℚ) / (2 + 1 / (3 + 1 / 4)) = 13 / 30 := by
  sorry

end nested_fraction_evaluation_l1513_151390


namespace nested_fourth_root_equation_solution_l1513_151300

noncomputable def nested_fourth_root (x : ℝ) : ℝ := Real.sqrt (x + Real.sqrt (x + Real.sqrt (x + Real.sqrt x)))

noncomputable def nested_fourth_root_product (x : ℝ) : ℝ := Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x)))

def cubic_equation (y : ℝ) : Prop := y^3 - y^2 - 1 = 0

theorem nested_fourth_root_equation_solution :
  ∃ (x : ℝ), x > 0 ∧ nested_fourth_root x = nested_fourth_root_product x ∧
  ∃ (y : ℝ), cubic_equation y ∧ x = y^3 := by
  sorry

end nested_fourth_root_equation_solution_l1513_151300


namespace three_x_plus_five_y_equals_six_l1513_151341

theorem three_x_plus_five_y_equals_six 
  (h1 : x + 4 * y = 5) 
  (h2 : 5 * x + 6 * y = 7) : 
  3 * x + 5 * y = 6 := by
  sorry

end three_x_plus_five_y_equals_six_l1513_151341


namespace apples_sold_per_day_l1513_151304

/-- Calculates the average number of apples sold per day given the total number of boxes,
    days, and apples per box. -/
def average_apples_per_day (boxes : ℕ) (days : ℕ) (apples_per_box : ℕ) : ℚ :=
  (boxes * apples_per_box : ℚ) / days

/-- Theorem stating that given 12 boxes of apples sold in 4 days,
    with 25 apples per box, the average number of apples sold per day is 75. -/
theorem apples_sold_per_day :
  average_apples_per_day 12 4 25 = 75 := by
  sorry

end apples_sold_per_day_l1513_151304


namespace no_real_solutions_l1513_151305

theorem no_real_solutions : ∀ x : ℝ, (2*x - 4*x + 7)^2 + 1 ≠ -|x^2 - 1| := by
  sorry

end no_real_solutions_l1513_151305


namespace sets_subset_theorem_l1513_151364

-- Define the sets P₁, P₂, Q₁, and Q₂
def P₁ (a : ℝ) : Set ℝ := {x | x^2 + a*x + 1 > 0}
def P₂ (a : ℝ) : Set ℝ := {x | x^2 + a*x + 2 > 0}
def Q₁ (b : ℝ) : Set ℝ := {x | x^2 + x + b > 0}
def Q₂ (b : ℝ) : Set ℝ := {x | x^2 + 2*x + b > 0}

-- State the theorem
theorem sets_subset_theorem :
  (∀ a : ℝ, P₁ a ⊆ P₂ a) ∧ (∃ b : ℝ, Q₁ b ⊆ Q₂ b) := by
  sorry


end sets_subset_theorem_l1513_151364


namespace smallest_x_for_fraction_l1513_151355

theorem smallest_x_for_fraction (x : ℕ) (y : ℤ) : 
  (3 : ℚ) / 4 = y / (256 + x) → x = 0 := by
  sorry

end smallest_x_for_fraction_l1513_151355


namespace hypotenuse_product_squared_l1513_151381

-- Define the triangles
structure RightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  hypotenuse : ℝ
  right_angle : leg1^2 + leg2^2 = hypotenuse^2

-- Define the problem
def triangle_problem (T1 T2 : RightTriangle) : Prop :=
  -- Areas of the triangles
  T1.leg1 * T1.leg2 / 2 = 2 ∧
  T2.leg1 * T2.leg2 / 2 = 3 ∧
  -- Congruent sides
  (T1.leg1 = T2.leg1 ∨ T1.leg1 = T2.leg2) ∧
  (T1.leg2 = T2.leg1 ∨ T1.leg2 = T2.leg2) ∧
  -- Similar triangles
  T1.leg1 / T2.leg1 = T1.leg2 / T2.leg2

-- Theorem statement
theorem hypotenuse_product_squared (T1 T2 : RightTriangle) 
  (h : triangle_problem T1 T2) : 
  (T1.hypotenuse * T2.hypotenuse)^2 = 9216 / 25 := by
  sorry

end hypotenuse_product_squared_l1513_151381


namespace mobius_trip_time_l1513_151340

-- Define the constants from the problem
def distance : ℝ := 143
def speed_with_load : ℝ := 11
def speed_without_load : ℝ := 13
def rest_time_per_stop : ℝ := 0.5
def num_rest_stops : ℕ := 4

-- Define the theorem
theorem mobius_trip_time :
  let time_with_load := distance / speed_with_load
  let time_without_load := distance / speed_without_load
  let total_rest_time := rest_time_per_stop * num_rest_stops
  time_with_load + time_without_load + total_rest_time = 26 := by
sorry


end mobius_trip_time_l1513_151340


namespace largeSum_congruence_l1513_151317

/-- A function that calculates the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The property that a number is congruent to the sum of its digits modulo 9 -/
axiom sum_of_digits_congruence (n : ℕ) : n ≡ sumOfDigits n [MOD 9]

/-- The sum we want to evaluate -/
def largeSum : ℕ := 2 + 55 + 444 + 3333 + 66666 + 777777 + 8888888 + 99999999

/-- Theorem stating that the large sum is congruent to 2 modulo 9 -/
theorem largeSum_congruence : largeSum ≡ 2 [MOD 9] := by sorry

end largeSum_congruence_l1513_151317


namespace largest_number_l1513_151362

theorem largest_number : ∀ (a b c : ℝ), 
  a = 5 ∧ b = 0 ∧ c = -2 → 
  a > b ∧ a > c ∧ a > -Real.sqrt 2 := by
  sorry

end largest_number_l1513_151362


namespace absolute_value_theorem_l1513_151351

theorem absolute_value_theorem (x y : ℝ) (hx : x > 0) :
  |x + 1 - Real.sqrt ((x + y)^2)| = 
    if x + y ≥ 0 then |1 - y| else |2*x + y + 1| := by sorry

end absolute_value_theorem_l1513_151351


namespace smallest_fraction_given_inequalities_l1513_151399

theorem smallest_fraction_given_inequalities :
  ∀ r s : ℤ, 3 * r ≥ 2 * s - 3 → 4 * s ≥ r + 12 → 
  (∃ r' s' : ℤ, r' * s = r * s' ∧ s' > 0 ∧ r' * 2 = s') →
  ∀ r' s' : ℤ, r' * s = r * s' ∧ s' > 0 → r' * 2 ≤ s' :=
by sorry

end smallest_fraction_given_inequalities_l1513_151399


namespace inverse_proportion_problem_l1513_151321

/-- Given that x is inversely proportional to y, prove that if x = 5 when y = 10, then x = -25/2 when y = -4 -/
theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : 5 * 10 = k) :
  -4 * x = k → x = -25/2 := by sorry

end inverse_proportion_problem_l1513_151321


namespace probability_theorem_l1513_151384

/-- The number of days the performance lasts -/
def total_days : ℕ := 8

/-- The number of consecutive days Resident A watches -/
def watch_days : ℕ := 3

/-- The number of days we're interested in (first to fourth day) -/
def interest_days : ℕ := 4

/-- The total number of ways to choose 3 consecutive days out of 8 days -/
def total_choices : ℕ := total_days - watch_days + 1

/-- The number of ways to choose 3 consecutive days within the first 4 days -/
def interest_choices : ℕ := interest_days - watch_days + 1

/-- The probability of choosing 3 consecutive days within the first 4 days out of 8 total days -/
theorem probability_theorem : 
  (interest_choices : ℚ) / total_choices = 1 / 3 := by sorry

end probability_theorem_l1513_151384


namespace overtake_twice_implies_double_speed_l1513_151375

/-- Represents a runner with a constant speed -/
structure Runner where
  speed : ℝ
  speed_pos : speed > 0

/-- Represents the race course -/
structure Course where
  distance_to_stadium : ℝ
  lap_length : ℝ
  total_laps : ℕ
  distance_to_stadium_pos : distance_to_stadium > 0
  lap_length_pos : lap_length > 0
  total_laps_pos : total_laps > 0

/-- Theorem: If a runner overtakes another runner twice in a race with three laps,
    then the faster runner's speed is at least twice the slower runner's speed -/
theorem overtake_twice_implies_double_speed
  (runner1 runner2 : Runner) (course : Course) :
  course.total_laps = 3 →
  (∃ (t1 t2 : ℝ), 0 < t1 ∧ t1 < t2 ∧
    runner1.speed * t1 = runner2.speed * t1 + course.lap_length ∧
    runner1.speed * t2 = runner2.speed * t2 + 2 * course.lap_length) →
  runner1.speed ≥ 2 * runner2.speed :=
by sorry

end overtake_twice_implies_double_speed_l1513_151375


namespace least_subtraction_for_divisibility_l1513_151397

theorem least_subtraction_for_divisibility : ∃! x : ℕ, 
  (∀ y : ℕ, y < x → ¬((50248 - y) % 20 = 0 ∧ (50248 - y) % 37 = 0)) ∧ 
  (50248 - x) % 20 = 0 ∧ 
  (50248 - x) % 37 = 0 :=
by
  -- The proof goes here
  sorry

end least_subtraction_for_divisibility_l1513_151397


namespace tan_neg_seven_pi_sixths_l1513_151335

theorem tan_neg_seven_pi_sixths : Real.tan (-7 * Real.pi / 6) = -Real.sqrt 3 / 3 := by
  sorry

end tan_neg_seven_pi_sixths_l1513_151335


namespace imaginary_part_of_complex_fraction_l1513_151309

def i : ℂ := Complex.I

theorem imaginary_part_of_complex_fraction :
  ((-1 + i) / (2 - i)).im = 1 / 5 := by sorry

end imaginary_part_of_complex_fraction_l1513_151309


namespace parabolas_intersection_l1513_151385

/-- The x-coordinates of the intersection points of two parabolas -/
def intersection_x : Set ℝ :=
  {x : ℝ | 3 * x^2 + 6 * x - 4 = x^2 + 2 * x + 1}

/-- The y-coordinates of the intersection points of two parabolas -/
def intersection_y (x : ℝ) : ℝ :=
  x^2 + 2 * x + 1

/-- The set of intersection points of two parabolas -/
def intersection_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 ∈ intersection_x ∧ p.2 = intersection_y p.1}

theorem parabolas_intersection :
  intersection_points = {(-5, 16), (1/2, 9/4)} :=
by sorry

end parabolas_intersection_l1513_151385


namespace largest_prime_factor_of_expression_l1513_151372

theorem largest_prime_factor_of_expression :
  (Nat.factors (18^3 + 15^4 - 10^5)).maximum? = some 98359 := by
  sorry

end largest_prime_factor_of_expression_l1513_151372


namespace sum_of_multiples_of_6_and_9_is_multiple_of_3_l1513_151349

theorem sum_of_multiples_of_6_and_9_is_multiple_of_3 (c d : ℤ) 
  (hc : ∃ m : ℤ, c = 6 * m) (hd : ∃ n : ℤ, d = 9 * n) : 
  ∃ k : ℤ, c + d = 3 * k := by
sorry

end sum_of_multiples_of_6_and_9_is_multiple_of_3_l1513_151349


namespace solution_exists_l1513_151370

theorem solution_exists : ∃ (x y z : ℝ), 
  (15 + (1/4) * x = 27) ∧ 
  ((1/2) * x - y^2 = 37) ∧ 
  (y^3 + z = 50) ∧ 
  (x = 48) ∧ 
  ((y = Real.sqrt 13 ∧ z = 50 - 13 * Real.sqrt 13) ∨ 
   (y = -Real.sqrt 13 ∧ z = 50 + 13 * Real.sqrt 13)) := by
  sorry

end solution_exists_l1513_151370


namespace call_center_efficiency_l1513_151324

/-- Represents the efficiency and size of call center teams relative to Team B -/
structure CallCenterTeams where
  team_a_efficiency : ℚ  -- Efficiency of Team A relative to Team B
  team_c_efficiency : ℚ  -- Efficiency of Team C relative to Team B
  team_a_size : ℚ        -- Size of Team A relative to Team B
  team_c_size : ℚ        -- Size of Team C relative to Team B

/-- Calculates the fraction of total calls processed by all three teams combined -/
def fraction_of_total_calls (teams : CallCenterTeams) : ℚ :=
  sorry

/-- Theorem stating that the fraction of total calls processed is 19/32 -/
theorem call_center_efficiency (teams : CallCenterTeams) 
  (h1 : teams.team_a_efficiency = 1/5)
  (h2 : teams.team_c_efficiency = 7/8)
  (h3 : teams.team_a_size = 5/8)
  (h4 : teams.team_c_size = 3/4) :
  fraction_of_total_calls teams = 19/32 :=
  sorry

end call_center_efficiency_l1513_151324


namespace nina_total_spent_l1513_151393

/-- The total amount Nina spends on her children's presents -/
def total_spent (toy_price toy_quantity card_price card_quantity shirt_price shirt_quantity : ℕ) : ℕ :=
  toy_price * toy_quantity + card_price * card_quantity + shirt_price * shirt_quantity

/-- Theorem stating that Nina spends $70 in total -/
theorem nina_total_spent :
  total_spent 10 3 5 2 6 5 = 70 := by
  sorry

end nina_total_spent_l1513_151393


namespace parabola_intersection_l1513_151319

theorem parabola_intersection :
  let f (x : ℝ) := 3 * x^2 - 12 * x + 15
  let g (x : ℝ) := 2 * x^2 - 8 * x + 12
  ∀ x y : ℝ, f x = g x ∧ y = f x ↔ (x = 1 ∧ y = 6) ∨ (x = 3 ∧ y = 6) := by
  sorry

end parabola_intersection_l1513_151319


namespace charles_pictures_l1513_151330

theorem charles_pictures (initial_papers : ℕ) (today_pictures : ℕ) (yesterday_before_work : ℕ) (papers_left : ℕ) :
  initial_papers = 20 →
  today_pictures = 6 →
  yesterday_before_work = 6 →
  papers_left = 2 →
  initial_papers - today_pictures - yesterday_before_work - papers_left = 6 := by
  sorry

end charles_pictures_l1513_151330


namespace arithmetic_sequence_equality_l1513_151343

theorem arithmetic_sequence_equality (N : ℕ) : 
  (3 + 4 + 5 + 6 + 7) / 5 = (1993 + 1994 + 1995 + 1996 + 1997) / N → N = 1995 := by
  sorry

end arithmetic_sequence_equality_l1513_151343


namespace triangle_abc_area_l1513_151365

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle with a center and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- The line to which the circles are tangent -/
def line_m : Set Point := sorry

theorem triangle_abc_area :
  let circle_a : Circle := { center := { x := -5, y := 2 }, radius := 2 }
  let circle_b : Circle := { center := { x := 0, y := 3 }, radius := 3 }
  let circle_c : Circle := { center := { x := 7, y := 4 }, radius := 4 }
  let point_a' : Point := sorry
  let point_b' : Point := sorry
  let point_c' : Point := sorry

  -- Circles are tangent to line m
  (point_a' ∈ line_m) ∧
  (point_b' ∈ line_m) ∧
  (point_c' ∈ line_m) →

  -- Circle B is externally tangent to circles A and C
  (circle_b.center.x - circle_a.center.x)^2 + (circle_b.center.y - circle_a.center.y)^2 = (circle_b.radius + circle_a.radius)^2 ∧
  (circle_b.center.x - circle_c.center.x)^2 + (circle_b.center.y - circle_c.center.y)^2 = (circle_b.radius + circle_c.radius)^2 →

  -- B' is between A' and C' on line m
  (point_b'.x > point_a'.x ∧ point_b'.x < point_c'.x) →

  -- Centers A and C are aligned horizontally
  circle_a.center.y = circle_c.center.y →

  -- The area of triangle ABC is 6
  abs ((circle_a.center.x * (circle_b.center.y - circle_c.center.y) +
        circle_b.center.x * (circle_c.center.y - circle_a.center.y) +
        circle_c.center.x * (circle_a.center.y - circle_b.center.y)) / 2) = 6 := by
  sorry

end triangle_abc_area_l1513_151365


namespace car_speed_problem_l1513_151326

theorem car_speed_problem (speed_A : ℝ) (time_A : ℝ) (time_B : ℝ) (distance_ratio : ℝ) :
  speed_A = 70 →
  time_A = 10 →
  time_B = 10 →
  distance_ratio = 2 →
  ∃ speed_B : ℝ, speed_B = 35 ∧ speed_A * time_A = distance_ratio * (speed_B * time_B) :=
by sorry

end car_speed_problem_l1513_151326


namespace square_sum_equals_product_implies_zero_l1513_151374

theorem square_sum_equals_product_implies_zero (x y z : ℤ) :
  x^2 + y^2 + z^2 = 2*x*y*z → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end square_sum_equals_product_implies_zero_l1513_151374


namespace smallest_dual_base_representation_l1513_151380

theorem smallest_dual_base_representation :
  ∃ (a b : ℕ), a > 2 ∧ b > 2 ∧
  (1 * a + 2 * 1 = 7) ∧
  (2 * b + 1 * 1 = 7) ∧
  (∀ (x y : ℕ), x > 2 → y > 2 → 1 * x + 2 * 1 = 2 * y + 1 * 1 → 1 * x + 2 * 1 ≥ 7) :=
by sorry

end smallest_dual_base_representation_l1513_151380


namespace sum_percentage_l1513_151382

theorem sum_percentage (A B : ℝ) : 
  (0.4 * A = 160) → 
  (160 = (2/3) * B) → 
  (0.6 * (A + B) = 384) := by
sorry

end sum_percentage_l1513_151382


namespace max_value_of_f_l1513_151331

/-- A cubic function with a constant term -/
def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + m

/-- The minimum value of f on the interval [1,3] -/
def min_value : ℝ := 2

/-- The interval on which we're considering the function -/
def interval : Set ℝ := Set.Icc 1 3

theorem max_value_of_f (m : ℝ) (h : ∃ x ∈ interval, ∀ y ∈ interval, f m y ≥ f m x ∧ f m x = min_value) :
  ∃ x ∈ interval, ∀ y ∈ interval, f m y ≤ f m x ∧ f m x = 10 :=
sorry

end max_value_of_f_l1513_151331


namespace min_perimeter_rectangle_l1513_151345

/-- Represents a rectangle with integer side lengths where one side is 5 feet longer than the other. -/
structure Rectangle where
  short_side : ℕ
  long_side : ℕ
  constraint : long_side = short_side + 5

/-- Calculates the area of a rectangle. -/
def area (r : Rectangle) : ℕ := r.short_side * r.long_side

/-- Calculates the perimeter of a rectangle. -/
def perimeter (r : Rectangle) : ℕ := 2 * (r.short_side + r.long_side)

/-- Theorem: The rectangle with minimum perimeter satisfying the given conditions has dimensions 23 and 28 feet. -/
theorem min_perimeter_rectangle :
  ∀ r : Rectangle,
    area r ≥ 600 →
    perimeter r ≥ 102 ∧
    (perimeter r = 102 → r.short_side = 23 ∧ r.long_side = 28) :=
by sorry

end min_perimeter_rectangle_l1513_151345


namespace fruit_salad_weight_l1513_151325

theorem fruit_salad_weight (melon_weight berries_weight : ℝ) 
  (h1 : melon_weight = 0.25)
  (h2 : berries_weight = 0.38) : 
  melon_weight + berries_weight = 0.63 := by
  sorry

end fruit_salad_weight_l1513_151325


namespace smallest_possible_student_count_l1513_151320

/-- The smallest possible number of students in a classroom with the given seating arrangement --/
def smallest_student_count : ℕ := 42

/-- The number of rows in the classroom --/
def num_rows : ℕ := 5

/-- Represents the number of students in each of the first four rows --/
def students_per_row : ℕ := 8

theorem smallest_possible_student_count :
  (num_rows - 1) * students_per_row + (students_per_row + 2) = smallest_student_count ∧
  smallest_student_count > 40 ∧
  ∀ n : ℕ, n < smallest_student_count →
    (num_rows - 1) * (n / num_rows) + (n / num_rows + 2) ≠ n ∨ n ≤ 40 :=
by sorry

end smallest_possible_student_count_l1513_151320


namespace intersection_property_l1513_151396

noncomputable section

-- Define the line l
def line_l (α t : ℝ) : ℝ × ℝ := (2 + t * Real.cos α, 1 + t * Real.sin α)

-- Define the curve C in polar coordinates
def curve_C_polar (θ : ℝ) : ℝ := 4 * Real.cos θ

-- Define the curve C in Cartesian coordinates
def curve_C_cartesian (x y : ℝ) : Prop := x^2 + y^2 = 4*x

-- Define the point P
def point_P : ℝ × ℝ := (2, 1)

-- Theorem statement
theorem intersection_property (α : ℝ) (h_α : 0 ≤ α ∧ α < Real.pi) :
  ∃ t₁ t₂ : ℝ, 
    let A := line_l α t₁
    let B := line_l α t₂
    let P := point_P
    curve_C_cartesian A.1 A.2 ∧ 
    curve_C_cartesian B.1 B.2 ∧
    (A.1 - P.1, A.2 - P.2) = 2 • (P.1 - B.1, P.2 - B.2) →
    Real.tan α = Real.sqrt (3/5) ∨ Real.tan α = -Real.sqrt (3/5) := by
  sorry

end

end intersection_property_l1513_151396


namespace special_parallelogram_sides_prove_special_parallelogram_sides_l1513_151337

/-- A parallelogram with specific properties -/
structure SpecialParallelogram where
  -- The perimeter of the parallelogram
  perimeter : ℝ
  -- The measure of the acute angle in degrees
  acuteAngle : ℝ
  -- The ratio in which the diagonal divides the obtuse angle
  diagonalRatio : ℝ × ℝ
  -- The sides of the parallelogram
  sides : ℝ × ℝ × ℝ × ℝ

/-- The theorem stating the properties of the special parallelogram -/
theorem special_parallelogram_sides (p : SpecialParallelogram) :
  p.perimeter = 90 ∧ 
  p.acuteAngle = 60 ∧ 
  p.diagonalRatio = (1, 3) →
  p.sides = (15, 15, 30, 30) := by
  sorry

/-- Proof that the sides of the special parallelogram are 15, 15, 30, and 30 -/
theorem prove_special_parallelogram_sides : 
  ∃ (p : SpecialParallelogram), 
    p.perimeter = 90 ∧ 
    p.acuteAngle = 60 ∧ 
    p.diagonalRatio = (1, 3) ∧ 
    p.sides = (15, 15, 30, 30) := by
  sorry

end special_parallelogram_sides_prove_special_parallelogram_sides_l1513_151337


namespace divisor_sum_representation_l1513_151307

theorem divisor_sum_representation (n : ℕ) :
  ∀ k : ℕ, k ≤ n! → ∃ (S : Finset ℕ), 
    (∀ x ∈ S, x ∣ n!) ∧ 
    S.card ≤ n ∧ 
    k = S.sum id :=
by sorry

end divisor_sum_representation_l1513_151307


namespace permanent_non_technicians_percentage_l1513_151387

structure Factory where
  total_workers : ℕ
  technicians : ℕ
  non_technicians : ℕ
  permanent_technicians : ℕ
  temporary_workers : ℕ

def Factory.valid (f : Factory) : Prop :=
  f.technicians + f.non_technicians = f.total_workers ∧
  f.technicians = f.non_technicians ∧
  f.permanent_technicians = f.technicians / 2 ∧
  f.temporary_workers = f.total_workers / 2

theorem permanent_non_technicians_percentage (f : Factory) 
  (h : f.valid) : 
  (f.non_technicians - (f.temporary_workers - f.permanent_technicians)) / f.non_technicians = 1 / 2 := by
  sorry

end permanent_non_technicians_percentage_l1513_151387


namespace steves_emails_l1513_151339

theorem steves_emails (initial_emails : ℕ) : 
  (initial_emails / 2 : ℚ) * (1 - 0.4) = 120 → initial_emails = 400 :=
by
  sorry

end steves_emails_l1513_151339


namespace abs_two_i_over_one_minus_i_l1513_151367

/-- The absolute value of the complex number 2i / (1-i) is √2 -/
theorem abs_two_i_over_one_minus_i :
  let i : ℂ := Complex.I
  let z : ℂ := 2 * i / (1 - i)
  Complex.abs z = Real.sqrt 2 := by
  sorry

end abs_two_i_over_one_minus_i_l1513_151367


namespace shopkeeper_pricing_l1513_151311

/-- Proves that the original selling price is 800 given the conditions of the problem -/
theorem shopkeeper_pricing (cost_price : ℝ) : 
  (1.25 * cost_price = 800) ∧ (0.8 * cost_price = 512) := by
  sorry

#check shopkeeper_pricing

end shopkeeper_pricing_l1513_151311


namespace polynomial_factor_l1513_151336

theorem polynomial_factor (a : ℚ) : 
  (∀ x : ℚ, (x + 5) ∣ (a * x^4 + 12 * x^2 - 5 * a * x + 42)) → 
  a = -57/100 := by
sorry

end polynomial_factor_l1513_151336


namespace original_number_proof_l1513_151373

theorem original_number_proof (h : 204 / 12.75 = 16) : 
  ∃ x : ℝ, x / 1.275 = 1.6 ∧ x = 2.04 := by
  sorry

end original_number_proof_l1513_151373


namespace interest_rate_calculation_l1513_151323

/-- Given a principal sum and a time period of 8 years, if the simple interest
    is one-fifth of the principal, then the rate of interest per annum is 2.5%. -/
theorem interest_rate_calculation (P : ℝ) (P_pos : P > 0) : 
  P / 5 = P * 8 * (2.5 / 100) := by
  sorry

end interest_rate_calculation_l1513_151323


namespace sqrt_equation_solution_l1513_151352

theorem sqrt_equation_solution (x : ℝ) :
  (Real.sqrt (5 * x - 4) + 15 / Real.sqrt (5 * x - 4) = 8) ↔ (x = 29 / 5 ∨ x = 13 / 5) := by
  sorry

end sqrt_equation_solution_l1513_151352


namespace overlapping_sectors_area_l1513_151369

/-- The area of the shaded region formed by the overlap of two 30° sectors in a circle with radius 10 is equal to the area of a single 30° sector. -/
theorem overlapping_sectors_area (r : ℝ) (angle : ℝ) : 
  r = 10 → angle = 30 * (π / 180) → 
  let sector_area := (angle / (2 * π)) * π * r^2
  let shaded_area := sector_area
  ∀ ε > 0, |shaded_area - sector_area| < ε :=
sorry

end overlapping_sectors_area_l1513_151369


namespace apples_in_basket_l1513_151348

theorem apples_in_basket (initial_apples : ℕ) : 
  let ricki_removed : ℕ := 14
  let samson_removed : ℕ := 2 * ricki_removed
  let apples_left : ℕ := 32
  initial_apples = apples_left + ricki_removed + samson_removed :=
by sorry

end apples_in_basket_l1513_151348


namespace sufficient_condition_implies_m_geq_four_l1513_151306

theorem sufficient_condition_implies_m_geq_four (m : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x < 4 → x < m) → m ≥ 4 := by
  sorry

end sufficient_condition_implies_m_geq_four_l1513_151306


namespace sandcastle_height_difference_l1513_151388

theorem sandcastle_height_difference (miki_height sister_height : ℝ) 
  (h1 : miki_height = 0.83)
  (h2 : sister_height = 0.5) :
  miki_height - sister_height = 0.33 := by
sorry

end sandcastle_height_difference_l1513_151388


namespace eggplant_pounds_l1513_151358

/-- Represents the ingredients and costs for Scott's ratatouille recipe --/
structure Ratatouille where
  zucchini_pounds : ℝ
  zucchini_price : ℝ
  tomato_pounds : ℝ
  tomato_price : ℝ
  onion_pounds : ℝ
  onion_price : ℝ
  basil_pounds : ℝ
  basil_price : ℝ
  quart_yield : ℝ
  quart_price : ℝ
  eggplant_price : ℝ

/-- Calculates the total cost of ingredients excluding eggplants --/
def other_ingredients_cost (r : Ratatouille) : ℝ :=
  r.zucchini_pounds * r.zucchini_price +
  r.tomato_pounds * r.tomato_price +
  r.onion_pounds * r.onion_price +
  r.basil_pounds * r.basil_price

/-- Calculates the total cost of the recipe --/
def total_recipe_cost (r : Ratatouille) : ℝ :=
  r.quart_yield * r.quart_price

/-- Calculates the cost spent on eggplants --/
def eggplant_cost (r : Ratatouille) : ℝ :=
  total_recipe_cost r - other_ingredients_cost r

/-- Theorem stating the amount of eggplants bought --/
theorem eggplant_pounds (r : Ratatouille) 
  (h1 : r.zucchini_pounds = 4)
  (h2 : r.zucchini_price = 2)
  (h3 : r.tomato_pounds = 4)
  (h4 : r.tomato_price = 3.5)
  (h5 : r.onion_pounds = 3)
  (h6 : r.onion_price = 1)
  (h7 : r.basil_pounds = 1)
  (h8 : r.basil_price = 5)
  (h9 : r.quart_yield = 4)
  (h10 : r.quart_price = 10)
  (h11 : r.eggplant_price = 2) :
  eggplant_cost r / r.eggplant_price = 5 := by
  sorry


end eggplant_pounds_l1513_151358


namespace tangent_line_equation_l1513_151316

/-- The curve function -/
def f (x : ℝ) : ℝ := x^3 + 3*x^2 - 5

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x

/-- The point of tangency -/
def p : ℝ × ℝ := (-1, -3)

/-- Theorem: The equation of the tangent line to the curve y = x³ + 3x² - 5
    at the point (-1, -3) is 3x + y + 6 = 0 -/
theorem tangent_line_equation :
  ∀ (x y : ℝ), y = f' p.1 * (x - p.1) + p.2 ↔ 3*x + y + 6 = 0 :=
by sorry

end tangent_line_equation_l1513_151316


namespace absolute_value_integral_l1513_151366

theorem absolute_value_integral : ∫ x in (0:ℝ)..2, |1 - x| = 1 := by
  sorry

end absolute_value_integral_l1513_151366


namespace derivative_f_at_pi_l1513_151398

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / (x^2)

theorem derivative_f_at_pi : 
  deriv f π = -(1 / π^2) := by sorry

end derivative_f_at_pi_l1513_151398


namespace school_workbook_cost_l1513_151376

/-- The total cost for purchasing workbooks -/
def total_cost (num_workbooks : ℕ) (cost_per_workbook : ℚ) : ℚ :=
  num_workbooks * cost_per_workbook

/-- Theorem: The total cost for the school to purchase 400 workbooks, each costing x yuan, is equal to 400x yuan -/
theorem school_workbook_cost (x : ℚ) : 
  total_cost 400 x = 400 * x := by
  sorry

end school_workbook_cost_l1513_151376


namespace ratio_of_arithmetic_sequences_l1513_151318

def arithmetic_sequence_sum (a₁ : ℚ) (d : ℚ) (aₙ : ℚ) : ℚ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem ratio_of_arithmetic_sequences : 
  let seq1_sum := arithmetic_sequence_sum 4 4 48
  let seq2_sum := arithmetic_sequence_sum 2 3 35
  seq1_sum / seq2_sum = 52 / 37 := by
  sorry

end ratio_of_arithmetic_sequences_l1513_151318


namespace integer_solutions_for_mn_squared_equation_l1513_151338

theorem integer_solutions_for_mn_squared_equation : 
  ∀ (m n : ℤ), m * n^2 = 2009 * (n + 1) ↔ (m = 4018 ∧ n = 1) ∨ (m = 0 ∧ n = -1) :=
by sorry

end integer_solutions_for_mn_squared_equation_l1513_151338


namespace copy_pages_theorem_l1513_151327

def cost_per_5_pages : ℚ := 7
def pages_per_5 : ℚ := 5
def total_money : ℚ := 1500  -- in cents

def pages_copied : ℕ := 1071

theorem copy_pages_theorem :
  ⌊(total_money / cost_per_5_pages) * pages_per_5⌋ = pages_copied :=
by sorry

end copy_pages_theorem_l1513_151327


namespace tigers_wins_l1513_151347

def total_games : ℕ := 56
def losses : ℕ := 12

theorem tigers_wins : 
  let ties := losses / 2
  let wins := total_games - (losses + ties)
  wins = 38 := by sorry

end tigers_wins_l1513_151347


namespace boys_camp_science_percentage_l1513_151359

theorem boys_camp_science_percentage (total_boys : ℕ) (school_A_boys : ℕ) (non_science_boys : ℕ) :
  total_boys = 550 →
  school_A_boys = (20 : ℕ) * total_boys / 100 →
  non_science_boys = 77 →
  (((school_A_boys - non_science_boys) : ℚ) / school_A_boys) * 100 = 30 := by
  sorry

end boys_camp_science_percentage_l1513_151359


namespace glove_selection_count_l1513_151333

def num_glove_pairs : ℕ := 6
def num_gloves_to_choose : ℕ := 4
def num_paired_gloves : ℕ := 2

theorem glove_selection_count :
  (num_glove_pairs.choose 1) * ((2 * num_glove_pairs - 2).choose (num_gloves_to_choose - num_paired_gloves) - (num_glove_pairs - 1)) = 240 := by
  sorry

end glove_selection_count_l1513_151333


namespace complement_intersection_theorem_l1513_151342

def I : Set ℕ := {x | 0 < x ∧ x ≤ 6}
def P : Set ℕ := {x | 6 % x = 0}
def Q : Set ℕ := {1, 3, 4, 5}

theorem complement_intersection_theorem :
  (I \ P) ∩ Q = {4, 5} := by sorry

end complement_intersection_theorem_l1513_151342


namespace prob_at_least_one_of_each_l1513_151356

/-- The probability of having a boy or a girl -/
def p_boy_or_girl : ℚ := 1/2

/-- The number of children in the family -/
def num_children : ℕ := 4

/-- The probability of having all boys or all girls -/
def p_all_same : ℚ := 2 * (p_boy_or_girl ^ num_children)

/-- The probability of having at least one boy and one girl -/
def p_at_least_one_of_each : ℚ := 1 - p_all_same

theorem prob_at_least_one_of_each :
  p_at_least_one_of_each = 7/8 :=
sorry

end prob_at_least_one_of_each_l1513_151356


namespace a_45_value_l1513_151329

def a : ℕ → ℤ
  | 0 => 11
  | 1 => 11
  | n + 2 => sorry  -- This will be defined using the recurrence relation

-- Define the recurrence relation
axiom a_rec : ∀ (m n : ℕ), a (m + n) = (1/2) * (a (2*m) + a (2*n)) - (m - n)^2

theorem a_45_value : a 45 = 1991 := by
  sorry

end a_45_value_l1513_151329


namespace inscribed_rectangle_max_area_l1513_151377

theorem inscribed_rectangle_max_area :
  ∀ (x : ℝ) (r l b : ℝ),
  x > 0 ∧
  x^2 - 25*x + 144 = 0 ∧
  r^2 = x ∧
  l = (2/5) * r ∧
  ∃ (ratio : ℝ), ratio^2 - 3*ratio - 10 = 0 ∧ ratio > 0 ∧ l / b = ratio →
  l * b ≤ 0.512 :=
by sorry

end inscribed_rectangle_max_area_l1513_151377


namespace factorization_equality_l1513_151346

theorem factorization_equality (y : ℝ) : 5*y*(y+2) + 8*(y+2) + 15 = (5*y+8)*(y+2) + 15 := by
  sorry

end factorization_equality_l1513_151346


namespace equation_proof_l1513_151314

theorem equation_proof : 578 - 214 = 364 := by sorry

end equation_proof_l1513_151314


namespace arithmetic_sequence_max_sum_l1513_151322

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ+ → ℚ
  is_arithmetic : ∀ n : ℕ+, a (n + 1) - a n = a 2 - a 1
  sum_condition : a 1 + a 3 + a 8 = 99
  fifth_term : a 5 = 31

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ+) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- The theorem to be proven -/
theorem arithmetic_sequence_max_sum (seq : ArithmeticSequence) :
  ∃ k : ℕ+, ∀ n : ℕ+, S seq n ≤ S seq k ∧ k = 20 := by sorry

end arithmetic_sequence_max_sum_l1513_151322


namespace f_properties_l1513_151395

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log x - m * x + m

theorem f_properties (m : ℝ) :
  (∀ x > 0, f m x ≤ 0) →
  (m = 1 ∧
   ∀ a b, 0 < a → a < b →
     (f m b - f m a) / (b - a) < 1 / (a * (a + 1))) :=
by sorry

end f_properties_l1513_151395


namespace arithmetic_sequence_unique_l1513_151313

/-- Represents a sequence of five natural numbers with a constant difference -/
structure ArithmeticSequence :=
  (first : ℕ)
  (diff : ℕ)

/-- Converts a natural number to a string representation -/
def toLetterRepresentation (n : ℕ) : String :=
  match n with
  | 5 => "T"
  | 12 => "EL"
  | 19 => "EK"
  | 26 => "LA"
  | 33 => "SS"
  | _ => ""

/-- The main theorem to be proved -/
theorem arithmetic_sequence_unique :
  ∀ (seq : ArithmeticSequence),
    (seq.first = 5 ∧ seq.diff = 7) ↔
    (toLetterRepresentation seq.first = "T" ∧
     toLetterRepresentation (seq.first + seq.diff) = "EL" ∧
     toLetterRepresentation (seq.first + 2 * seq.diff) = "EK" ∧
     toLetterRepresentation (seq.first + 3 * seq.diff) = "LA" ∧
     toLetterRepresentation (seq.first + 4 * seq.diff) = "SS") :=
by sorry

end arithmetic_sequence_unique_l1513_151313


namespace max_consecutive_set_size_l1513_151354

/-- Sum of digits of a positive integer -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Property: sum of digits is not a multiple of 11 -/
def validNumber (n : ℕ) : Prop :=
  sumOfDigits n % 11 ≠ 0

/-- A set of consecutive positive integers with the given property -/
structure ConsecutiveSet :=
  (start : ℕ)
  (size : ℕ)
  (property : ∀ k, k ∈ Finset.range size → validNumber (start + k))

/-- The theorem to be proved -/
theorem max_consecutive_set_size :
  (∃ S : ConsecutiveSet, S.size = 38) ∧
  (∀ S : ConsecutiveSet, S.size ≤ 38) :=
sorry

end max_consecutive_set_size_l1513_151354


namespace sphere_prism_area_difference_l1513_151302

theorem sphere_prism_area_difference :
  let r : ℝ := 2  -- radius of the sphere
  let a : ℝ := 2  -- base edge length of the prism
  let sphere_surface_area : ℝ := 4 * π * r^2
  let max_prism_lateral_area : ℝ := 16 * Real.sqrt 2
  sphere_surface_area - max_prism_lateral_area = 16 * (π - Real.sqrt 2) := by
sorry


end sphere_prism_area_difference_l1513_151302


namespace quadrilateral_diagonal_length_l1513_151392

theorem quadrilateral_diagonal_length 
  (A B C D O : ℝ × ℝ) 
  (h1 : dist O A = 5)
  (h2 : dist O C = 12)
  (h3 : dist O D = 5)
  (h4 : dist O B = 7)
  (h5 : dist B D = 9) :
  dist A C = 13 := by sorry

end quadrilateral_diagonal_length_l1513_151392
