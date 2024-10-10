import Mathlib

namespace quadratic_equation_solution_l1855_185532

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  x₁ = 5 ∧ x₂ = -1 ∧ 
  x₁^2 - 4*x₁ - 5 = 0 ∧ 
  x₂^2 - 4*x₂ - 5 = 0 := by
  sorry

end quadratic_equation_solution_l1855_185532


namespace marble_ratio_l1855_185528

theorem marble_ratio (total : ℕ) (red : ℕ) (yellow : ℕ) 
  (h_total : total = 85)
  (h_red : red = 14)
  (h_yellow : yellow = 29) :
  (total - red - yellow) / red = 3 := by
sorry

end marble_ratio_l1855_185528


namespace shooting_target_proof_l1855_185587

theorem shooting_target_proof (p q : Prop) : 
  (¬p ∨ ¬q) ↔ (¬(p ∧ q)) :=
sorry

end shooting_target_proof_l1855_185587


namespace first_group_work_days_l1855_185521

/-- Represents the daily work units done by a person -/
@[ext] structure WorkUnit where
  value : ℚ

/-- Represents a group of workers -/
structure WorkGroup where
  men : ℕ
  boys : ℕ

/-- Calculates the total work done by a group in a given number of days -/
def totalWork (g : WorkGroup) (manUnit boyUnit : WorkUnit) (days : ℚ) : ℚ :=
  (g.men : ℚ) * manUnit.value * days + (g.boys : ℚ) * boyUnit.value * days

theorem first_group_work_days : 
  let manUnit : WorkUnit := ⟨2⟩
  let boyUnit : WorkUnit := ⟨1⟩
  let firstGroup : WorkGroup := ⟨12, 16⟩
  let secondGroup : WorkGroup := ⟨13, 24⟩
  let secondGroupDays : ℚ := 4
  totalWork firstGroup manUnit boyUnit 5 = totalWork secondGroup manUnit boyUnit secondGroupDays := by
  sorry

end first_group_work_days_l1855_185521


namespace cost_calculation_l1855_185573

/-- The total cost of buying apples and bananas -/
def total_cost (a b : ℝ) : ℝ := 2 * a + 3 * b

/-- Theorem: The total cost of buying 2 kg of apples at 'a' yuan/kg and 3 kg of bananas at 'b' yuan/kg is (2a + 3b) yuan -/
theorem cost_calculation (a b : ℝ) :
  total_cost a b = 2 * a + 3 * b := by
  sorry

end cost_calculation_l1855_185573


namespace current_age_proof_l1855_185574

theorem current_age_proof (my_age : ℕ) (son_age : ℕ) : 
  (my_age - 9 = 5 * (son_age - 9)) →
  (my_age = 3 * son_age) →
  my_age = 54 := by
  sorry

end current_age_proof_l1855_185574


namespace simplify_A_minus_B_A_minus_B_value_l1855_185584

/-- Given two real numbers a and b, we define A and B as follows -/
def A (a b : ℝ) : ℝ := (a + b)^2 - 3 * b^2

def B (a b : ℝ) : ℝ := 2 * (a + b) * (a - b) - 3 * a * b

/-- Theorem stating that A - B simplifies to -a^2 + 5ab -/
theorem simplify_A_minus_B (a b : ℝ) : A a b - B a b = -a^2 + 5*a*b := by sorry

/-- Theorem stating that if (a-3)^2 + |b-4| = 0, then A - B = 51 -/
theorem A_minus_B_value (a b : ℝ) (h : (a - 3)^2 + |b - 4| = 0) : A a b - B a b = 51 := by sorry

end simplify_A_minus_B_A_minus_B_value_l1855_185584


namespace water_fountain_length_l1855_185504

/-- Given the conditions for building water fountains, prove the length of the fountain built by 20 men in 7 days -/
theorem water_fountain_length 
  (men1 : ℕ) (days1 : ℕ) (men2 : ℕ) (days2 : ℕ) (length2 : ℝ)
  (h1 : men1 = 20)
  (h2 : days1 = 7)
  (h3 : men2 = 35)
  (h4 : days2 = 3)
  (h5 : length2 = 42)
  (h_prop : ∀ (m d : ℕ) (l : ℝ), (m * d : ℝ) / (men2 * days2 : ℝ) = l / length2) :
  let length1 := (men1 * days1 : ℝ) * length2 / (men2 * days2 : ℝ)
  length1 = 56 := by
  sorry

end water_fountain_length_l1855_185504


namespace percentage_sixth_graders_combined_l1855_185575

theorem percentage_sixth_graders_combined (annville_total : ℕ) (cleona_total : ℕ)
  (annville_sixth_percent : ℚ) (cleona_sixth_percent : ℚ) :
  annville_total = 100 →
  cleona_total = 200 →
  annville_sixth_percent = 11 / 100 →
  cleona_sixth_percent = 17 / 100 →
  let annville_sixth := (annville_sixth_percent * annville_total : ℚ).floor
  let cleona_sixth := (cleona_sixth_percent * cleona_total : ℚ).floor
  let total_sixth := annville_sixth + cleona_sixth
  let total_students := annville_total + cleona_total
  (total_sixth : ℚ) / total_students = 15 / 100 :=
by sorry

end percentage_sixth_graders_combined_l1855_185575


namespace fraction_1840s_eq_four_fifteenths_l1855_185549

/-- The number of states admitted between 1840 and 1849 -/
def states_1840s : ℕ := 8

/-- The total number of states in Alice's collection -/
def total_states : ℕ := 30

/-- The fraction of states admitted between 1840 and 1849 out of the first 30 states -/
def fraction_1840s : ℚ := states_1840s / total_states

theorem fraction_1840s_eq_four_fifteenths : fraction_1840s = 4 / 15 := by
  sorry

end fraction_1840s_eq_four_fifteenths_l1855_185549


namespace smallest_x_for_perfect_cube_l1855_185533

def certain_number : ℕ := 1152

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^3

theorem smallest_x_for_perfect_cube :
  ∃! x : ℕ, x > 0 ∧ is_perfect_cube (certain_number * x) ∧
    ∀ y : ℕ, y > 0 ∧ y < x → ¬is_perfect_cube (certain_number * y) ∧
    certain_number * x = 12 * certain_number :=
by
  sorry

end smallest_x_for_perfect_cube_l1855_185533


namespace calculate_expression_l1855_185518

theorem calculate_expression : 3000 * (3000^2999) * 2 = 2 * 3000^3000 := by
  sorry

end calculate_expression_l1855_185518


namespace nancy_folders_l1855_185523

-- Define the problem parameters
def initial_files : ℕ := 80
def deleted_files : ℕ := 31
def files_per_folder : ℕ := 7

-- Define the function to calculate the number of folders
def calculate_folders (initial : ℕ) (deleted : ℕ) (per_folder : ℕ) : ℕ :=
  (initial - deleted) / per_folder

-- State the theorem
theorem nancy_folders :
  calculate_folders initial_files deleted_files files_per_folder = 7 := by
  sorry

end nancy_folders_l1855_185523


namespace shirt_price_satisfies_conditions_l1855_185594

/-- The original price of a shirt, given the following conditions:
  1. Three items: shirt, pants, jacket
  2. Shirt: 25% discount, then additional 25% discount
  3. Pants: 30% discount, original price $50
  4. Jacket: two successive 20% discounts, original price $75
  5. 10% loyalty discount on total after individual discounts
  6. 15% sales tax on final price
  7. Total price paid: $150
-/
def shirt_price : ℝ :=
  let pants_price : ℝ := 50
  let jacket_price : ℝ := 75
  let pants_discount : ℝ := 0.30
  let jacket_discount : ℝ := 0.20
  let loyalty_discount : ℝ := 0.10
  let sales_tax : ℝ := 0.15
  let total_paid : ℝ := 150
  sorry

/-- Theorem stating that the calculated shirt price satisfies the given conditions -/
theorem shirt_price_satisfies_conditions :
  let S := shirt_price
  let pants_discounted := 50 * (1 - 0.30)
  let jacket_discounted := 75 * (1 - 0.20) * (1 - 0.20)
  (S * 0.75 * 0.75 + pants_discounted + jacket_discounted) * (1 - 0.10) * (1 + 0.15) = 150 := by
  sorry

end shirt_price_satisfies_conditions_l1855_185594


namespace arithmetic_sequence_common_difference_l1855_185577

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_incr : ∀ n : ℕ, a n < a (n + 1))
  (h_sum_squares : a 1 ^ 2 + a 10 ^ 2 = 101)
  (h_sum_mid : a 5 + a 6 = 11) :
  ∃ d : ℝ, d = 1 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end arithmetic_sequence_common_difference_l1855_185577


namespace furniture_dealer_profit_l1855_185550

/-- Calculates the gross profit for a furniture dealer selling a desk -/
theorem furniture_dealer_profit
  (purchase_price : ℝ)
  (markup_percentage : ℝ)
  (discount_percentage : ℝ)
  (h1 : purchase_price = 150)
  (h2 : markup_percentage = 0.5)
  (h3 : discount_percentage = 0.2) :
  let selling_price := purchase_price / (1 - markup_percentage)
  let discounted_price := selling_price * (1 - discount_percentage)
  let gross_profit := discounted_price - purchase_price
  gross_profit = 90 := by sorry

end furniture_dealer_profit_l1855_185550


namespace total_toys_is_56_l1855_185576

/-- Given the number of toys Mike has, calculate the total number of toys for Annie, Mike, and Tom. -/
def totalToys (mikeToys : ℕ) : ℕ :=
  let annieToys := 3 * mikeToys
  let tomToys := annieToys + 2
  mikeToys + annieToys + tomToys

/-- Theorem stating that given Mike has 6 toys, the total number of toys for Annie, Mike, and Tom is 56. -/
theorem total_toys_is_56 : totalToys 6 = 56 := by
  sorry

#eval totalToys 6  -- This will evaluate to 56

end total_toys_is_56_l1855_185576


namespace johns_per_sheet_price_l1855_185570

def johns_sitting_fee : ℝ := 125
def sams_sitting_fee : ℝ := 140
def sams_per_sheet : ℝ := 1.50
def num_sheets : ℝ := 12

theorem johns_per_sheet_price (johns_per_sheet : ℝ) : 
  johns_per_sheet * num_sheets + johns_sitting_fee = 
  sams_per_sheet * num_sheets + sams_sitting_fee → 
  johns_per_sheet = 2.75 := by
sorry

end johns_per_sheet_price_l1855_185570


namespace original_girls_count_l1855_185508

/-- Represents the number of boys and girls in a school club. -/
structure ClubMembers where
  boys : ℕ
  girls : ℕ

/-- Defines the conditions of the club membership problem. -/
def ClubProblem (initial : ClubMembers) : Prop :=
  -- Initially, there was one boy for every girl
  initial.boys = initial.girls ∧
  -- After 25 girls leave, there are three boys for each remaining girl
  3 * (initial.girls - 25) = initial.boys ∧
  -- After that, 60 boys leave, and then there are six girls for each remaining boy
  6 * (initial.boys - 60) = initial.girls - 25

/-- Theorem stating that given the conditions, the original number of girls is 67. -/
theorem original_girls_count (initial : ClubMembers) :
  ClubProblem initial → initial.girls = 67 := by
  sorry


end original_girls_count_l1855_185508


namespace unique_number_l1855_185583

def is_valid_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧ 
  n / 100000 = 1 ∧
  (n % 100000) * 10 + 1 = 3 * n

theorem unique_number : ∃! n : ℕ, is_valid_number n :=
  sorry

end unique_number_l1855_185583


namespace equation_solution_l1855_185547

theorem equation_solution (x : ℚ) (h : x ≠ -2) :
  (4 * x / (x + 2) - 2 / (x + 2) = 3 / (x + 2)) → x = 5 / 4 :=
by sorry

end equation_solution_l1855_185547


namespace ratio_of_numbers_l1855_185553

theorem ratio_of_numbers (x y : ℝ) (h1 : x + y = 14) (h2 : y = 3.5) (h3 : x > y) :
  x / y = 3 := by
sorry

end ratio_of_numbers_l1855_185553


namespace floor_squared_sum_four_l1855_185598

theorem floor_squared_sum_four (x y : ℝ) : 
  (Int.floor x)^2 + (Int.floor y)^2 = 4 ↔ 
    ((2 ≤ x ∧ x < 3 ∧ 0 ≤ y ∧ y < 1) ∨
     (0 ≤ x ∧ x < 1 ∧ 2 ≤ y ∧ y < 3) ∨
     (-2 ≤ x ∧ x < -1 ∧ 0 ≤ y ∧ y < 1) ∨
     (0 ≤ x ∧ x < 1 ∧ -2 ≤ y ∧ y < -1)) :=
by sorry

end floor_squared_sum_four_l1855_185598


namespace sprinkles_remaining_l1855_185569

theorem sprinkles_remaining (initial_cans : ℕ) (remaining_cans : ℕ) : 
  initial_cans = 12 →
  remaining_cans = initial_cans / 2 - 3 →
  remaining_cans = 3 := by
sorry

end sprinkles_remaining_l1855_185569


namespace pool_capacity_l1855_185501

theorem pool_capacity (current_water : ℝ) (h1 : current_water > 0) 
  (h2 : current_water + 300 = 0.8 * 1875) 
  (h3 : current_water + 300 = 1.25 * current_water) : 
  1875 = 1875 := by
  sorry

end pool_capacity_l1855_185501


namespace abs_ratio_equal_sqrt_seven_thirds_l1855_185515

theorem abs_ratio_equal_sqrt_seven_thirds (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 + b^2 = 5*a*b) :
  |((a + b) / (a - b))| = Real.sqrt (7/3) := by
  sorry

end abs_ratio_equal_sqrt_seven_thirds_l1855_185515


namespace rebecca_hours_l1855_185546

/-- Given the working hours of Thomas, Toby, and Rebecca, prove that Rebecca worked 56 hours. -/
theorem rebecca_hours :
  ∀ x : ℕ,
  (x + (2*x - 10) + (2*x - 18) = 157) →
  (2*x - 18 = 56) :=
by
  sorry

end rebecca_hours_l1855_185546


namespace thousandth_spirit_enters_on_fourth_floor_l1855_185555

/-- Represents the number of house spirits that enter the elevator on each floor during a complete up-and-down trip -/
def spirits_per_cycle (num_floors : ℕ) : ℕ := 2 * (num_floors - 1) + 2

/-- Calculates the floor on which the nth house spirit enters the elevator -/
def floor_of_nth_spirit (n : ℕ) (num_floors : ℕ) : ℕ :=
  let complete_cycles := (n - 1) / spirits_per_cycle num_floors
  let remaining_spirits := (n - 1) % spirits_per_cycle num_floors
  if remaining_spirits < num_floors then
    remaining_spirits + 1
  else
    2 * num_floors - remaining_spirits - 1

theorem thousandth_spirit_enters_on_fourth_floor :
  floor_of_nth_spirit 1000 7 = 4 := by sorry

end thousandth_spirit_enters_on_fourth_floor_l1855_185555


namespace square_side_estimate_l1855_185517

theorem square_side_estimate (A : ℝ) (h : A = 30) :
  ∃ s : ℝ, s^2 = A ∧ 5 < s ∧ s < 6 := by
  sorry

end square_side_estimate_l1855_185517


namespace negation_of_exp_greater_than_x_l1855_185568

theorem negation_of_exp_greater_than_x :
  (¬ ∀ x : ℝ, Real.exp x > x) ↔ (∃ x : ℝ, Real.exp x ≤ x) := by sorry

end negation_of_exp_greater_than_x_l1855_185568


namespace tank_fill_time_l1855_185552

/-- The time it takes to fill a tank with two pipes and a leak -/
theorem tank_fill_time (pipe1_time pipe2_time : ℝ) (leak_fraction : ℝ) : 
  pipe1_time = 20 →
  pipe2_time = 30 →
  leak_fraction = 1/3 →
  (1 / ((1 / pipe1_time + 1 / pipe2_time) * (1 - leak_fraction))) = 18 := by
  sorry

end tank_fill_time_l1855_185552


namespace credit_card_balance_calculation_l1855_185544

/-- Calculates the final balance on a credit card after two interest applications -/
def final_balance (initial_balance : ℝ) (interest_rate : ℝ) (additional_charge : ℝ) : ℝ :=
  let balance_after_first_interest := initial_balance * (1 + interest_rate)
  let balance_before_second_interest := balance_after_first_interest + additional_charge
  balance_before_second_interest * (1 + interest_rate)

/-- Theorem stating that given the specific conditions, the final balance is $96.00 -/
theorem credit_card_balance_calculation :
  final_balance 50 0.2 20 = 96 := by
  sorry

#eval final_balance 50 0.2 20

end credit_card_balance_calculation_l1855_185544


namespace sum_of_digits_of_7_pow_25_l1855_185588

/-- The sum of the tens digit and the ones digit of 7^25 -/
def sum_of_digits : ℕ :=
  let n : ℕ := 7^25
  (n / 10 % 10) + (n % 10)

/-- Theorem stating that the sum of the tens digit and the ones digit of 7^25 is 7 -/
theorem sum_of_digits_of_7_pow_25 : sum_of_digits = 7 := by
  sorry

end sum_of_digits_of_7_pow_25_l1855_185588


namespace solve_unknown_months_l1855_185557

/-- Represents the grazing arrangement for a milkman -/
structure GrazingArrangement where
  cows : ℕ
  months : ℕ

/-- Represents the rental arrangement for the pasture -/
structure RentalArrangement where
  milkmenCount : ℕ
  totalRent : ℕ
  arrangements : List GrazingArrangement
  unknownMonths : ℕ
  knownRentShare : ℕ

def pasture : RentalArrangement := {
  milkmenCount := 4,
  totalRent := 6500,
  arrangements := [
    { cows := 24, months := 3 },  -- A
    { cows := 10, months := 0 },  -- B (months unknown)
    { cows := 35, months := 4 },  -- C
    { cows := 21, months := 3 }   -- D
  ],
  unknownMonths := 0,  -- We'll solve for this
  knownRentShare := 1440  -- A's share
}

theorem solve_unknown_months (p : RentalArrangement) : p.unknownMonths = 5 :=
  sorry

end solve_unknown_months_l1855_185557


namespace forty_percent_of_number_l1855_185559

theorem forty_percent_of_number (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 15 → (40/100 : ℝ) * N = 180 := by
  sorry

end forty_percent_of_number_l1855_185559


namespace calculation_error_exists_l1855_185522

def numbers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

def is_valid_expression (expr : List (Bool × ℕ)) : Prop :=
  expr.map (Prod.snd) = numbers

def evaluate_expression (expr : List (Bool × ℕ)) : ℤ :=
  expr.foldl (λ acc (op, n) => if op then acc + n else acc - n) 0

theorem calculation_error_exists 
  (expr1 expr2 : List (Bool × ℕ)) 
  (h1 : is_valid_expression expr1)
  (h2 : is_valid_expression expr2)
  (h3 : Odd (evaluate_expression expr1))
  (h4 : Even (evaluate_expression expr2)) :
  ∃ expr, expr ∈ [expr1, expr2] ∧ evaluate_expression expr ≠ 33 ∧ evaluate_expression expr ≠ 32 := by
  sorry

end calculation_error_exists_l1855_185522


namespace horse_speed_calculation_l1855_185593

/-- The speed of the bullet in feet per second -/
def bullet_speed : ℝ := 400

/-- The difference in speed between firing in the same direction as the horse
    and the opposite direction, in feet per second -/
def speed_difference : ℝ := 40

/-- The speed of the horse in feet per second -/
def horse_speed : ℝ := 20

/-- Theorem stating that given the bullet speed and speed difference,
    the horse's speed is 20 feet per second -/
theorem horse_speed_calculation :
  (bullet_speed + horse_speed) - (bullet_speed - horse_speed) = speed_difference :=
by sorry

end horse_speed_calculation_l1855_185593


namespace bill_with_late_charges_l1855_185563

/-- The final bill amount after two late charges -/
def final_bill_amount (original_bill : ℝ) (first_charge_rate : ℝ) (second_charge_rate : ℝ) : ℝ :=
  original_bill * (1 + first_charge_rate) * (1 + second_charge_rate)

/-- Theorem stating the final bill amount after specific late charges -/
theorem bill_with_late_charges :
  final_bill_amount 500 0.02 0.03 = 525.30 := by
  sorry

end bill_with_late_charges_l1855_185563


namespace sin_two_alpha_zero_l1855_185548

open Real

theorem sin_two_alpha_zero (α : ℝ) (f : ℝ → ℝ) (h : f = λ x => sin x - cos x) (h1 : f α = 1) : sin (2 * α) = 0 := by
  sorry

end sin_two_alpha_zero_l1855_185548


namespace average_problem_l1855_185561

theorem average_problem (y : ℝ) (h : (15 + 24 + 32 + y) / 4 = 26) : y = 33 := by
  sorry

end average_problem_l1855_185561


namespace sally_quarters_now_l1855_185509

/-- The number of quarters Sally had initially -/
def initial_quarters : ℕ := 760

/-- The number of quarters Sally spent -/
def spent_quarters : ℕ := 418

/-- Theorem: Sally has 342 quarters now -/
theorem sally_quarters_now : initial_quarters - spent_quarters = 342 := by
  sorry

end sally_quarters_now_l1855_185509


namespace specific_ellipse_area_l1855_185529

/-- An ellipse with given major axis endpoints and a point on its curve -/
structure Ellipse where
  major_axis_end1 : ℝ × ℝ
  major_axis_end2 : ℝ × ℝ
  point_on_curve : ℝ × ℝ

/-- Calculate the area of the ellipse -/
def ellipse_area (e : Ellipse) : ℝ := sorry

/-- Theorem: The area of the specific ellipse is 50π -/
theorem specific_ellipse_area :
  let e : Ellipse := {
    major_axis_end1 := (2, -3),
    major_axis_end2 := (22, -3),
    point_on_curve := (20, 0)
  }
  ellipse_area e = 50 * Real.pi := by sorry

end specific_ellipse_area_l1855_185529


namespace supplement_of_complementary_l1855_185566

/-- Two angles are complementary if their sum is 90 degrees -/
def complementary (α β : ℝ) : Prop := α + β = 90

/-- The supplement of an angle is 180 degrees minus the angle -/
def supplement (θ : ℝ) : ℝ := 180 - θ

/-- 
If two angles α and β are complementary, 
then the supplement of α is 90 degrees greater than β 
-/
theorem supplement_of_complementary (α β : ℝ) 
  (h : complementary α β) : 
  supplement α = β + 90 := by sorry

end supplement_of_complementary_l1855_185566


namespace triangle_angle_proof_l1855_185513

theorem triangle_angle_proof (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → A < π →
  B > 0 → B < π →
  C > 0 → C < π →
  a^2 - b^2 = Real.sqrt 3 * b * c →
  Real.sin C = 2 * Real.sqrt 3 * Real.sin B →
  A = π / 6 := by
sorry

end triangle_angle_proof_l1855_185513


namespace problem_statement_l1855_185582

theorem problem_statement : (1 / (64^(1/3))^9) * 8^6 = 1 := by
  sorry

end problem_statement_l1855_185582


namespace erdos_szekeres_l1855_185535

theorem erdos_szekeres (m n : ℕ) (seq : Fin (m * n + 1) → ℝ) :
  (∃ (subseq : Fin (m + 1) → Fin (m * n + 1)),
    (∀ i j, i < j → seq (subseq i) ≤ seq (subseq j))) ∨
  (∃ (subseq : Fin (n + 1) → Fin (m * n + 1)),
    (∀ i j, i < j → seq (subseq i) ≥ seq (subseq j))) :=
sorry

end erdos_szekeres_l1855_185535


namespace election_total_votes_l1855_185539

-- Define the set of candidates
inductive Candidate : Type
  | Alicia : Candidate
  | Brenda : Candidate
  | Colby : Candidate
  | David : Candidate

-- Define the election
structure Election where
  totalVotes : ℕ
  brendaVotes : ℕ
  brendaPercentage : ℚ

-- Theorem statement
theorem election_total_votes (e : Election) 
  (h1 : e.brendaVotes = 40)
  (h2 : e.brendaPercentage = 1/4) :
  e.totalVotes = 160 := by
  sorry


end election_total_votes_l1855_185539


namespace point_on_curve_iff_f_eq_zero_l1855_185565

-- Define a function f representing the curve
variable (f : ℝ → ℝ → ℝ)

-- Define a point P
variable (x₀ y₀ : ℝ)

-- Theorem stating the necessary and sufficient condition
theorem point_on_curve_iff_f_eq_zero :
  (∃ (x y : ℝ), f x y = 0 ∧ x = x₀ ∧ y = y₀) ↔ f x₀ y₀ = 0 := by sorry

end point_on_curve_iff_f_eq_zero_l1855_185565


namespace cos_phase_shift_l1855_185580

/-- The phase shift of y = cos(2x + π/2) is -π/4 --/
theorem cos_phase_shift : 
  let f := fun x => Real.cos (2 * x + π / 2)
  let phase_shift := fun (B C : ℝ) => -C / B
  phase_shift 2 (π / 2) = -π / 4 := by
sorry

end cos_phase_shift_l1855_185580


namespace f_range_l1855_185514

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x - 5

-- Define the domain
def domain : Set ℝ := { x | -3 ≤ x ∧ x ≤ 0 }

-- Define the range
def range : Set ℝ := { y | ∃ x ∈ domain, f x = y }

-- Theorem statement
theorem f_range : range = { y | -6 ≤ y ∧ y ≤ -2 } := by sorry

end f_range_l1855_185514


namespace galaxy_gym_member_ratio_l1855_185503

theorem galaxy_gym_member_ratio :
  ∀ (f m : ℕ) (f_avg m_avg total_avg : ℝ),
    f_avg = 35 →
    m_avg = 45 →
    total_avg = 40 →
    (f_avg * f + m_avg * m) / (f + m) = total_avg →
    f = m :=
by
  sorry

end galaxy_gym_member_ratio_l1855_185503


namespace pen_count_theorem_l1855_185558

theorem pen_count_theorem : ∀ (red black blue green purple : ℕ),
  red = 8 →
  black = (150 * red) / 100 →
  blue = black + 5 →
  green = blue / 2 →
  purple = 5 →
  red + black + blue + green + purple = 50 :=
by
  sorry

end pen_count_theorem_l1855_185558


namespace inner_probability_is_16_25_l1855_185527

/-- The size of one side of the square checkerboard -/
def boardSize : ℕ := 10

/-- The total number of squares on the checkerboard -/
def totalSquares : ℕ := boardSize * boardSize

/-- The number of squares on the perimeter of the checkerboard -/
def perimeterSquares : ℕ := 4 * boardSize - 4

/-- The number of squares not on the perimeter of the checkerboard -/
def innerSquares : ℕ := totalSquares - perimeterSquares

/-- The probability of choosing a square not on the perimeter -/
def innerProbability : ℚ := innerSquares / totalSquares

theorem inner_probability_is_16_25 : innerProbability = 16 / 25 := by
  sorry

end inner_probability_is_16_25_l1855_185527


namespace vision_assistance_l1855_185560

theorem vision_assistance (total : ℕ) (glasses_percent : ℚ) (contacts_percent : ℚ)
  (h_total : total = 40)
  (h_glasses : glasses_percent = 25 / 100)
  (h_contacts : contacts_percent = 40 / 100) :
  total - (total * glasses_percent).floor - (total * contacts_percent).floor = 14 := by
  sorry

end vision_assistance_l1855_185560


namespace greatest_integer_with_gcd_six_l1855_185534

def is_target (n : ℕ) : Prop :=
  n < 150 ∧ Nat.gcd n 18 = 6

theorem greatest_integer_with_gcd_six :
  ∃ (m : ℕ), is_target m ∧ ∀ (k : ℕ), is_target k → k ≤ m :=
by
  use 144
  sorry

end greatest_integer_with_gcd_six_l1855_185534


namespace phone_not_answered_probability_l1855_185537

theorem phone_not_answered_probability 
  (p1 : ℝ) (p2 : ℝ) (p3 : ℝ) (p4 : ℝ)
  (h1 : p1 = 0.1) (h2 : p2 = 0.3) (h3 : p3 = 0.4) (h4 : p4 = 0.1) :
  (1 - p1) * (1 - p2) * (1 - p3) * (1 - p4) = 0.1 := by
  sorry

end phone_not_answered_probability_l1855_185537


namespace solution_set_for_a_zero_range_of_a_for_solution_exists_l1855_185554

-- Define the functions f and g
def f (x : ℝ) : ℝ := abs (x + 1)
def g (a : ℝ) (x : ℝ) : ℝ := 2 * abs x + a

-- Theorem for part (I)
theorem solution_set_for_a_zero :
  {x : ℝ | f x ≥ g 0 x} = Set.Icc (-1/3) 1 := by sorry

-- Theorem for part (II)
theorem range_of_a_for_solution_exists :
  {a : ℝ | ∃ x, f x ≥ g a x} = Set.Iic 1 := by sorry

end solution_set_for_a_zero_range_of_a_for_solution_exists_l1855_185554


namespace only_baseball_count_l1855_185572

/-- Represents the number of people in different categories in a class --/
structure ClassSports where
  total : ℕ
  both : ℕ
  onlyFootball : ℕ
  neither : ℕ

/-- Theorem stating the number of people who only like baseball --/
theorem only_baseball_count (c : ClassSports) 
  (h1 : c.total = 16)
  (h2 : c.both = 5)
  (h3 : c.onlyFootball = 3)
  (h4 : c.neither = 6) :
  c.total - (c.both + c.onlyFootball + c.neither) = 2 :=
sorry

end only_baseball_count_l1855_185572


namespace f_increasing_on_interval_l1855_185590

open Real

noncomputable def f (x : ℝ) : ℝ := x - 2 * sin x

theorem f_increasing_on_interval :
  ∀ x ∈ Set.Ioo (π/3) (5*π/3), 
    x ∈ Set.Ioo 0 (2*π) → 
    ∀ y ∈ Set.Ioo (π/3) (5*π/3), 
      x < y → f x < f y :=
by sorry

end f_increasing_on_interval_l1855_185590


namespace equation_solution_l1855_185571

theorem equation_solution : 
  {x : ℝ | Real.sqrt ((1 + Real.sqrt 2) ^ x) + Real.sqrt ((1 - Real.sqrt 2) ^ x) = 3} = {2, -2} :=
by sorry

end equation_solution_l1855_185571


namespace largest_divisor_of_n_l1855_185581

theorem largest_divisor_of_n (n : ℕ) (h1 : n > 0) (h2 : 72 ∣ n^2) : ∃ d : ℕ, d > 0 ∧ d ∣ n ∧ ∀ k : ℕ, k > 0 → k ∣ n → k ≤ d := by
  sorry

end largest_divisor_of_n_l1855_185581


namespace easter_egg_hunt_l1855_185592

theorem easter_egg_hunt (baskets : ℕ) (eggs_per_basket : ℕ) (eggs_per_person : ℕ) 
  (shondas_kids : ℕ) (friends : ℕ) (shonda : ℕ) :
  baskets = 15 →
  eggs_per_basket = 12 →
  eggs_per_person = 9 →
  shondas_kids = 2 →
  friends = 10 →
  shonda = 1 →
  (baskets * eggs_per_basket) / eggs_per_person - (shondas_kids + friends + shonda) = 7 :=
by sorry

end easter_egg_hunt_l1855_185592


namespace tangent_line_equations_l1855_185538

/-- The equations of the lines passing through point (1,1) and tangent to the curve y = x³ + 1 -/
theorem tangent_line_equations : 
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b → (x = 1 ∧ y = 1)) ∧ 
    (∃ x₀ : ℝ, 
      (x₀^3 + 1 = m * x₀ + b) ∧ 
      (3 * x₀^2 = m)) ∧
    ((m = 0 ∧ b = 1) ∨ (m = 27/4 ∧ b = -23/4)) :=
by sorry

end tangent_line_equations_l1855_185538


namespace hyperbola_vertex_distance_l1855_185531

/-- The distance between the vertices of the hyperbola x^2/64 - y^2/49 = 1 is 16 -/
theorem hyperbola_vertex_distance : 
  let h : ℝ → ℝ → Prop := λ x y => x^2/64 - y^2/49 = 1
  ∃ x₁ x₂ : ℝ, h x₁ 0 ∧ h x₂ 0 ∧ |x₁ - x₂| = 16 := by
  sorry

end hyperbola_vertex_distance_l1855_185531


namespace moms_dimes_l1855_185536

/-- Given the initial number of dimes, the number of dimes given by dad, and the final number of dimes,
    proves that the number of dimes given by mom is 4. -/
theorem moms_dimes (initial : ℕ) (from_dad : ℕ) (final : ℕ)
  (h1 : initial = 7)
  (h2 : from_dad = 8)
  (h3 : final = 19) :
  final - (initial + from_dad) = 4 := by
  sorry

end moms_dimes_l1855_185536


namespace not_p_or_q_false_implies_p_or_q_l1855_185506

theorem not_p_or_q_false_implies_p_or_q (p q : Prop) :
  ¬(¬(p ∨ q)) → (p ∨ q) := by
  sorry

end not_p_or_q_false_implies_p_or_q_l1855_185506


namespace angle_B_in_triangle_l1855_185525

theorem angle_B_in_triangle (A B C : ℝ) (BC AC : ℝ) (h1 : BC = 6) (h2 : AC = 4) (h3 : Real.sin A = 3/4) :
  B = π/6 := by
  sorry

end angle_B_in_triangle_l1855_185525


namespace system_1_solution_system_2_solution_l1855_185596

-- System 1
theorem system_1_solution :
  ∃ (x y : ℝ), x - y = 3 ∧ x = 3 * y - 1 ∧ x = 5 ∧ y = 2 := by
  sorry

-- System 2
theorem system_2_solution :
  ∃ (x y : ℝ), 2 * x + 3 * y = -1 ∧ 3 * x - 2 * y = 18 ∧ x = 4 ∧ y = -3 := by
  sorry

end system_1_solution_system_2_solution_l1855_185596


namespace house_construction_fraction_l1855_185595

theorem house_construction_fraction (total : ℕ) (additional : ℕ) (remaining : ℕ) 
  (h_total : total = 2000)
  (h_additional : additional = 300)
  (h_remaining : remaining = 500) :
  (total - additional - remaining : ℚ) / total = 3 / 5 :=
sorry

end house_construction_fraction_l1855_185595


namespace y_gets_20_percent_more_than_z_l1855_185551

/-- The problem setup with given conditions -/
def problem_setup (x y z : ℝ) : Prop :=
  x = y * 1.25 ∧  -- x gets 25% more than y
  740 = x + y + z ∧  -- total amount is 740
  z = 200  -- z's share is 200

/-- The theorem to prove -/
theorem y_gets_20_percent_more_than_z 
  (x y z : ℝ) (h : problem_setup x y z) : y = z * 1.2 := by
  sorry


end y_gets_20_percent_more_than_z_l1855_185551


namespace cylinder_volume_constant_l1855_185541

/-- Given a cube with side length 3 and a cylinder with the same surface area,
    if the volume of the cylinder is (M * sqrt(6)) / sqrt(π),
    then M = 9 * sqrt(6) * π -/
theorem cylinder_volume_constant (M : ℝ) : 
  let cube_side : ℝ := 3
  let cube_surface_area : ℝ := 6 * cube_side^2
  ∃ (r h : ℝ),
    (2 * π * r^2 + 2 * π * r * h = cube_surface_area) ∧ 
    (π * r^2 * h = (M * Real.sqrt 6) / Real.sqrt π) →
    M = 9 * Real.sqrt 6 * π :=
by sorry

end cylinder_volume_constant_l1855_185541


namespace luisa_pet_store_distance_l1855_185585

theorem luisa_pet_store_distance (grocery_store_distance : ℝ) (mall_distance : ℝ) (home_distance : ℝ) 
  (miles_per_gallon : ℝ) (cost_per_gallon : ℝ) (total_cost : ℝ) :
  grocery_store_distance = 10 →
  mall_distance = 6 →
  home_distance = 9 →
  miles_per_gallon = 15 →
  cost_per_gallon = 3.5 →
  total_cost = 7 →
  ∃ (pet_store_distance : ℝ),
    pet_store_distance = 5 ∧
    grocery_store_distance + mall_distance + pet_store_distance + home_distance = 
      (total_cost / cost_per_gallon) * miles_per_gallon :=
by sorry

end luisa_pet_store_distance_l1855_185585


namespace rhombus_diagonals_property_inequality_or_equality_l1855_185500

-- Definition for rhombus properties
def diagonals_perpendicular (r : Type) : Prop := sorry
def diagonals_bisect (r : Type) : Prop := sorry

-- Theorem for the first compound proposition
theorem rhombus_diagonals_property :
  ∀ (r : Type), diagonals_perpendicular r ∧ diagonals_bisect r :=
sorry

-- Theorem for the second compound proposition
theorem inequality_or_equality : 2 < 3 ∨ 2 = 3 :=
sorry

end rhombus_diagonals_property_inequality_or_equality_l1855_185500


namespace inscribed_squares_circles_area_difference_l1855_185516

/-- The difference between the sum of areas of squares and circles in an infinite inscribed sequence -/
theorem inscribed_squares_circles_area_difference :
  let square_areas : ℕ → ℝ := λ n => (1 / 2 : ℝ) ^ n
  let circle_areas : ℕ → ℝ := λ n => π / 4 * (1 / 2 : ℝ) ^ n
  (∑' n, square_areas n) - (∑' n, circle_areas n) = 2 - π / 2 := by
  sorry

end inscribed_squares_circles_area_difference_l1855_185516


namespace fraction_reciprocal_l1855_185526

theorem fraction_reciprocal (a b : ℚ) (h : a ≠ b) :
  let c := -(a + b)
  (a + c) / (b + c) = b / a := by
sorry

end fraction_reciprocal_l1855_185526


namespace bird_nest_difference_l1855_185507

theorem bird_nest_difference :
  let num_birds : ℕ := 6
  let num_nests : ℕ := 3
  num_birds - num_nests = 3 := by sorry

end bird_nest_difference_l1855_185507


namespace solution_set_equivalence_range_of_a_l1855_185556

-- Define the function f
def f (a b x : ℝ) := x^2 - a*x + b

-- Part 1
theorem solution_set_equivalence (a b : ℝ) :
  (∀ x, f a b x < 0 ↔ 2 < x ∧ x < 3) →
  (∀ x, b*x^2 - a*x + 1 < 0 ↔ 1/3 < x ∧ x < 1/2) :=
sorry

-- Part 2
theorem range_of_a (a : ℝ) :
  (∀ x, f a (2*a - 3) x ≥ 0) →
  2 ≤ a ∧ a ≤ 6 :=
sorry

end solution_set_equivalence_range_of_a_l1855_185556


namespace hundredths_place_of_seven_twentyfifths_l1855_185589

theorem hundredths_place_of_seven_twentyfifths : ∃ (n : ℕ), (7 : ℚ) / 25 = (n + 28) / 100 ∧ n % 10 = 0 :=
sorry

end hundredths_place_of_seven_twentyfifths_l1855_185589


namespace circle_configuration_theorem_l1855_185540

/-- A configuration of circles as described in the problem -/
structure CircleConfiguration where
  R : ℝ  -- Radius of the semicircle
  r : ℝ  -- Radius of circle O
  r₁ : ℝ  -- Radius of circle O₁
  r₂ : ℝ  -- Radius of circle O₂
  h_positive_R : 0 < R
  h_positive_r : 0 < r
  h_positive_r₁ : 0 < r₁
  h_positive_r₂ : 0 < r₂
  h_tangent_O : r < R  -- O is tangent to the semicircle and its diameter
  h_tangent_O₁ : r₁ < R  -- O₁ is tangent to the semicircle and its diameter
  h_tangent_O₂ : r₂ < R  -- O₂ is tangent to the semicircle and its diameter
  h_tangent_O₁_O : r + r₁ < R  -- O₁ is tangent to O
  h_tangent_O₂_O : r + r₂ < R  -- O₂ is tangent to O

/-- The main theorem to be proved -/
theorem circle_configuration_theorem (c : CircleConfiguration) :
  1 / Real.sqrt c.r₁ + 1 / Real.sqrt c.r₂ = 2 * Real.sqrt 2 / Real.sqrt c.r :=
sorry

end circle_configuration_theorem_l1855_185540


namespace triangle_inequality_l1855_185543

theorem triangle_inequality (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_perimeter : a + b + c = 1) :
  a^2 + b^2 + c^2 + 4*a*b*c < 1/2 := by
  sorry

end triangle_inequality_l1855_185543


namespace distance_difference_l1855_185578

theorem distance_difference (john_distance nina_distance : ℝ) 
  (h1 : john_distance = 0.7)
  (h2 : nina_distance = 0.4) :
  john_distance - nina_distance = 0.3 := by
sorry

end distance_difference_l1855_185578


namespace subjective_not_set_l1855_185512

-- Define what it means for a collection to have objective membership criteria
def has_objective_criteria (C : Type → Prop) : Prop :=
  ∀ (x : Type), (C x ∨ ¬C x) ∧ (∃ (f : Type → Bool), ∀ y, C y ↔ f y = true)

-- Define a set as a collection with objective membership criteria
def is_set (S : Type → Prop) : Prop := has_objective_criteria S

-- Define a collection with subjective criteria (e.g., "good friends")
def subjective_collection (x : Type) : Prop := sorry

-- Theorem: A collection with subjective criteria cannot form a set
theorem subjective_not_set : ¬(is_set subjective_collection) :=
sorry

end subjective_not_set_l1855_185512


namespace tony_temperature_l1855_185586

/-- Represents the temperature change caused by an illness -/
structure Illness where
  temp_change : Int

/-- Calculates the final temperature and its relation to the fever threshold -/
def calculate_temperature (normal_temp : Int) (illnesses : List Illness) (fever_threshold : Int) :
  (Int × Int) :=
  let final_temp := normal_temp + (illnesses.map (·.temp_change)).sum
  let above_threshold := final_temp - fever_threshold
  (final_temp, above_threshold)

theorem tony_temperature :
  let normal_temp := 95
  let illness_a := Illness.mk 10
  let illness_b := Illness.mk 4
  let illness_c := Illness.mk (-2)
  let illnesses := [illness_a, illness_b, illness_c]
  let fever_threshold := 100
  calculate_temperature normal_temp illnesses fever_threshold = (107, 7) := by
  sorry

end tony_temperature_l1855_185586


namespace total_noodles_and_pirates_l1855_185599

theorem total_noodles_and_pirates (pirates : ℕ) (noodle_difference : ℕ) : 
  pirates = 45 → noodle_difference = 7 → pirates + (pirates - noodle_difference) = 83 := by
  sorry

end total_noodles_and_pirates_l1855_185599


namespace trivia_game_score_l1855_185502

/-- Calculates the final score in a trivia game given the specified conditions -/
def calculateFinalScore (firstHalfCorrect secondHalfCorrect : ℕ) 
  (firstHalfOddPoints firstHalfEvenPoints : ℕ)
  (secondHalfOddPoints secondHalfEvenPoints : ℕ)
  (bonusPoints : ℕ) : ℕ :=
  let firstHalfOdd := firstHalfCorrect / 2 + firstHalfCorrect % 2
  let firstHalfEven := firstHalfCorrect / 2
  let secondHalfOdd := secondHalfCorrect / 2 + secondHalfCorrect % 2
  let secondHalfEven := secondHalfCorrect / 2
  let firstHalfMultiplesOf3 := (firstHalfCorrect + 2) / 3
  let secondHalfMultiplesOf3 := (secondHalfCorrect + 1) / 3
  (firstHalfOdd * firstHalfOddPoints + firstHalfEven * firstHalfEvenPoints +
   secondHalfOdd * secondHalfOddPoints + secondHalfEven * secondHalfEvenPoints +
   (firstHalfMultiplesOf3 + secondHalfMultiplesOf3) * bonusPoints)

theorem trivia_game_score :
  calculateFinalScore 10 12 2 4 3 5 5 = 113 := by
  sorry

end trivia_game_score_l1855_185502


namespace translation_result_l1855_185505

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Translates a point horizontally and vertically -/
def translate (p : Point2D) (dx dy : ℝ) : Point2D :=
  { x := p.x + dx, y := p.y + dy }

theorem translation_result :
  let p := Point2D.mk (-3) 2
  let p_translated := translate (translate p 2 0) 0 (-4)
  p_translated = Point2D.mk (-1) (-2) := by
  sorry


end translation_result_l1855_185505


namespace dance_class_boys_count_l1855_185564

theorem dance_class_boys_count :
  ∀ (girls boys : ℕ),
  girls + boys = 35 →
  4 * girls = 3 * boys →
  boys = 20 :=
by
  sorry

end dance_class_boys_count_l1855_185564


namespace willie_stickers_l1855_185591

/-- The number of stickers Willie ends up with after giving some away -/
def stickers_left (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem: Willie ends up with 29 stickers -/
theorem willie_stickers : stickers_left 36 7 = 29 := by
  sorry

end willie_stickers_l1855_185591


namespace cell_phone_price_l1855_185545

/-- The price of a cell phone given the total cost and monthly payments --/
theorem cell_phone_price (total_cost : ℕ) (monthly_payment : ℕ) (num_months : ℕ) 
  (h1 : total_cost = 30)
  (h2 : monthly_payment = 7)
  (h3 : num_months = 4) :
  total_cost - (monthly_payment * num_months) = 2 := by
  sorry

end cell_phone_price_l1855_185545


namespace circle_area_tripled_l1855_185597

theorem circle_area_tripled (r n : ℝ) : 
  (π * (r + n)^2 = 3 * π * r^2) → (r = n * (Real.sqrt 3 - 1) / 2) :=
by sorry

end circle_area_tripled_l1855_185597


namespace two_distinct_roots_l1855_185579

/-- The cubic function f(x) = x^3 - 3x + a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + a

/-- Theorem stating that f(x) has exactly two distinct roots iff a = 2/√3 -/
theorem two_distinct_roots (a : ℝ) (h : a > 0) :
  (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧ ∀ z : ℝ, f a z = 0 → z = x ∨ z = y) ↔
  a = 2 / Real.sqrt 3 := by
  sorry

end two_distinct_roots_l1855_185579


namespace gunther_free_time_l1855_185524

def cleaning_time (vacuum_time dust_time mop_time brush_time_per_cat num_cats : ℕ) : ℕ :=
  vacuum_time + dust_time + mop_time + brush_time_per_cat * num_cats

theorem gunther_free_time 
  (free_time : ℕ) 
  (vacuum_time : ℕ)
  (dust_time : ℕ)
  (mop_time : ℕ)
  (brush_time_per_cat : ℕ)
  (num_cats : ℕ)
  (h1 : free_time = 3 * 60)
  (h2 : vacuum_time = 45)
  (h3 : dust_time = 60)
  (h4 : mop_time = 30)
  (h5 : brush_time_per_cat = 5)
  (h6 : num_cats = 3) :
  free_time - cleaning_time vacuum_time dust_time mop_time brush_time_per_cat num_cats = 30 :=
by sorry

end gunther_free_time_l1855_185524


namespace fraction_denominator_problem_l1855_185520

theorem fraction_denominator_problem (y : ℝ) (x : ℝ) (h1 : y > 0) 
  (h2 : (y / 20) + (3 * y / x) = 0.35 * y) : x = 10 := by
  sorry

end fraction_denominator_problem_l1855_185520


namespace max_value_of_expression_max_value_achievable_l1855_185562

theorem max_value_of_expression (y : ℝ) :
  y^6 / (y^12 + 3*y^9 - 9*y^6 + 27*y^3 + 81) ≤ 1/27 :=
by sorry

theorem max_value_achievable :
  ∃ y : ℝ, y^6 / (y^12 + 3*y^9 - 9*y^6 + 27*y^3 + 81) = 1/27 :=
by sorry

end max_value_of_expression_max_value_achievable_l1855_185562


namespace half_sum_squares_even_odd_l1855_185519

theorem half_sum_squares_even_odd (a b : ℤ) :
  (∃ x y : ℤ, (4 * a^2 + 4 * b^2) / 2 = x^2 + y^2) ∨
  (∃ x y : ℤ, ((2 * a + 1)^2 + (2 * b + 1)^2) / 2 = x^2 + y^2) :=
by sorry

end half_sum_squares_even_odd_l1855_185519


namespace quadratic_inequality_range_l1855_185530

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ a ∈ Set.Ioc (-2) 2 := by
sorry

end quadratic_inequality_range_l1855_185530


namespace base7_addition_l1855_185511

-- Define a function to convert base 7 numbers to natural numbers
def base7ToNat (a b c : Nat) : Nat :=
  a * 7^2 + b * 7 + c

-- Define the two numbers in base 7
def num1 : Nat := base7ToNat 0 2 5
def num2 : Nat := base7ToNat 2 4 6

-- Define the result in base 7
def result : Nat := base7ToNat 3 1 3

-- Theorem statement
theorem base7_addition :
  num1 + num2 = result := by
  sorry

end base7_addition_l1855_185511


namespace set_equality_l1855_185567

theorem set_equality : {x : ℕ | x - 3 < 2} = {0, 1, 2, 3, 4} := by
  sorry

end set_equality_l1855_185567


namespace shortest_path_length_on_tetrahedron_l1855_185510

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  edge_length : ℝ

/-- A path on the surface of a regular tetrahedron -/
structure SurfacePath (t : RegularTetrahedron) where
  length : ℝ
  start_vertex : Fin 4
  end_midpoint : Fin 6

/-- The shortest path on the surface of a regular tetrahedron -/
def shortest_path (t : RegularTetrahedron) : SurfacePath t :=
  sorry

theorem shortest_path_length_on_tetrahedron :
  let t : RegularTetrahedron := ⟨2⟩
  (shortest_path t).length = 3 := by sorry

end shortest_path_length_on_tetrahedron_l1855_185510


namespace sin_90_degrees_l1855_185542

theorem sin_90_degrees : Real.sin (π / 2) = 1 := by
  sorry

end sin_90_degrees_l1855_185542
