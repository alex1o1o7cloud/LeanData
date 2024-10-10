import Mathlib

namespace garden_tulips_percentage_l3633_363326

theorem garden_tulips_percentage :
  ∀ (total_flowers : ℕ) (pink_flowers red_flowers pink_roses red_roses pink_tulips red_tulips lilies : ℕ),
    pink_flowers + red_flowers + lilies = total_flowers →
    pink_roses + pink_tulips = pink_flowers →
    red_roses + red_tulips = red_flowers →
    2 * pink_roses = pink_flowers →
    3 * red_tulips = 2 * red_flowers →
    4 * pink_flowers = 3 * total_flowers →
    10 * lilies = total_flowers →
    100 * (pink_tulips + red_tulips) = 61 * total_flowers :=
by
  sorry

end garden_tulips_percentage_l3633_363326


namespace only_eleven_not_sum_of_two_primes_l3633_363390

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def is_sum_of_two_primes (n : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ n = p + q

theorem only_eleven_not_sum_of_two_primes :
  is_sum_of_two_primes 5 ∧
  is_sum_of_two_primes 7 ∧
  is_sum_of_two_primes 9 ∧
  ¬(is_sum_of_two_primes 11) ∧
  is_sum_of_two_primes 13 :=
by sorry

end only_eleven_not_sum_of_two_primes_l3633_363390


namespace expr_is_monomial_of_degree_3_l3633_363339

/-- A monomial is an algebraic expression consisting of one term. This term can be a constant, a variable, or a product of constants and variables raised to whole number powers. -/
def is_monomial (e : Expr) : Prop := sorry

/-- The degree of a monomial is the sum of the exponents of all its variables. -/
def monomial_degree (e : Expr) : ℕ := sorry

/-- An algebraic expression. -/
inductive Expr
| const : ℚ → Expr
| var : String → Expr
| mul : Expr → Expr → Expr
| pow : Expr → ℕ → Expr

/-- The expression -x^2y -/
def expr : Expr :=
  Expr.mul (Expr.const (-1))
    (Expr.mul (Expr.pow (Expr.var "x") 2) (Expr.var "y"))

theorem expr_is_monomial_of_degree_3 :
  is_monomial expr ∧ monomial_degree expr = 3 := by sorry

end expr_is_monomial_of_degree_3_l3633_363339


namespace theater_ticket_sales_l3633_363358

theorem theater_ticket_sales (total_tickets : ℕ) (adult_price senior_price : ℕ) (senior_tickets : ℕ) : 
  total_tickets = 529 →
  adult_price = 25 →
  senior_price = 15 →
  senior_tickets = 348 →
  (total_tickets - senior_tickets) * adult_price + senior_tickets * senior_price = 9745 :=
by
  sorry

end theater_ticket_sales_l3633_363358


namespace function_properties_l3633_363368

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x / 4 + a / x - Real.log x - 3 / 2

-- Define the derivative of f
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 1 / 4 - a / (x^2) - 1 / x

-- State the theorem
theorem function_properties :
  ∃ (a : ℝ),
    -- The tangent at (1, f(1)) is perpendicular to y = (1/2)x
    f_derivative a 1 = -2 ∧
    -- a = 5/4
    a = 5 / 4 ∧
    -- f(x) is decreasing on (0, 5) and increasing on (5, +∞)
    (∀ x ∈ Set.Ioo 0 5, f_derivative a x < 0) ∧
    (∀ x ∈ Set.Ioi 5, f_derivative a x > 0) ∧
    -- The minimum value of f(x) is -ln(5) at x = 5
    (∀ x > 0, f a x ≥ f a 5) ∧
    f a 5 = -Real.log 5 :=
by
  sorry

end

end function_properties_l3633_363368


namespace interest_rate_proof_l3633_363367

def simple_interest (P r t : ℝ) : ℝ := P * (1 + r * t)

theorem interest_rate_proof (P : ℝ) (h1 : P > 0) :
  ∃ r : ℝ, r > 0 ∧ simple_interest P r 2 = 100 ∧ simple_interest P r 6 = 200 →
  r = 0.5 := by
sorry

end interest_rate_proof_l3633_363367


namespace trajectory_area_is_8_l3633_363392

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cuboid -/
structure Cuboid where
  a : Point3D
  b : Point3D
  c : Point3D
  d : Point3D
  a₁ : Point3D
  b₁ : Point3D
  c₁ : Point3D
  d₁ : Point3D

/-- The length of AB in the cuboid -/
def ab_length : ℝ := 6

/-- The length of BC in the cuboid -/
def bc_length : ℝ := 3

/-- A moving point P on line segment BD -/
def P (t : ℝ) : Point3D :=
  sorry

/-- A moving point Q on line segment A₁C₁ -/
def Q (t : ℝ) : Point3D :=
  sorry

/-- Point M on PQ such that PM = 2MQ -/
def M (t₁ t₂ : ℝ) : Point3D :=
  sorry

/-- The area of the trajectory of point M -/
def trajectory_area (c : Cuboid) : ℝ :=
  sorry

/-- Theorem stating that the area of the trajectory of point M is 8 -/
theorem trajectory_area_is_8 (c : Cuboid) :
  trajectory_area c = 8 :=
  sorry

end trajectory_area_is_8_l3633_363392


namespace box_volume_in_cubic_yards_l3633_363338

-- Define the conversion factor from feet to yards
def feet_to_yards : ℝ := 3

-- Define the volume of the box in cubic feet
def box_volume_cubic_feet : ℝ := 216

-- Theorem to prove
theorem box_volume_in_cubic_yards :
  box_volume_cubic_feet / (feet_to_yards ^ 3) = 8 := by
  sorry


end box_volume_in_cubic_yards_l3633_363338


namespace complex_number_location_l3633_363350

theorem complex_number_location (z : ℂ) (h : (1 + 2*I)/z = I) : 
  z = 2/5 + 1/5*I ∧ z.re > 0 ∧ z.im > 0 := by
  sorry

end complex_number_location_l3633_363350


namespace special_number_composite_l3633_363332

/-- Represents the number formed by n+1 ones, followed by a 2, followed by n+1 ones -/
def special_number (n : ℕ) : ℕ :=
  (10^(n+1) - 1) / 9 * 10^(n+1) + (10^(n+1) - 1) / 9

/-- A number is composite if it has a factor between 1 and itself -/
def is_composite (m : ℕ) : Prop :=
  ∃ k, 1 < k ∧ k < m ∧ m % k = 0

/-- Theorem stating that the special number is composite for all natural numbers n -/
theorem special_number_composite (n : ℕ) : is_composite (special_number n) := by
  sorry


end special_number_composite_l3633_363332


namespace largest_modulus_root_real_part_l3633_363309

theorem largest_modulus_root_real_part 
  (z : ℂ) 
  (hz : 5 * z^4 + 10 * z^3 + 10 * z^2 + 5 * z + 1 = 0) 
  (hmax : ∀ w : ℂ, 5 * w^4 + 10 * w^3 + 10 * w^2 + 5 * w + 1 = 0 → Complex.abs w ≤ Complex.abs z) :
  z.re = -1/2 :=
sorry

end largest_modulus_root_real_part_l3633_363309


namespace intersection_distance_squared_is_396_8_l3633_363357

/-- Two circles in a 2D plane -/
structure TwoCircles where
  /-- Center of the first circle -/
  center1 : ℝ × ℝ
  /-- Radius of the first circle -/
  radius1 : ℝ
  /-- Center of the second circle -/
  center2 : ℝ × ℝ
  /-- Radius of the second circle -/
  radius2 : ℝ

/-- The square of the distance between intersection points of two circles -/
def intersectionPointsDistanceSquared (circles : TwoCircles) : ℝ :=
  sorry

/-- Theorem stating that the square of the distance between intersection points
    of the given circles is 396.8 -/
theorem intersection_distance_squared_is_396_8 :
  let circles : TwoCircles := {
    center1 := (0, 0),
    radius1 := 5,
    center2 := (4, -2),
    radius2 := 3
  }
  intersectionPointsDistanceSquared circles = 396.8 := by
  sorry

end intersection_distance_squared_is_396_8_l3633_363357


namespace difference_of_sums_l3633_363328

def sum_even_up_to (n : ℕ) : ℕ :=
  (n / 2) * (2 + n)

def sum_odd_up_to (n : ℕ) : ℕ :=
  ((n + 1) / 2) ^ 2

theorem difference_of_sums : sum_even_up_to 100 - sum_odd_up_to 29 = 2325 := by
  sorry

end difference_of_sums_l3633_363328


namespace stratified_sampling_correct_l3633_363376

/-- Represents the number of employees in each title category -/
structure EmployeeCount where
  total : ℕ
  senior : ℕ
  intermediate : ℕ
  junior : ℕ

/-- Represents the sample size for each title category -/
structure SampleSize where
  senior : ℕ
  intermediate : ℕ
  junior : ℕ

/-- Calculates the stratified sample size for a given category -/
def stratifiedSampleSize (totalEmployees : ℕ) (categoryCount : ℕ) (sampleSize : ℕ) : ℕ :=
  (sampleSize * categoryCount) / totalEmployees

/-- Theorem: The stratified sampling results in the correct sample sizes -/
theorem stratified_sampling_correct 
  (employees : EmployeeCount) 
  (sample : SampleSize) : 
  employees.total = 150 ∧ 
  employees.senior = 15 ∧ 
  employees.intermediate = 45 ∧ 
  employees.junior = 90 ∧
  sample.senior = stratifiedSampleSize employees.total employees.senior 30 ∧
  sample.intermediate = stratifiedSampleSize employees.total employees.intermediate 30 ∧
  sample.junior = stratifiedSampleSize employees.total employees.junior 30 →
  sample.senior = 3 ∧ sample.intermediate = 9 ∧ sample.junior = 18 := by
  sorry

end stratified_sampling_correct_l3633_363376


namespace zero_not_in_range_of_g_l3633_363305

noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then Int.ceil (1 / (x + 3))
  else Int.floor (1 / (x + 3))

theorem zero_not_in_range_of_g :
  ¬ ∃ (x : ℝ), g x = 0 :=
sorry

end zero_not_in_range_of_g_l3633_363305


namespace dried_fruit_business_theorem_l3633_363355

/-- Represents the daily sales quantity as a function of selling price -/
def sales_quantity (x : ℝ) : ℝ := -80 * x + 560

/-- Represents the daily profit as a function of selling price -/
def daily_profit (x : ℝ) : ℝ := (x - 3) * (sales_quantity x) - 80

theorem dried_fruit_business_theorem 
  (cost_per_bag : ℝ) 
  (other_expenses : ℝ) 
  (min_price max_price : ℝ) :
  cost_per_bag = 3 →
  other_expenses = 80 →
  min_price = 3.5 →
  max_price = 5.5 →
  sales_quantity 3.5 = 280 →
  sales_quantity 5.5 = 120 →
  (∀ x, min_price ≤ x ∧ x ≤ max_price → 
    sales_quantity x = -80 * x + 560) →
  daily_profit 4 = 160 ∧
  (∀ x, min_price ≤ x ∧ x ≤ max_price → 
    daily_profit x ≤ 240) ∧
  daily_profit 5 = 240 := by
sorry

end dried_fruit_business_theorem_l3633_363355


namespace sqrt_relationship_l3633_363327

theorem sqrt_relationship (h : Real.sqrt 22500 = 150) : Real.sqrt 0.0225 = 0.15 := by
  sorry

end sqrt_relationship_l3633_363327


namespace group_age_calculation_l3633_363312

theorem group_age_calculation (total_members : ℕ) (total_average_age : ℚ) (zero_age_members : ℕ) : 
  total_members = 50 →
  total_average_age = 5 →
  zero_age_members = 10 →
  let non_zero_members : ℕ := total_members - zero_age_members
  let total_age : ℚ := total_members * total_average_age
  let non_zero_average_age : ℚ := total_age / non_zero_members
  non_zero_average_age = 25/4 := by
sorry

#eval (25 : ℚ) / 4  -- This should output 6.25

end group_age_calculation_l3633_363312


namespace lindas_wallet_l3633_363388

theorem lindas_wallet (total_amount : ℕ) (total_bills : ℕ) (five_dollar_bills : ℕ) (ten_dollar_bills : ℕ) :
  total_amount = 100 →
  total_bills = 15 →
  five_dollar_bills + ten_dollar_bills = total_bills →
  5 * five_dollar_bills + 10 * ten_dollar_bills = total_amount →
  five_dollar_bills = 10 :=
by sorry

end lindas_wallet_l3633_363388


namespace hot_dogs_remainder_l3633_363363

theorem hot_dogs_remainder : 25197629 % 6 = 5 := by
  sorry

end hot_dogs_remainder_l3633_363363


namespace sum_odd_integers_eq_1040_l3633_363311

/-- The sum of odd integers from 15 to 65, inclusive -/
def sum_odd_integers : ℕ :=
  let first := 15
  let last := 65
  let n := (last - first) / 2 + 1
  n * (first + last) / 2

theorem sum_odd_integers_eq_1040 : sum_odd_integers = 1040 := by
  sorry

end sum_odd_integers_eq_1040_l3633_363311


namespace work_completion_rate_l3633_363349

/-- Given that A can finish a work in 18 days and B can do the same work in half the time taken by A,
    prove that A and B working together can finish 1/6 of the work in a day. -/
theorem work_completion_rate (days_A : ℕ) (days_B : ℕ) : 
  days_A = 18 →
  days_B = days_A / 2 →
  (1 : ℚ) / days_A + (1 : ℚ) / days_B = (1 : ℚ) / 6 := by
  sorry

end work_completion_rate_l3633_363349


namespace cross_number_puzzle_l3633_363318

theorem cross_number_puzzle :
  ∃! (a1 a3 d1 d2 : ℕ),
    (100 ≤ a1 ∧ a1 < 1000) ∧
    (100 ≤ a3 ∧ a3 < 1000) ∧
    (100 ≤ d1 ∧ d1 < 1000) ∧
    (100 ≤ d2 ∧ d2 < 1000) ∧
    (∃ n : ℕ, a1 = n^2) ∧
    (∃ n : ℕ, a3 = n^4) ∧
    (∃ n : ℕ, d1 = 2 * n^5) ∧
    (∃ n : ℕ, d2 = n^3) ∧
    (a1 / 100 = 4) :=
by
  sorry

end cross_number_puzzle_l3633_363318


namespace sum_remainder_five_l3633_363314

theorem sum_remainder_five (n : ℤ) : ((5 - n) + (n + 4)) % 5 = 4 := by
  sorry

end sum_remainder_five_l3633_363314


namespace shortest_return_path_length_l3633_363334

/-- Represents a truncated right circular cone with given properties --/
structure TruncatedCone where
  lowerBaseCircumference : ℝ
  upperBaseCircumference : ℝ
  slopeAngle : ℝ

/-- Represents the tourist's path on the cone --/
def touristPath (cone : TruncatedCone) (upperBaseTravel : ℝ) : ℝ := sorry

/-- Theorem stating the shortest return path length --/
theorem shortest_return_path_length 
  (cone : TruncatedCone) 
  (h1 : cone.lowerBaseCircumference = 10)
  (h2 : cone.upperBaseCircumference = 9)
  (h3 : cone.slopeAngle = π / 3) -- 60 degrees in radians
  (h4 : upperBaseTravel = 3) :
  touristPath cone upperBaseTravel = (5 * Real.sqrt 3) / π :=
sorry

end shortest_return_path_length_l3633_363334


namespace problem_statement_l3633_363375

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) :
  (9 / a + 1 / b ≥ 4) ∧ ((a + 3 / b) * (b + 3 / a) ≥ 12) := by
  sorry

end problem_statement_l3633_363375


namespace school_enrollment_problem_l3633_363351

theorem school_enrollment_problem (total_last_year : ℕ) 
  (xx_increase_rate yy_increase_rate : ℚ)
  (xx_to_yy yy_to_xx : ℕ)
  (xx_dropout_rate yy_dropout_rate : ℚ)
  (net_growth_diff : ℕ) :
  total_last_year = 4000 ∧
  xx_increase_rate = 7/100 ∧
  yy_increase_rate = 3/100 ∧
  xx_to_yy = 10 ∧
  yy_to_xx = 5 ∧
  xx_dropout_rate = 3/100 ∧
  yy_dropout_rate = 1/100 ∧
  net_growth_diff = 40 →
  ∃ (xx_last_year yy_last_year : ℕ),
    xx_last_year + yy_last_year = total_last_year ∧
    (xx_last_year * xx_increase_rate - xx_last_year * xx_dropout_rate - xx_to_yy) -
    (yy_last_year * yy_increase_rate - yy_last_year * yy_dropout_rate + yy_to_xx) = net_growth_diff ∧
    yy_last_year = 1750 :=
by sorry

end school_enrollment_problem_l3633_363351


namespace birds_wings_count_johns_birds_wings_l3633_363343

/-- The number of wings of all birds John can buy with money from his grandparents -/
theorem birds_wings_count (money_per_grandparent : ℕ) (num_grandparents : ℕ) (cost_per_bird : ℕ) (wings_per_bird : ℕ) : ℕ :=
  by
  sorry

/-- Proof that John can buy birds with 20 wings in total -/
theorem johns_birds_wings :
  birds_wings_count 50 4 20 2 = 20 :=
by
  sorry

end birds_wings_count_johns_birds_wings_l3633_363343


namespace mobile_chip_transistor_count_l3633_363353

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem mobile_chip_transistor_count :
  toScientificNotation 15300000000 = ScientificNotation.mk 1.53 10 (by norm_num) :=
sorry

end mobile_chip_transistor_count_l3633_363353


namespace certain_number_proof_l3633_363391

theorem certain_number_proof (x : ℝ) : (1.68 * x) / 6 = 354.2 ↔ x = 1265 := by
  sorry

end certain_number_proof_l3633_363391


namespace container_capacity_container_capacity_proof_l3633_363389

theorem container_capacity : ℝ → Prop :=
  fun capacity =>
    capacity > 0 ∧
    0.4 * capacity + 28 = 0.75 * capacity →
    capacity = 80

-- The proof is omitted
theorem container_capacity_proof : ∃ (capacity : ℝ), container_capacity capacity :=
  sorry

end container_capacity_container_capacity_proof_l3633_363389


namespace total_chips_in_bag_l3633_363331

/-- Represents the number of chips Marnie eats on the first day -/
def first_day_chips : ℕ := 10

/-- Represents the number of chips Marnie eats per day after the first day -/
def daily_chips : ℕ := 10

/-- Represents the total number of days it takes Marnie to finish the bag -/
def total_days : ℕ := 10

/-- Theorem stating that the total number of chips in the bag is 100 -/
theorem total_chips_in_bag : 
  first_day_chips + (total_days - 1) * daily_chips = 100 := by
  sorry

end total_chips_in_bag_l3633_363331


namespace paulines_garden_capacity_l3633_363301

/-- Represents Pauline's garden -/
structure Garden where
  tomato_kinds : ℕ
  tomatoes_per_kind : ℕ
  cucumber_kinds : ℕ
  cucumbers_per_kind : ℕ
  potatoes : ℕ
  rows : ℕ
  spaces_per_row : ℕ

/-- Calculates the number of additional vegetables that can be planted in the garden -/
def additional_vegetables (g : Garden) : ℕ :=
  g.rows * g.spaces_per_row - 
  (g.tomato_kinds * g.tomatoes_per_kind + 
   g.cucumber_kinds * g.cucumbers_per_kind + 
   g.potatoes)

/-- Theorem stating that in Pauline's specific garden, 85 more vegetables can be planted -/
theorem paulines_garden_capacity :
  ∃ (g : Garden), 
    g.tomato_kinds = 3 ∧ 
    g.tomatoes_per_kind = 5 ∧ 
    g.cucumber_kinds = 5 ∧ 
    g.cucumbers_per_kind = 4 ∧ 
    g.potatoes = 30 ∧ 
    g.rows = 10 ∧ 
    g.spaces_per_row = 15 ∧ 
    additional_vegetables g = 85 := by
  sorry

end paulines_garden_capacity_l3633_363301


namespace strawberry_milk_count_total_milk_sum_l3633_363373

/-- The number of students who selected strawberry milk -/
def strawberry_milk_students : ℕ := sorry

/-- The number of students who selected chocolate milk -/
def chocolate_milk_students : ℕ := 2

/-- The number of students who selected regular milk -/
def regular_milk_students : ℕ := 3

/-- The total number of milks taken -/
def total_milks : ℕ := 20

/-- Theorem stating that the number of students who selected strawberry milk is 15 -/
theorem strawberry_milk_count : 
  strawberry_milk_students = 15 :=
by
  sorry

/-- Theorem stating that the total number of milks is the sum of all milk selections -/
theorem total_milk_sum :
  total_milks = chocolate_milk_students + strawberry_milk_students + regular_milk_students :=
by
  sorry

end strawberry_milk_count_total_milk_sum_l3633_363373


namespace age_problem_l3633_363323

theorem age_problem (a b c d : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  d = 3 * a →
  a + b + c + d = 72 →
  b = 12 := by
sorry

end age_problem_l3633_363323


namespace temperature_at_3pm_l3633_363302

-- Define the temperature function
def T (t : ℝ) : ℝ := t^3 - 3*t + 60

-- State the theorem
theorem temperature_at_3pm : T 3 = 78 := by
  sorry

end temperature_at_3pm_l3633_363302


namespace forty_seventh_digit_of_1_17_l3633_363361

/-- The decimal representation of 1/17 -/
def decimal_rep_1_17 : ℚ := 1 / 17

/-- The function that returns the nth digit after the decimal point in a rational number's decimal representation -/
noncomputable def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ := sorry

/-- Theorem: The 47th digit after the decimal point in the decimal representation of 1/17 is 6 -/
theorem forty_seventh_digit_of_1_17 : nth_digit_after_decimal decimal_rep_1_17 47 = 6 := by sorry

end forty_seventh_digit_of_1_17_l3633_363361


namespace forgotten_and_doubled_sum_l3633_363315

/-- The sum of the first n positive integers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem forgotten_and_doubled_sum (luke_sum carissa_sum : ℕ) : 
  sum_first_n 20 = 210 →
  luke_sum = 207 →
  carissa_sum = 225 →
  (sum_first_n 20 - luke_sum) + (carissa_sum - sum_first_n 20) = 18 := by
  sorry

end forgotten_and_doubled_sum_l3633_363315


namespace highlighter_count_l3633_363307

/-- The number of pink highlighters in Kaya's teacher's desk -/
def pink_highlighters : ℕ := 10

/-- The number of yellow highlighters in Kaya's teacher's desk -/
def yellow_highlighters : ℕ := 15

/-- The number of blue highlighters in Kaya's teacher's desk -/
def blue_highlighters : ℕ := 8

/-- The total number of highlighters in Kaya's teacher's desk -/
def total_highlighters : ℕ := pink_highlighters + yellow_highlighters + blue_highlighters

theorem highlighter_count : total_highlighters = 33 := by
  sorry

end highlighter_count_l3633_363307


namespace clive_change_l3633_363352

/-- The amount of money Clive has to spend -/
def budget : ℚ := 10

/-- The number of olives Clive needs -/
def olives_needed : ℕ := 80

/-- The number of olives in each jar -/
def olives_per_jar : ℕ := 20

/-- The cost of one jar of olives -/
def cost_per_jar : ℚ := 3/2

/-- The change Clive will have after buying the required number of olive jars -/
def change : ℚ := budget - (↑(olives_needed / olives_per_jar) * cost_per_jar)

theorem clive_change :
  change = 4 := by sorry

end clive_change_l3633_363352


namespace max_withdrawal_theorem_l3633_363317

/-- Represents the possible transactions -/
inductive Transaction
| withdraw : Transaction
| deposit : Transaction

/-- Represents the bank account -/
structure BankAccount where
  balance : ℕ

/-- Applies a transaction to the bank account -/
def applyTransaction (account : BankAccount) (t : Transaction) : BankAccount :=
  match t with
  | Transaction.withdraw => ⟨if account.balance ≥ 300 then account.balance - 300 else account.balance⟩
  | Transaction.deposit => ⟨account.balance + 198⟩

/-- Checks if a sequence of transactions is valid -/
def isValidSequence (initial : ℕ) (transactions : List Transaction) : Prop :=
  let finalAccount := transactions.foldl applyTransaction ⟨initial⟩
  finalAccount.balance ≥ 0

/-- The maximum amount that can be withdrawn -/
def maxWithdrawal (initial : ℕ) : ℕ :=
  initial - (initial % 6)

/-- Theorem stating the maximum withdrawal amount -/
theorem max_withdrawal_theorem (initial : ℕ) :
  initial = 500 →
  maxWithdrawal initial = 300 ∧
  ∃ (transactions : List Transaction), isValidSequence initial transactions ∧
    (initial - (transactions.foldl applyTransaction ⟨initial⟩).balance = maxWithdrawal initial) :=
by sorry

#check max_withdrawal_theorem

end max_withdrawal_theorem_l3633_363317


namespace unique_solution_is_one_point_five_l3633_363345

/-- Given that (3a+2b)x^2+ax+b=0 is a linear equation in x with a unique solution, prove that x = 1.5 -/
theorem unique_solution_is_one_point_five (a b x : ℝ) :
  ((3*a + 2*b) * x^2 + a*x + b = 0) →  -- The equation
  (∃! x, (3*a + 2*b) * x^2 + a*x + b = 0) →  -- Unique solution exists
  (∀ y, (3*a + 2*b) * y^2 + a*y + b = 0 → y = x) →  -- Linear equation condition
  x = 1.5 := by
sorry

end unique_solution_is_one_point_five_l3633_363345


namespace log_4_64_sqrt_4_l3633_363396

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_4_64_sqrt_4 : log 4 (64 * Real.sqrt 4) = 7/2 := by
  sorry

end log_4_64_sqrt_4_l3633_363396


namespace prob_odd_second_roll_l3633_363330

/-- A fair die with six faces -/
structure Die :=
  (faces : Finset Nat)
  (fair : faces = {1, 2, 3, 4, 5, 6})

/-- The set of odd numbers on a die -/
def oddFaces (d : Die) : Finset Nat :=
  d.faces.filter (λ n => n % 2 = 1)

/-- Probability of an event in a finite sample space -/
def probability (event : Finset α) (sampleSpace : Finset α) : ℚ :=
  (event.card : ℚ) / (sampleSpace.card : ℚ)

theorem prob_odd_second_roll (d : Die) :
  probability (oddFaces d) d.faces = 1/2 := by
  sorry

end prob_odd_second_roll_l3633_363330


namespace last_digit_89_base_4_l3633_363300

def last_digit_base_4 (n : ℕ) : ℕ := n % 4

theorem last_digit_89_base_4 : last_digit_base_4 89 = 1 := by
  sorry

end last_digit_89_base_4_l3633_363300


namespace rectangle_shading_theorem_l3633_363379

theorem rectangle_shading_theorem :
  let r : ℝ := 1/4
  let series_sum : ℝ := r / (1 - r)
  series_sum = 1/3 := by sorry

end rectangle_shading_theorem_l3633_363379


namespace circle_equation_tangent_lines_center_x_range_l3633_363378

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define line l
def line_l (x : ℝ) : ℝ := 2 * x - 4

-- Define the point A
def point_A : ℝ × ℝ := (0, 3)

-- Define the circle C
def circle_C : Circle :=
  { center := (3, 2), radius := 1 }

-- Theorem 1
theorem circle_equation (C : Circle) (h1 : C.center.2 = line_l C.center.1) 
  (h2 : C.center.2 = -C.center.1 + 5) (h3 : C.radius = 1) :
  ∀ x y, (x - C.center.1)^2 + (y - C.center.2)^2 = 1 ↔ (x - 3)^2 + (y - 2)^2 = 1 :=
sorry

-- Theorem 2
theorem tangent_lines (C : Circle) (h : ∀ x y, (x - C.center.1)^2 + (y - C.center.2)^2 = 1 ↔ (x - 3)^2 + (y - 2)^2 = 1) :
  (∀ x, x = 3) ∨ (∀ x y, 3*x + 4*y - 12 = 0) :=
sorry

-- Theorem 3
theorem center_x_range (C : Circle) 
  (h : ∃ M : ℝ × ℝ, (M.1 - C.center.1)^2 + (M.2 - C.center.2)^2 = C.radius^2 ∧ 
                    (M.1 - point_A.1)^2 + (M.2 - point_A.2)^2 = M.1^2 + M.2^2) :
  9/4 ≤ C.center.1 ∧ C.center.1 ≤ 13/4 :=
sorry

end circle_equation_tangent_lines_center_x_range_l3633_363378


namespace projectile_max_height_l3633_363359

/-- The height of a projectile as a function of time -/
def h (t : ℝ) : ℝ := -18 * t^2 + 72 * t + 25

/-- Theorem: The maximum height reached by the projectile is 97 feet -/
theorem projectile_max_height : 
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 97 := by
  sorry

end projectile_max_height_l3633_363359


namespace chess_tournament_boys_l3633_363377

theorem chess_tournament_boys (n : ℕ) (k : ℚ) : 
  n > 2 →  -- There are more than 2 boys
  (6 : ℚ) + n * k = (n + 2) * (n + 1) / 2 →  -- Total points equation
  (∀ m : ℕ, m > 2 ∧ m ≠ n → (6 : ℚ) + m * ((m + 2) * (m + 1) / 2 - 6) / m ≠ (m + 2) * (m + 1) / 2) →  -- n is the only solution > 2
  n = 5 ∨ n = 10 :=
by sorry

end chess_tournament_boys_l3633_363377


namespace orange_juice_revenue_l3633_363382

/-- Represents the number of trees each sister owns -/
def trees : ℕ := 110

/-- Represents the number of oranges Gabriela's trees produce per tree -/
def gabriela_oranges : ℕ := 600

/-- Represents the number of oranges Alba's trees produce per tree -/
def alba_oranges : ℕ := 400

/-- Represents the number of oranges Maricela's trees produce per tree -/
def maricela_oranges : ℕ := 500

/-- Represents the number of oranges needed to make one cup of juice -/
def oranges_per_cup : ℕ := 3

/-- Represents the price of one cup of juice in dollars -/
def price_per_cup : ℕ := 4

/-- Theorem stating that the total revenue from selling orange juice is $220,000 -/
theorem orange_juice_revenue :
  (trees * gabriela_oranges + trees * alba_oranges + trees * maricela_oranges) / oranges_per_cup * price_per_cup = 220000 := by
  sorry

end orange_juice_revenue_l3633_363382


namespace dividend_calculation_l3633_363374

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 17) 
  (h2 : quotient = 9) 
  (h3 : remainder = 5) : 
  divisor * quotient + remainder = 158 := by
sorry

end dividend_calculation_l3633_363374


namespace cakes_sold_l3633_363387

theorem cakes_sold (total : ℕ) (left : ℕ) (sold : ℕ) 
  (h1 : total = 54)
  (h2 : left = 13)
  (h3 : sold = total - left) : sold = 41 := by
  sorry

end cakes_sold_l3633_363387


namespace balloon_arrangements_count_l3633_363397

/-- The number of distinct arrangements of letters in "balloon" -/
def balloon_arrangements : ℕ :=
  Nat.factorial 7 / (Nat.factorial 2 * Nat.factorial 2)

/-- The word "balloon" has 7 letters -/
axiom balloon_length : balloon_arrangements = Nat.factorial 7 / (Nat.factorial 2 * Nat.factorial 2)

/-- Theorem: The number of distinct arrangements of letters in "balloon" is 1260 -/
theorem balloon_arrangements_count : balloon_arrangements = 1260 := by
  sorry


end balloon_arrangements_count_l3633_363397


namespace S_minimized_at_two_l3633_363346

/-- The area S(a) bounded by a line and a parabola -/
noncomputable def S (a : ℝ) : ℝ :=
  (1/6) * ((a^2 - 4*a + 8) ^ (3/2))

/-- The theorem stating that S(a) is minimized when a = 2 -/
theorem S_minimized_at_two :
  ∃ (a : ℝ), 0 ≤ a ∧ a ≤ 6 ∧ ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 6 → S a ≤ S x :=
by
  -- The proof goes here
  sorry

#check S_minimized_at_two

end S_minimized_at_two_l3633_363346


namespace solution_set_equivalence_l3633_363340

-- Define a decreasing function on ℝ
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- Define the set of x that satisfies the inequality
def SolutionSet (f : ℝ → ℝ) : Set ℝ :=
  {x | f (x^2 - 3*x - 3) < f 1}

-- Theorem statement
theorem solution_set_equivalence (f : ℝ → ℝ) (h : DecreasingFunction f) :
  SolutionSet f = {x | x < -1 ∨ x > 4} := by
  sorry

end solution_set_equivalence_l3633_363340


namespace envelope_width_l3633_363383

/-- Given a rectangular envelope with length 4 inches and area 16 square inches, prove its width is 4 inches. -/
theorem envelope_width (length : ℝ) (area : ℝ) (width : ℝ) 
  (h1 : length = 4)
  (h2 : area = 16)
  (h3 : area = length * width) : 
  width = 4 := by
  sorry

end envelope_width_l3633_363383


namespace sum_of_roots_l3633_363324

theorem sum_of_roots (a b : ℝ) : 
  (∀ x : ℝ, (x + a) * (x + b) = x^2 + 4*x + 3) → a + b = 4 := by
  sorry

end sum_of_roots_l3633_363324


namespace triangle_angle_calculation_l3633_363348

theorem triangle_angle_calculation (A B C : ℝ) (a b c : ℝ) :
  a = 1 → b = Real.sqrt 2 → B = π / 4 → A = π / 6 := by
  sorry

end triangle_angle_calculation_l3633_363348


namespace complex_root_magnitude_one_iff_divisible_by_six_l3633_363310

theorem complex_root_magnitude_one_iff_divisible_by_six (n : ℕ) :
  (∃ z : ℂ, z^(n+1) - z^n - 1 = 0 ∧ Complex.abs z = 1) ↔ (∃ k : ℤ, n + 2 = 6 * k) :=
sorry

end complex_root_magnitude_one_iff_divisible_by_six_l3633_363310


namespace infinitely_many_planes_through_collinear_points_l3633_363372

/-- Three distinct points on a line -/
structure ThreeCollinearPoints (V : Type*) [NormedAddCommGroup V] [NormedSpace ℝ V] :=
  (p₁ p₂ p₃ : V)
  (distinct : p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃)
  (collinear : ∃ (t₁ t₂ : ℝ), p₃ - p₁ = t₁ • (p₂ - p₁) ∧ p₂ - p₁ = t₂ • (p₃ - p₁))

/-- A plane passing through three points -/
def Plane (V : Type*) [NormedAddCommGroup V] [NormedSpace ℝ V] (p₁ p₂ p₃ : V) :=
  {x : V | ∃ (a b c : ℝ), x = a • p₁ + b • p₂ + c • p₃ ∧ a + b + c = 1}

/-- Theorem: There are infinitely many planes passing through three collinear points -/
theorem infinitely_many_planes_through_collinear_points
  {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]
  (points : ThreeCollinearPoints V) :
  ∃ (planes : Set (Plane V points.p₁ points.p₂ points.p₃)), Infinite planes :=
sorry

end infinitely_many_planes_through_collinear_points_l3633_363372


namespace repeating_decimal_to_fraction_l3633_363306

theorem repeating_decimal_to_fraction :
  ∃ (n d : ℕ), d ≠ 0 ∧ gcd n d = 1 ∧ (n : ℚ) / d = 0.4 + (36 : ℚ) / 99 :=
by
  sorry

end repeating_decimal_to_fraction_l3633_363306


namespace f_min_max_l3633_363371

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 - 2*x + 3

-- Define the domain
def domain : Set ℝ := {x | -2 ≤ x ∧ x ≤ 1}

-- Theorem statement
theorem f_min_max :
  (∃ (x : ℝ), x ∈ domain ∧ f x = 0) ∧
  (∀ (y : ℝ), y ∈ domain → f y ≥ 0) ∧
  (∃ (z : ℝ), z ∈ domain ∧ f z = 4) ∧
  (∀ (w : ℝ), w ∈ domain → f w ≤ 4) :=
sorry

end f_min_max_l3633_363371


namespace total_wall_length_divisible_by_four_l3633_363369

/-- Represents a partition of a square room into smaller square rooms -/
structure RoomPartition where
  size : ℕ  -- Size of the original square room
  partitions : List (ℕ × ℕ × ℕ)  -- List of (x, y, size) for each smaller room

/-- The sum of all partition wall lengths in a room partition -/
def totalWallLength (rp : RoomPartition) : ℕ :=
  sorry

/-- Theorem: The total wall length of any valid room partition is divisible by 4 -/
theorem total_wall_length_divisible_by_four (rp : RoomPartition) :
  4 ∣ totalWallLength rp :=
sorry

end total_wall_length_divisible_by_four_l3633_363369


namespace right_triangular_prism_volume_l3633_363333

/-- The volume of a right triangular prism with base side lengths 14 and height 8 is 784 cubic units. -/
theorem right_triangular_prism_volume : 
  ∀ (base_side_length height : ℝ), 
    base_side_length = 14 → 
    height = 8 → 
    (1/2 * base_side_length * base_side_length) * height = 784 := by
  sorry

end right_triangular_prism_volume_l3633_363333


namespace rotation_cycle_implies_equilateral_l3633_363360

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A₁ : Point
  A₂ : Point
  A₃ : Point

/-- Rotation of a point around a center by an angle -/
def rotate (center : Point) (angle : ℝ) (p : Point) : Point :=
  sorry

/-- Check if a triangle is equilateral -/
def Triangle.isEquilateral (t : Triangle) : Prop :=
  sorry

/-- The sequence of points A_s -/
def A (s : ℕ) : Point :=
  sorry

/-- The sequence of points P_k -/
def P (k : ℕ) : Point :=
  sorry

/-- The main theorem -/
theorem rotation_cycle_implies_equilateral 
  (t : Triangle) (P₀ : Point) : 
  (P 1986 = P₀) → Triangle.isEquilateral t :=
sorry

end rotation_cycle_implies_equilateral_l3633_363360


namespace quadratic_roots_and_integer_case_l3633_363399

-- Define the quadratic equation
def quadratic_equation (m : ℕ+) (x : ℝ) : ℝ :=
  m * x^2 - (3 * m + 2) * x + 6

-- Theorem statement
theorem quadratic_roots_and_integer_case (m : ℕ+) :
  (∃ x y : ℝ, x ≠ y ∧ quadratic_equation m x = 0 ∧ quadratic_equation m y = 0) ∧
  ((∃ x y : ℤ, x ≠ y ∧ quadratic_equation m x = 0 ∧ quadratic_equation m y = 0) →
   (m = 1 ∨ m = 2)) :=
by sorry

end quadratic_roots_and_integer_case_l3633_363399


namespace math_competition_unattempted_questions_l3633_363336

theorem math_competition_unattempted_questions :
  ∀ (total_questions : ℕ) (correct_points incorrect_points : ℤ) (score : ℕ),
    total_questions = 20 →
    correct_points = 8 →
    incorrect_points = -5 →
    (∃ k : ℕ, score = 13 * k) →
    ∀ (correct attempted : ℕ),
      attempted ≤ total_questions →
      score = correct_points * correct + incorrect_points * (attempted - correct) →
      (total_questions - attempted = 20 ∨ total_questions - attempted = 7) :=
by sorry

end math_competition_unattempted_questions_l3633_363336


namespace triangle_properties_l3633_363370

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) (D : ℝ × ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  b * Real.cos A + a * Real.cos B = 2 * c * Real.cos A →
  D.1 + D.2 = 1 →
  D.1 = 3 * D.2 →
  (D.1 * a + D.2 * b)^2 + (D.1 * c)^2 - 2 * (D.1 * a + D.2 * b) * (D.1 * c) * Real.cos A = 9 →
  A = π / 3 ∧
  (∀ (a' b' c' : ℝ), a' > 0 → b' > 0 → c' > 0 →
    (D.1 * a' + D.2 * b')^2 + (D.1 * c')^2 - 2 * (D.1 * a' + D.2 * b') * (D.1 * c') * Real.cos A = 9 →
    1/2 * b' * c' * Real.sin A ≤ 4 * Real.sqrt 3) :=
by sorry

end triangle_properties_l3633_363370


namespace hexagon_diagonal_length_l3633_363325

/-- The length of a diagonal in a regular hexagon --/
theorem hexagon_diagonal_length (side_length : ℝ) (h : side_length = 12) :
  let diagonal_length := side_length * Real.sqrt 3
  diagonal_length = 12 * Real.sqrt 3 := by
  sorry

end hexagon_diagonal_length_l3633_363325


namespace alice_favorite_number_l3633_363381

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem alice_favorite_number :
  ∃! n : ℕ, 30 < n ∧ n < 150 ∧ n % 11 = 0 ∧ n % 2 ≠ 0 ∧ sum_of_digits n % 5 = 0 ∧ n = 55 := by
  sorry

end alice_favorite_number_l3633_363381


namespace simplify_and_evaluate_expression_l3633_363356

theorem simplify_and_evaluate_expression (a : ℚ) : 
  a = -1/2 → a * (a^4 - a + 1) * (a - 2) = 59/32 := by sorry

end simplify_and_evaluate_expression_l3633_363356


namespace flour_bags_comparison_l3633_363385

theorem flour_bags_comparison (W : ℝ) : 
  (W > 0) →
  let remaining_first := W - W / 3
  let remaining_second := W - 1000 / 3
  (W > 1000 → remaining_second > remaining_first) ∧
  (W = 1000 → remaining_second = remaining_first) ∧
  (W < 1000 → remaining_first > remaining_second) :=
by sorry

end flour_bags_comparison_l3633_363385


namespace polynomial_coefficient_sums_l3633_363321

theorem polynomial_coefficient_sums (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x : ℝ, (x + 2)^8 = a₀ + a₁*(x + 1) + a₂*(x + 1)^2 + a₃*(x + 1)^3 + 
                        a₄*(x + 1)^4 + a₅*(x + 1)^5 + a₆*(x + 1)^6 + 
                        a₇*(x + 1)^7 + a₈*(x + 1)^8) →
  (a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ = 255 ∧
   a₁ + a₃ + a₅ + a₇ = 128) := by
sorry

end polynomial_coefficient_sums_l3633_363321


namespace point_in_second_quadrant_x_range_l3633_363337

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

theorem point_in_second_quadrant_x_range :
  ∀ x : ℝ, second_quadrant ⟨x - 2, x⟩ → 0 < x ∧ x < 2 := by
  sorry

end point_in_second_quadrant_x_range_l3633_363337


namespace geometric_sequence_property_l3633_363344

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

/-- Given a geometric sequence a_n where a_6 + a_8 = 4, prove that a_8(a_4 + 2a_6 + a_8) = 16 -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
    (h_geometric : IsGeometricSequence a) 
    (h_sum : a 6 + a 8 = 4) : 
  a 8 * (a 4 + 2 * a 6 + a 8) = 16 := by
  sorry

end geometric_sequence_property_l3633_363344


namespace unique_solution_l3633_363393

/-- A triplet of natural numbers (a, b, c) where b and c are two-digit numbers. -/
structure Triplet where
  a : ℕ
  b : ℕ
  c : ℕ
  b_twodigit : 10 ≤ b ∧ b ≤ 99
  c_twodigit : 10 ≤ c ∧ c ≤ 99

/-- The property that a triplet (a, b, c) satisfies the equation 10^4*a + 100*b + c = (a + b + c)^3. -/
def satisfies_equation (t : Triplet) : Prop :=
  10^4 * t.a + 100 * t.b + t.c = (t.a + t.b + t.c)^3

/-- Theorem stating that (9, 11, 25) is the only triplet satisfying the equation. -/
theorem unique_solution :
  ∃! t : Triplet, satisfies_equation t ∧ t.a = 9 ∧ t.b = 11 ∧ t.c = 25 :=
sorry

end unique_solution_l3633_363393


namespace fried_chicken_cost_is_12_l3633_363386

/-- Calculates the cost of a fried chicken bucket given budget information and beef purchase details. -/
def fried_chicken_cost (total_budget : ℕ) (amount_left : ℕ) (beef_quantity : ℕ) (beef_price : ℕ) : ℕ :=
  total_budget - amount_left - beef_quantity * beef_price

/-- Proves that the cost of the fried chicken bucket is $12 given the problem conditions. -/
theorem fried_chicken_cost_is_12 :
  fried_chicken_cost 80 53 5 3 = 12 := by
  sorry

end fried_chicken_cost_is_12_l3633_363386


namespace salary_increase_comparison_l3633_363313

theorem salary_increase_comparison (initial_salary : ℝ) (h : initial_salary > 0) :
  let first_worker_new_salary := 2 * initial_salary
  let second_worker_new_salary := 1.5 * initial_salary
  (first_worker_new_salary - second_worker_new_salary) / second_worker_new_salary = 1 / 3 := by
sorry

end salary_increase_comparison_l3633_363313


namespace total_cars_theorem_l3633_363394

/-- Calculates the total number of non-defective cars produced by two factories over a week --/
def total_non_defective_cars (factory_a_monday : ℕ) : ℕ :=
  let factory_a_production := [
    factory_a_monday,
    (factory_a_monday * 2 * 95) / 100,  -- Tuesday with 5% defect rate
    factory_a_monday * 4,
    factory_a_monday * 8,
    factory_a_monday * 16
  ]
  let factory_b_production := [
    factory_a_monday * 2,
    factory_a_monday * 4,
    factory_a_monday * 8,
    (factory_a_monday * 16 * 97) / 100,  -- Thursday with 3% defect rate
    factory_a_monday * 32
  ]
  (factory_a_production.sum + factory_b_production.sum)

/-- Theorem stating the total number of non-defective cars produced --/
theorem total_cars_theorem : total_non_defective_cars 60 = 5545 := by
  sorry


end total_cars_theorem_l3633_363394


namespace inequality_proof_l3633_363384

theorem inequality_proof (a b c : ℝ) :
  (a^2 + 1) * (b^2 + 1) * (c^2 + 1) - (a*b + b*c + c*a - 1)^2 ≥ 0 := by
  sorry

end inequality_proof_l3633_363384


namespace base_7_to_base_10_l3633_363316

-- Define the base-7 number 435₇
def base_7_435 : ℕ := 4 * 7^2 + 3 * 7 + 5

-- Define the function to convert a three-digit base-10 number to its digits
def to_digits (n : ℕ) : ℕ × ℕ × ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  (hundreds, tens, ones)

-- Theorem statement
theorem base_7_to_base_10 :
  ∃ (c d : ℕ), c < 10 ∧ d < 10 ∧ base_7_435 = 300 + 10 * c + d →
  (c * d : ℚ) / 18 = 2 / 9 := by sorry

end base_7_to_base_10_l3633_363316


namespace oscar_elmer_difference_l3633_363303

-- Define the given constants
def elmer_strides_per_gap : ℕ := 44
def oscar_leaps_per_gap : ℕ := 12
def total_poles : ℕ := 41
def total_distance : ℕ := 5280

-- Define the theorem
theorem oscar_elmer_difference : 
  let gaps := total_poles - 1
  let elmer_total_strides := elmer_strides_per_gap * gaps
  let oscar_total_leaps := oscar_leaps_per_gap * gaps
  let elmer_stride_length := total_distance / elmer_total_strides
  let oscar_leap_length := total_distance / oscar_total_leaps
  oscar_leap_length - elmer_stride_length = 8 := by
  sorry

end oscar_elmer_difference_l3633_363303


namespace f_symmetric_l3633_363366

def f (a b x : ℝ) : ℝ := x^5 + a*x^3 + b*x + 1

theorem f_symmetric (a b : ℝ) : f a b (-2) = 10 → f a b 2 = -10 := by
  sorry

end f_symmetric_l3633_363366


namespace quadratic_equation_unique_solution_l3633_363341

theorem quadratic_equation_unique_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 36 * x + c = 0) →
  (a + c = 41) →
  (a < c) →
  (a = (41 - Real.sqrt 385) / 2 ∧ c = (41 + Real.sqrt 385) / 2) := by
sorry

end quadratic_equation_unique_solution_l3633_363341


namespace line_of_symmetry_l3633_363347

-- Define the circles
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y + 4 = 0

-- Define the line of symmetry
def line_l (x y : ℝ) : Prop := x - y + 2 = 0

-- Theorem statement
theorem line_of_symmetry :
  ∀ (x y : ℝ), 
    (∃ (x' y' : ℝ), circle_O x' y' ∧ circle_C x y ∧ 
      line_l ((x + x')/2) ((y + y')/2) ∧
      (x - x')^2 + (y - y')^2 = (x' - x)^2 + (y' - y)^2) :=
sorry

end line_of_symmetry_l3633_363347


namespace two_out_graph_partition_theorem_l3633_363329

/-- A directed graph where each vertex has exactly two outgoing edges -/
structure TwoOutGraph (V : Type*) :=
  (edges : V → V × V)

/-- A partition of vertices into districts -/
def DistrictPartition (V : Type*) := V → Fin 1014

theorem two_out_graph_partition_theorem {V : Type*} (G : TwoOutGraph V) :
  ∃ (partition : DistrictPartition V),
    (∀ v w : V, (partition v = partition w) → 
      (G.edges v).1 ≠ w ∧ (G.edges v).2 ≠ w) ∧
    (∀ d1 d2 : Fin 1014, d1 ≠ d2 → 
      (∀ v w : V, partition v = d1 → partition w = d2 → 
        ((G.edges v).1 = w ∨ (G.edges v).2 = w) → 
        ∀ x y : V, partition x = d1 → partition y = d2 → 
          ((G.edges x).1 = y ∨ (G.edges x).2 = y) → 
          ((G.edges v).1 = w ∨ (G.edges x).1 = y) ∧ 
          ((G.edges v).2 = w ∨ (G.edges x).2 = y))) :=
sorry

end two_out_graph_partition_theorem_l3633_363329


namespace elliott_triangle_hypotenuse_l3633_363304

-- Define a right-angle triangle
structure RightTriangle where
  base : ℝ
  height : ℝ
  hypotenuse : ℝ
  right_angle : base^2 + height^2 = hypotenuse^2

-- Theorem statement
theorem elliott_triangle_hypotenuse (t : RightTriangle) 
  (h1 : t.base = 4)
  (h2 : t.height = 3) : 
  t.hypotenuse = 5 := by
  sorry

end elliott_triangle_hypotenuse_l3633_363304


namespace min_colors_100x100_board_l3633_363380

/-- Represents a board with cells divided into triangles -/
structure Board :=
  (size : Nat)
  (cells_divided : Bool)

/-- Represents a coloring of the triangles on the board -/
def Coloring := Board → Nat → Nat → Bool → Nat

/-- Checks if a coloring is valid (no adjacent triangles have the same color) -/
def is_valid_coloring (b : Board) (c : Coloring) : Prop := sorry

/-- The minimum number of colors needed for a valid coloring -/
def min_colors (b : Board) : Nat := sorry

/-- Theorem stating the minimum number of colors for a 100x100 board -/
theorem min_colors_100x100_board :
  ∀ (b : Board),
    b.size = 100 ∧
    b.cells_divided →
    min_colors b = 8 := by sorry

end min_colors_100x100_board_l3633_363380


namespace functional_equation_solutions_l3633_363322

/-- The functional equation problem -/
def FunctionalEquation (t : ℝ) (f : ℝ → ℝ) : Prop :=
  t ≠ -1 ∧ ∀ x y : ℝ, (t + 1) * f (1 + x * y) - f (x + y) = f (x + 1) * f (y + 1)

/-- The set of solutions to the functional equation -/
def Solutions (t : ℝ) (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = 0) ∨ (∀ x, f x = t) ∨ (∀ x, f x = (t + 1) * x - (t + 2))

/-- The main theorem: all solutions to the functional equation -/
theorem functional_equation_solutions (t : ℝ) (f : ℝ → ℝ) :
  FunctionalEquation t f ↔ Solutions t f := by
  sorry

end functional_equation_solutions_l3633_363322


namespace percentage_both_correct_l3633_363354

theorem percentage_both_correct (total : ℝ) (first_correct : ℝ) (second_correct : ℝ) (neither_correct : ℝ) :
  total > 0 →
  first_correct / total = 0.75 →
  second_correct / total = 0.30 →
  neither_correct / total = 0.20 →
  (first_correct + second_correct - (total - neither_correct)) / total = 0.25 := by
sorry

end percentage_both_correct_l3633_363354


namespace mean_proportional_problem_l3633_363398

theorem mean_proportional_problem (x : ℝ) :
  (x * 9409 : ℝ).sqrt = 8665 → x = 7981 := by
  sorry

end mean_proportional_problem_l3633_363398


namespace ab_value_l3633_363395

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) : a * b = 10 := by
  sorry

end ab_value_l3633_363395


namespace f_7_equals_neg_2_l3633_363319

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_periodic_4 (f : ℝ → ℝ) : Prop := ∀ x, f (x + 4) = f x

theorem f_7_equals_neg_2 (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_periodic : is_periodic_4 f) 
  (h_interval : ∀ x ∈ Set.Ioo 0 2, f x = 2 * x^2) : 
  f 7 = -2 := by
  sorry

end f_7_equals_neg_2_l3633_363319


namespace inequality_proof_l3633_363308

theorem inequality_proof (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_eq : (a + 1)⁻¹ + (b + 1)⁻¹ + (c + 1)⁻¹ + (d + 1)⁻¹ = 3) : 
  (a * b * c)^(1/3) + (b * c * d)^(1/3) + (c * d * a)^(1/3) + (d * a * b)^(1/3) ≤ 4/3 := by
sorry

end inequality_proof_l3633_363308


namespace carols_peanuts_l3633_363364

theorem carols_peanuts (initial_peanuts : ℕ) :
  initial_peanuts + 5 = 7 → initial_peanuts = 2 := by
  sorry

end carols_peanuts_l3633_363364


namespace binary_representation_of_21_l3633_363320

theorem binary_representation_of_21 :
  (21 : ℕ).digits 2 = [1, 0, 1, 0, 1] :=
sorry

end binary_representation_of_21_l3633_363320


namespace hospital_staff_count_l3633_363342

theorem hospital_staff_count (doctors nurses : ℕ) (h1 : doctors * 11 = nurses * 8) (h2 : nurses = 264) : 
  doctors + nurses = 456 := by
sorry

end hospital_staff_count_l3633_363342


namespace partner_c_profit_l3633_363335

/-- Represents a business partner --/
inductive Partner
| A
| B
| C
| D

/-- Calculates the profit for a given partner in the first year of the business --/
def calculateProfit (totalProfit : ℕ) (partnerShares : Partner → ℕ) (dJoinTime : ℚ) (partner : Partner) : ℚ :=
  let fullYearShares := partnerShares Partner.A + partnerShares Partner.B + partnerShares Partner.C
  let dAdjustedShare := (partnerShares Partner.D : ℚ) * dJoinTime
  let totalAdjustedShares := (fullYearShares : ℚ) + dAdjustedShare
  let sharePerPart := (totalProfit : ℚ) / totalAdjustedShares
  sharePerPart * (partnerShares partner : ℚ)

/-- Theorem stating that partner C's profit is $20,250 given the problem conditions --/
theorem partner_c_profit :
  let totalProfit : ℕ := 56700
  let partnerShares : Partner → ℕ := fun
    | Partner.A => 7
    | Partner.B => 9
    | Partner.C => 10
    | Partner.D => 4
  let dJoinTime : ℚ := 1/2
  calculateProfit totalProfit partnerShares dJoinTime Partner.C = 20250 := by
  sorry


end partner_c_profit_l3633_363335


namespace count_factors_of_eight_squared_nine_cubed_seven_fifth_l3633_363365

theorem count_factors_of_eight_squared_nine_cubed_seven_fifth (n : Nat) :
  n = 8^2 * 9^3 * 7^5 →
  (Finset.filter (λ m : Nat => n % m = 0) (Finset.range (n + 1))).card = 294 :=
by sorry

end count_factors_of_eight_squared_nine_cubed_seven_fifth_l3633_363365


namespace total_travel_time_is_45_hours_l3633_363362

/-- Represents a city with its time zone offset from New Orleans --/
structure City where
  name : String
  offset : Int

/-- Represents a flight segment with departure and arrival cities, and duration --/
structure FlightSegment where
  departure : City
  arrival : City
  duration : Nat

/-- Represents a layover with city and duration --/
structure Layover where
  city : City
  duration : Nat

/-- Calculates the total travel time considering time zone changes --/
def totalTravelTime (segments : List FlightSegment) (layovers : List Layover) : Nat :=
  sorry

/-- The cities involved in Sue's journey --/
def newOrleans : City := { name := "New Orleans", offset := 0 }
def atlanta : City := { name := "Atlanta", offset := 0 }
def chicago : City := { name := "Chicago", offset := -1 }
def newYork : City := { name := "New York", offset := 0 }
def denver : City := { name := "Denver", offset := -2 }
def sanFrancisco : City := { name := "San Francisco", offset := -3 }

/-- Sue's flight segments --/
def flightSegments : List FlightSegment := [
  { departure := newOrleans, arrival := atlanta, duration := 2 },
  { departure := atlanta, arrival := chicago, duration := 5 },
  { departure := chicago, arrival := newYork, duration := 3 },
  { departure := newYork, arrival := denver, duration := 6 },
  { departure := denver, arrival := sanFrancisco, duration := 4 }
]

/-- Sue's layovers --/
def layovers : List Layover := [
  { city := atlanta, duration := 4 },
  { city := chicago, duration := 3 },
  { city := newYork, duration := 16 },
  { city := denver, duration := 5 }
]

/-- Theorem: The total travel time from New Orleans to San Francisco is 45 hours --/
theorem total_travel_time_is_45_hours :
  totalTravelTime flightSegments layovers = 45 := by sorry

end total_travel_time_is_45_hours_l3633_363362
