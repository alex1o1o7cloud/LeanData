import Mathlib

namespace NUMINAMATH_CALUDE_average_class_size_is_35_l1015_101505

/-- Represents the number of children in each age group --/
structure AgeGroups where
  three_year_olds : ℕ
  four_year_olds : ℕ
  five_year_olds : ℕ
  six_year_olds : ℕ

/-- Represents the Sunday school setup --/
def SundaySchool (ages : AgeGroups) : Prop :=
  ages.three_year_olds = 13 ∧
  ages.four_year_olds = 20 ∧
  ages.five_year_olds = 15 ∧
  ages.six_year_olds = 22

/-- Calculates the average class size --/
def averageClassSize (ages : AgeGroups) : ℚ :=
  let class1 := ages.three_year_olds + ages.four_year_olds
  let class2 := ages.five_year_olds + ages.six_year_olds
  (class1 + class2) / 2

/-- Theorem stating that the average class size is 35 --/
theorem average_class_size_is_35 (ages : AgeGroups) 
  (h : SundaySchool ages) : averageClassSize ages = 35 := by
  sorry

end NUMINAMATH_CALUDE_average_class_size_is_35_l1015_101505


namespace NUMINAMATH_CALUDE_inequality_proof_l1015_101568

theorem inequality_proof (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  (x^2 / (y - 1)) + (y^2 / (x - 1)) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1015_101568


namespace NUMINAMATH_CALUDE_gross_profit_calculation_l1015_101522

theorem gross_profit_calculation (sales_price : ℝ) (gross_profit_percentage : ℝ) :
  sales_price = 91 →
  gross_profit_percentage = 1.6 →
  ∃ (cost : ℝ), sales_price = cost + gross_profit_percentage * cost ∧
                 gross_profit_percentage * cost = 56 := by
  sorry

end NUMINAMATH_CALUDE_gross_profit_calculation_l1015_101522


namespace NUMINAMATH_CALUDE_competition_distance_l1015_101516

/-- Represents the distances cycled on each day of the week -/
structure WeekDistances where
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ
  thursday : ℝ
  friday : ℝ
  saturday : ℝ
  sunday : ℝ

/-- Calculates the total distance cycled in a week -/
def totalDistance (distances : WeekDistances) : ℝ :=
  distances.monday + distances.tuesday + distances.wednesday + 
  distances.thursday + distances.friday + distances.saturday + distances.sunday

/-- Theorem stating the total distance cycled in the competition week -/
theorem competition_distance : ∃ (distances : WeekDistances),
  distances.monday = 40 ∧
  distances.tuesday = 50 ∧
  distances.wednesday = distances.tuesday * 0.5 ∧
  distances.thursday = distances.monday + distances.wednesday ∧
  distances.friday = distances.thursday * 1.2 ∧
  distances.saturday = distances.friday * 0.75 ∧
  distances.sunday = distances.saturday - distances.wednesday ∧
  totalDistance distances = 350 := by
  sorry


end NUMINAMATH_CALUDE_competition_distance_l1015_101516


namespace NUMINAMATH_CALUDE_f_composition_one_third_l1015_101538

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3
  else 8^x

-- State the theorem
theorem f_composition_one_third : f (f (1/3)) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_one_third_l1015_101538


namespace NUMINAMATH_CALUDE_arithmetic_progression_prime_divisibility_l1015_101513

theorem arithmetic_progression_prime_divisibility
  (p : ℕ) (a : ℕ → ℕ) (d : ℕ) 
  (h_prime : Prime p)
  (h_seq : ∀ i ∈ Finset.range p, Prime (a i))
  (h_arith : ∀ i ∈ Finset.range (p - 1), a (i + 1) = a i + d)
  (h_incr : ∀ i ∈ Finset.range (p - 1), a i < a (i + 1))
  (h_greater : p < a 0) :
  p ∣ d := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_prime_divisibility_l1015_101513


namespace NUMINAMATH_CALUDE_function_symmetry_origin_l1015_101594

/-- The function f(x) = x^3 + x is symmetric with respect to the origin. -/
theorem function_symmetry_origin (x : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^3 + x
  f (-x) = -f x := by
  sorry

end NUMINAMATH_CALUDE_function_symmetry_origin_l1015_101594


namespace NUMINAMATH_CALUDE_sqrt_plus_square_zero_l1015_101550

theorem sqrt_plus_square_zero (m n : ℝ) : 
  Real.sqrt (m + 1) + (n - 2)^2 = 0 → m + n = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_plus_square_zero_l1015_101550


namespace NUMINAMATH_CALUDE_smallest_sum_of_coefficients_l1015_101547

theorem smallest_sum_of_coefficients (a b : ℤ) : 
  (∀ x : ℝ, (x^2 + a*x + 20)*(x^2 + 17*x + b) = 0 → (∃ k : ℤ, x = ↑k ∧ k < 0)) →
  (∀ c d : ℤ, (∀ y : ℝ, (y^2 + c*y + 20)*(y^2 + 17*y + d) = 0 → (∃ m : ℤ, y = ↑m ∧ m < 0)) → 
    a + b ≤ c + d) →
  a + b = -5 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_coefficients_l1015_101547


namespace NUMINAMATH_CALUDE_mike_weekly_pullups_l1015_101555

/-- The number of pull-ups Mike does in a week -/
def weekly_pullups (pullups_per_visit : ℕ) (visits_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  pullups_per_visit * visits_per_day * days_per_week

/-- Theorem stating that Mike does 70 pull-ups in a week -/
theorem mike_weekly_pullups :
  weekly_pullups 2 5 7 = 70 := by
  sorry

end NUMINAMATH_CALUDE_mike_weekly_pullups_l1015_101555


namespace NUMINAMATH_CALUDE_jackson_missed_wednesdays_l1015_101543

/-- The number of missed Wednesdays in Jackson's school year --/
def missed_wednesdays (weeks : ℕ) (total_sandwiches : ℕ) (missed_fridays : ℕ) : ℕ :=
  weeks * 2 - total_sandwiches - missed_fridays

theorem jackson_missed_wednesdays :
  missed_wednesdays 36 69 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_jackson_missed_wednesdays_l1015_101543


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1015_101578

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geometric : is_geometric_sequence a) 
  (h_condition : a 1 * a 13 + 2 * (a 7)^2 = 5 * Real.pi) : 
  Real.cos (a 2 * a 12) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1015_101578


namespace NUMINAMATH_CALUDE_equation_equivalence_l1015_101519

theorem equation_equivalence (a b : ℝ) (h : a + 2 * b + 2 = Real.sqrt 2) : 
  4 * a + 8 * b + 5 = 4 * Real.sqrt 2 - 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l1015_101519


namespace NUMINAMATH_CALUDE_cubic_roots_sum_of_cubes_l1015_101566

theorem cubic_roots_sum_of_cubes (p q s r₁ r₂ : ℝ) : 
  (∀ x, x^3 - p*x^2 + q*x - s = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = 0) →
  r₁^3 + r₂^3 = p^3 - 3*q*p :=
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_of_cubes_l1015_101566


namespace NUMINAMATH_CALUDE_earl_owes_fred_l1015_101598

/-- Represents the financial state of Earl, Fred, and Greg -/
structure FinancialState where
  earl : Int
  fred : Int
  greg : Int

/-- Calculates the final financial state after debts are paid -/
def finalState (initial : FinancialState) (earlOwes : Int) : FinancialState :=
  { earl := initial.earl - earlOwes + 40,
    fred := initial.fred + earlOwes - 32,
    greg := initial.greg + 32 - 40 }

/-- The theorem to be proved -/
theorem earl_owes_fred (initial : FinancialState) :
  initial.earl = 90 →
  initial.fred = 48 →
  initial.greg = 36 →
  (let final := finalState initial 28
   final.earl + final.greg = 130) :=
by sorry

end NUMINAMATH_CALUDE_earl_owes_fred_l1015_101598


namespace NUMINAMATH_CALUDE_min_value_theorem_max_value_theorem_l1015_101574

-- Problem 1
theorem min_value_theorem (x : ℝ) (h : x > 0) :
  x + 4 / x + 5 ≥ 9 :=
sorry

-- Problem 2
theorem max_value_theorem (x : ℝ) (h1 : x > 0) (h2 : x < 1/2) :
  1/2 * x * (1 - 2*x) ≤ 1/16 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_max_value_theorem_l1015_101574


namespace NUMINAMATH_CALUDE_odd_function_value_and_range_and_inequality_l1015_101562

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 - 4 / (2 * a^x + a)

theorem odd_function_value_and_range_and_inequality (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x, f a x = -f a (-x)) ∧
  (∀ x, -1 < (2^x - 1) / (2^x + 1) ∧ (2^x - 1) / (2^x + 1) < 1) ∧
  (∀ x ∈ Set.Ioo 0 1, ∃ t ≥ 0, ∀ s ≥ t, s * ((2^x - 1) / (2^x + 1)) ≥ 2^x - 2) :=
by sorry

end NUMINAMATH_CALUDE_odd_function_value_and_range_and_inequality_l1015_101562


namespace NUMINAMATH_CALUDE_rectangle_with_hole_area_l1015_101586

theorem rectangle_with_hole_area (x : ℝ) : 
  let large_length : ℝ := 2*x + 8
  let large_width : ℝ := x + 6
  let hole_length : ℝ := 3*x - 4
  let hole_width : ℝ := x + 1
  large_length * large_width - hole_length * hole_width = -x^2 + 22*x + 52 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_with_hole_area_l1015_101586


namespace NUMINAMATH_CALUDE_data_transformation_theorem_l1015_101565

/-- Represents a set of numerical data -/
structure DataSet where
  values : List ℝ

/-- Calculates the average of a DataSet -/
def average (d : DataSet) : ℝ := sorry

/-- Calculates the variance of a DataSet -/
def variance (d : DataSet) : ℝ := sorry

/-- Transforms a DataSet by subtracting a constant from each value -/
def transform (d : DataSet) (c : ℝ) : DataSet := sorry

theorem data_transformation_theorem (original : DataSet) :
  let transformed := transform original 80
  average transformed = 1.2 →
  variance transformed = 4.4 →
  average original = 81.2 ∧ variance original = 4.4 := by sorry

end NUMINAMATH_CALUDE_data_transformation_theorem_l1015_101565


namespace NUMINAMATH_CALUDE_gcd_problem_l1015_101549

theorem gcd_problem (b : ℤ) (h : 504 ∣ b) : 
  Nat.gcd (4*b^3 + 2*b^2 + 5*b + 63).natAbs b.natAbs = 63 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1015_101549


namespace NUMINAMATH_CALUDE_basketball_team_callbacks_l1015_101579

theorem basketball_team_callbacks (girls_tryout : ℕ) (boys_tryout : ℕ) (didnt_make_cut : ℕ) :
  girls_tryout = 9 →
  boys_tryout = 14 →
  didnt_make_cut = 21 →
  girls_tryout + boys_tryout - didnt_make_cut = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_team_callbacks_l1015_101579


namespace NUMINAMATH_CALUDE_product_mod_500_l1015_101521

theorem product_mod_500 : (1493 * 1998) % 500 = 14 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_500_l1015_101521


namespace NUMINAMATH_CALUDE_dice_probability_l1015_101528

/-- The number of dice --/
def n : ℕ := 7

/-- The number of sides on each die --/
def sides : ℕ := 12

/-- The number of favorable outcomes on each die (numbers less than 6) --/
def favorable : ℕ := 5

/-- The number of dice we want to show a favorable outcome --/
def k : ℕ := 3

/-- The probability of exactly k out of n dice showing a number less than 6 --/
def probability : ℚ := (n.choose k) * (favorable / sides) ^ k * ((sides - favorable) / sides) ^ (n - k)

theorem dice_probability : probability = 10504375 / 373248 := by
  sorry

end NUMINAMATH_CALUDE_dice_probability_l1015_101528


namespace NUMINAMATH_CALUDE_baking_powder_difference_l1015_101585

-- Define the constants
def yesterday_supply : Real := 1.5 -- in kg
def today_supply : Real := 1.2 -- in kg (converted from 1200 grams)
def box_size : Real := 5 -- kg per box

-- Define the theorem
theorem baking_powder_difference :
  yesterday_supply - today_supply = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_baking_powder_difference_l1015_101585


namespace NUMINAMATH_CALUDE_football_throw_distance_l1015_101580

theorem football_throw_distance (parker_distance grant_distance kyle_distance : ℝ) :
  parker_distance = 16 ∧
  grant_distance = parker_distance * 1.25 ∧
  kyle_distance = grant_distance * 2 →
  kyle_distance - parker_distance = 24 := by
  sorry

end NUMINAMATH_CALUDE_football_throw_distance_l1015_101580


namespace NUMINAMATH_CALUDE_wrapping_paper_area_l1015_101592

/-- The area of wrapping paper needed to cover a rectangular box with a small cube on top -/
theorem wrapping_paper_area (w h : ℝ) (w_pos : 0 < w) (h_pos : 0 < h) :
  let box_width := 2 * w
  let box_length := w
  let box_height := h
  let cube_side := w / 2
  let total_height := box_height + cube_side
  let paper_width := box_width + 2 * total_height
  let paper_length := box_length + 2 * total_height
  paper_width * paper_length = (3 * w + 2 * h) * (2 * w + 2 * h) :=
by sorry


end NUMINAMATH_CALUDE_wrapping_paper_area_l1015_101592


namespace NUMINAMATH_CALUDE_fifth_month_sale_l1015_101571

-- Define the sales for the first four months
def first_four_sales : List Int := [5420, 5660, 6200, 6350]

-- Define the sale for the sixth month
def sixth_month_sale : Int := 7070

-- Define the average sale for six months
def average_sale : Int := 6200

-- Define the number of months
def num_months : Int := 6

-- Theorem to prove
theorem fifth_month_sale :
  let total_sales := average_sale * num_months
  let known_sales := first_four_sales.sum + sixth_month_sale
  total_sales - known_sales = 6500 := by
  sorry

end NUMINAMATH_CALUDE_fifth_month_sale_l1015_101571


namespace NUMINAMATH_CALUDE_community_service_arrangements_l1015_101541

def arrange_people (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

theorem community_service_arrangements : 
  arrange_people 6 4 + arrange_people 6 3 + arrange_people 6 2 = 50 := by
  sorry

end NUMINAMATH_CALUDE_community_service_arrangements_l1015_101541


namespace NUMINAMATH_CALUDE_lloyds_hourly_rate_l1015_101593

def regular_hours : ℝ := 7.5
def overtime_rate : ℝ := 1.5
def total_hours : ℝ := 10.5
def total_earnings : ℝ := 66

def hourly_rate : ℝ := 5.5

theorem lloyds_hourly_rate : 
  regular_hours * hourly_rate + 
  (total_hours - regular_hours) * overtime_rate * hourly_rate = 
  total_earnings := by sorry

end NUMINAMATH_CALUDE_lloyds_hourly_rate_l1015_101593


namespace NUMINAMATH_CALUDE_furniture_shop_cost_price_l1015_101567

theorem furniture_shop_cost_price (markup_percentage : ℝ) (selling_price : ℝ) (cost_price : ℝ) : 
  markup_percentage = 20 →
  selling_price = 3600 →
  selling_price = cost_price * (1 + markup_percentage / 100) →
  cost_price = 3000 := by
  sorry

end NUMINAMATH_CALUDE_furniture_shop_cost_price_l1015_101567


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l1015_101569

theorem geometric_sequence_seventh_term
  (a₁ : ℝ)
  (a₁₀ : ℝ)
  (h₁ : a₁ = 12)
  (h₂ : a₁₀ = 78732)
  (h₃ : ∀ n : ℕ, 1 ≤ n → n ≤ 10 → ∃ r : ℝ, a₁ * r^(n-1) = a₁₀^((n-1)/9) * a₁^(1-(n-1)/9)) :
  ∃ a₇ : ℝ, a₇ = 8748 ∧ a₁ * (a₁₀ / a₁)^(6/9) = a₇ :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l1015_101569


namespace NUMINAMATH_CALUDE_unique_solution_for_cubic_equations_l1015_101526

/-- Represents the roots of a cubic equation -/
structure CubicRoots (α : Type*) [Field α] where
  r₁ : α
  r₂ : α
  r₃ : α

/-- Checks if three numbers form an arithmetic progression -/
def is_arithmetic_progression {α : Type*} [Field α] (x y z : α) : Prop :=
  y - x = z - y ∧ y - x ≠ 0

/-- Checks if three numbers form a geometric progression -/
def is_geometric_progression {α : Type*} [Field α] (x y z : α) : Prop :=
  ∃ r : α, r ≠ 1 ∧ y = x * r ∧ z = y * r

/-- Represents the coefficients of the first cubic equation -/
structure FirstEquationCoeffs (α : Type*) [Field α] where
  a : α
  b : α
  c : α

/-- Represents the coefficients of the second cubic equation -/
structure SecondEquationCoeffs (α : Type*) [Field α] where
  b : α
  c : α

/-- The main theorem -/
theorem unique_solution_for_cubic_equations 
  (f : FirstEquationCoeffs ℝ) 
  (g : SecondEquationCoeffs ℝ)
  (roots1 : CubicRoots ℝ)
  (roots2 : CubicRoots ℝ)
  (h1 : roots1.r₁^3 - 3*f.a*roots1.r₁^2 + f.b*roots1.r₁ + 18*f.c = 0)
  (h2 : roots1.r₂^3 - 3*f.a*roots1.r₂^2 + f.b*roots1.r₂ + 18*f.c = 0)
  (h3 : roots1.r₃^3 - 3*f.a*roots1.r₃^2 + f.b*roots1.r₃ + 18*f.c = 0)
  (h4 : is_arithmetic_progression roots1.r₁ roots1.r₂ roots1.r₃)
  (h5 : roots2.r₁^3 + g.b*roots2.r₁^2 + roots2.r₁ - g.c^3 = 0)
  (h6 : roots2.r₂^3 + g.b*roots2.r₂^2 + roots2.r₂ - g.c^3 = 0)
  (h7 : roots2.r₃^3 + g.b*roots2.r₃^2 + roots2.r₃ - g.c^3 = 0)
  (h8 : is_geometric_progression roots2.r₁ roots2.r₂ roots2.r₃)
  (h9 : f.b = g.b)
  (h10 : f.c = g.c)
  : f.a = 2 ∧ f.b = 9 := by sorry

end NUMINAMATH_CALUDE_unique_solution_for_cubic_equations_l1015_101526


namespace NUMINAMATH_CALUDE_a_investment_l1015_101525

/-- A's investment in a partnership business --/
def partners_investment (total_profit partner_a_total_received partner_b_investment : ℚ) : ℚ :=
  let management_fee := 0.1 * total_profit
  let remaining_profit := total_profit - management_fee
  let partner_a_profit_share := partner_a_total_received - management_fee
  (partner_a_profit_share * partner_b_investment) / (remaining_profit - partner_a_profit_share)

/-- Theorem stating A's investment given the problem conditions --/
theorem a_investment (total_profit partner_a_total_received partner_b_investment : ℚ) 
  (h1 : total_profit = 9600)
  (h2 : partner_a_total_received = 4800)
  (h3 : partner_b_investment = 25000) :
  partners_investment total_profit partner_a_total_received partner_b_investment = 20000 := by
  sorry

end NUMINAMATH_CALUDE_a_investment_l1015_101525


namespace NUMINAMATH_CALUDE_dance_attendance_l1015_101546

/-- The number of boys attending the dance -/
def num_boys : ℕ := 14

/-- The number of girls attending the dance -/
def num_girls : ℕ := num_boys / 2

theorem dance_attendance :
  (num_boys = 2 * num_girls) ∧
  (num_boys = (num_girls - 1) + 8) →
  num_boys = 14 :=
by sorry

end NUMINAMATH_CALUDE_dance_attendance_l1015_101546


namespace NUMINAMATH_CALUDE_min_value_sum_l1015_101573

theorem min_value_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / (a + 3) + 1 / (b + 3) = 1 / 4) : 
  a + 3 * b ≥ 12 + 16 * Real.sqrt 3 ∧ 
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 
    1 / (a₀ + 3) + 1 / (b₀ + 3) = 1 / 4 ∧
    a₀ + 3 * b₀ = 12 + 16 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_l1015_101573


namespace NUMINAMATH_CALUDE_calculate_expression_l1015_101508

theorem calculate_expression : 15 * 28 + 42 * 15 + 15^2 = 1275 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1015_101508


namespace NUMINAMATH_CALUDE_divisible_by_seven_l1015_101507

/-- The number of repeated digits -/
def n : ℕ := 50

/-- The number formed by n eights followed by x followed by n nines -/
def f (x : ℕ) : ℕ :=
  8 * (10^(2*n + 1) - 1) / 9 + x * 10^n + 9 * (10^n - 1) / 9

/-- The main theorem -/
theorem divisible_by_seven (x : ℕ) : 7 ∣ f x ↔ x = 0 := by sorry

end NUMINAMATH_CALUDE_divisible_by_seven_l1015_101507


namespace NUMINAMATH_CALUDE_mary_flour_calculation_l1015_101534

/-- The amount of flour needed for the recipe -/
def total_flour : ℕ := 9

/-- The amount of flour Mary has already added -/
def added_flour : ℕ := 3

/-- The remaining amount of flour Mary needs to add -/
def remaining_flour : ℕ := total_flour - added_flour

theorem mary_flour_calculation :
  remaining_flour = 6 := by
  sorry

end NUMINAMATH_CALUDE_mary_flour_calculation_l1015_101534


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l1015_101584

theorem divisibility_equivalence (n : ℕ) : 
  7 ∣ (3^n + n^3) ↔ 7 ∣ (3^n * n^3 + 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l1015_101584


namespace NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l1015_101527

theorem rectangle_circle_area_ratio (l w r : ℝ) (h1 : 2 * l + 2 * w = 2 * Real.pi * r) (h2 : l = 2 * w) :
  (l * w) / (Real.pi * r^2) = 2 * Real.pi / 9 := by
sorry

end NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l1015_101527


namespace NUMINAMATH_CALUDE_matthews_crackers_l1015_101524

theorem matthews_crackers (num_friends : ℕ) (crackers_eaten_per_friend : ℕ) 
  (h1 : num_friends = 18)
  (h2 : crackers_eaten_per_friend = 2) :
  num_friends * crackers_eaten_per_friend = 36 := by
  sorry

end NUMINAMATH_CALUDE_matthews_crackers_l1015_101524


namespace NUMINAMATH_CALUDE_tan_thirteen_pi_fourth_l1015_101599

theorem tan_thirteen_pi_fourth : Real.tan (13 * π / 4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_thirteen_pi_fourth_l1015_101599


namespace NUMINAMATH_CALUDE_inverse_existence_l1015_101531

-- Define the three functions
def linear_function (x : ℝ) : ℝ := sorry
def quadratic_function (x : ℝ) : ℝ := sorry
def exponential_function (x : ℝ) : ℝ := sorry

-- Define the property of having an inverse
def has_inverse (f : ℝ → ℝ) : Prop := sorry

-- Theorem statement
theorem inverse_existence :
  (has_inverse linear_function) ∧
  (¬ has_inverse quadratic_function) ∧
  (has_inverse exponential_function) := by sorry

end NUMINAMATH_CALUDE_inverse_existence_l1015_101531


namespace NUMINAMATH_CALUDE_rectangle_side_length_l1015_101540

/-- If a rectangle has area 4a²b³ and one side 2ab³, then the other side is 2a -/
theorem rectangle_side_length (a b : ℝ) (area : ℝ) (side1 : ℝ) :
  area = 4 * a^2 * b^3 → side1 = 2 * a * b^3 → area / side1 = 2 * a :=
by sorry

end NUMINAMATH_CALUDE_rectangle_side_length_l1015_101540


namespace NUMINAMATH_CALUDE_quadratic_inequality_sum_l1015_101517

theorem quadratic_inequality_sum (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + bx + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) → 
  a + b = -14 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_sum_l1015_101517


namespace NUMINAMATH_CALUDE_event_properties_l1015_101572

-- Define the types of events
inductive Event
| Train
| Shooting

-- Define the type of outcomes
inductive Outcome
| Success
| Failure

-- Define a function to get the number of trials for each event
def num_trials (e : Event) : ℕ :=
  match e with
  | Event.Train => 3
  | Event.Shooting => 2

-- Define a function to get the possible outcomes for each event
def possible_outcomes (e : Event) : List Outcome :=
  [Outcome.Success, Outcome.Failure]

-- Theorem statement
theorem event_properties :
  (∀ e : Event, num_trials e > 0) ∧
  (∀ e : Event, possible_outcomes e = [Outcome.Success, Outcome.Failure]) :=
by sorry

end NUMINAMATH_CALUDE_event_properties_l1015_101572


namespace NUMINAMATH_CALUDE_black_squares_10th_row_l1015_101514

def stair_step_squares (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else stair_step_squares (n - 1) + 2^(n - 1)

def black_squares (n : ℕ) : ℕ :=
  (stair_step_squares n - 1) / 2

theorem black_squares_10th_row :
  black_squares 10 = 511 := by
  sorry

end NUMINAMATH_CALUDE_black_squares_10th_row_l1015_101514


namespace NUMINAMATH_CALUDE_arithmetic_computation_l1015_101577

theorem arithmetic_computation : -10 * 5 - (-8 * -4) + (-12 * -6) + 2 * 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l1015_101577


namespace NUMINAMATH_CALUDE_second_smallest_five_digit_in_pascal_l1015_101595

/-- Pascal's triangle function -/
def pascal (n k : ℕ) : ℕ := sorry

/-- Predicate to check if a number is in Pascal's triangle -/
def inPascalTriangle (x : ℕ) : Prop :=
  ∃ n k : ℕ, pascal n k = x

/-- Predicate to check if a number is a five-digit number -/
def isFiveDigit (x : ℕ) : Prop :=
  10000 ≤ x ∧ x ≤ 99999

/-- The second smallest five-digit number in Pascal's triangle is 10001 -/
theorem second_smallest_five_digit_in_pascal :
  ∃! x : ℕ, inPascalTriangle x ∧ isFiveDigit x ∧
  (∃! y : ℕ, y < x ∧ inPascalTriangle y ∧ isFiveDigit y) ∧
  x = 10001 := by sorry

end NUMINAMATH_CALUDE_second_smallest_five_digit_in_pascal_l1015_101595


namespace NUMINAMATH_CALUDE_game_result_l1015_101512

def score_function (n : ℕ) : ℕ :=
  if n % 3 = 0 then 9
  else if n % 2 = 0 then 3
  else if n % 2 = 1 ∧ n % 3 ≠ 0 then 1
  else 0

def allie_rolls : List ℕ := [5, 2, 6, 1, 3]
def betty_rolls : List ℕ := [6, 4, 1, 2, 5]

theorem game_result :
  (List.sum (List.map score_function allie_rolls)) *
  (List.sum (List.map score_function betty_rolls)) = 391 := by
  sorry

end NUMINAMATH_CALUDE_game_result_l1015_101512


namespace NUMINAMATH_CALUDE_target_line_properties_l1015_101537

/-- The equation of line l₁ -/
def l₁ (x y : ℝ) : Prop := x - 6 * y + 4 = 0

/-- The equation of line l₂ -/
def l₂ (x y : ℝ) : Prop := 2 * x + y = 5

/-- The intersection point of l₁ and l₂ -/
def intersection_point : ℝ × ℝ := (2, 1)

/-- The equation of the line we're proving -/
def target_line (x y : ℝ) : Prop := x - 2 * y = 0

/-- Theorem stating that the target_line passes through the intersection point and is perpendicular to l₂ -/
theorem target_line_properties :
  (target_line (intersection_point.1) (intersection_point.2)) ∧
  (∀ x y : ℝ, l₂ x y → ∀ x' y' : ℝ, target_line x' y' →
    (y' - intersection_point.2) * (x - intersection_point.1) = 
    -(x' - intersection_point.1) * (y - intersection_point.2)) :=
sorry

end NUMINAMATH_CALUDE_target_line_properties_l1015_101537


namespace NUMINAMATH_CALUDE_usual_walking_time_l1015_101563

theorem usual_walking_time (usual_speed : ℝ) (usual_time : ℝ) : 
  usual_speed > 0 → usual_time > 0 →
  (4 / 5 * usual_speed) * (usual_time + 10) = usual_speed * usual_time →
  usual_time = 40 := by
sorry

end NUMINAMATH_CALUDE_usual_walking_time_l1015_101563


namespace NUMINAMATH_CALUDE_garden_area_l1015_101539

theorem garden_area (width length : ℝ) : 
  length = 3 * width + 30 →
  2 * (width + length) = 780 →
  width * length = 27000 := by
sorry

end NUMINAMATH_CALUDE_garden_area_l1015_101539


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1015_101536

theorem imaginary_part_of_complex_fraction :
  let i : ℂ := Complex.I
  let z : ℂ := (1 - i) / (3 - i)
  Complex.im z = -1/5 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1015_101536


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1015_101511

theorem trigonometric_identity (α : Real) : 
  Real.sin α ^ 2 + Real.cos (α + Real.pi / 6) ^ 2 + Real.sin α * Real.cos (α + Real.pi / 6) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1015_101511


namespace NUMINAMATH_CALUDE_distance_to_origin_problem_l1015_101544

theorem distance_to_origin_problem (a : ℝ) : 
  (|a| = 2) → (a - 2 = 0 ∨ a - 2 = -4) := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_problem_l1015_101544


namespace NUMINAMATH_CALUDE_tricycle_count_l1015_101576

/-- Represents the number of wheels for each vehicle type -/
def wheels_per_vehicle : Fin 3 → ℕ
  | 0 => 2  -- bicycles
  | 1 => 3  -- tricycles
  | 2 => 2  -- scooters

/-- Proves that the number of tricycles is 4 given the conditions of the parade -/
theorem tricycle_count (vehicles : Fin 3 → ℕ) 
  (total_children : vehicles 0 + vehicles 1 + vehicles 2 = 10)
  (total_wheels : vehicles 0 * wheels_per_vehicle 0 + 
                  vehicles 1 * wheels_per_vehicle 1 + 
                  vehicles 2 * wheels_per_vehicle 2 = 27) :
  vehicles 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_tricycle_count_l1015_101576


namespace NUMINAMATH_CALUDE_total_fault_movement_total_movement_is_17_25_l1015_101582

/-- Represents the movement of a fault line over two years -/
structure FaultMovement where
  pastYear : Float
  yearBefore : Float

/-- Calculates the total movement of a fault line over two years -/
def totalMovement (fault : FaultMovement) : Float :=
  fault.pastYear + fault.yearBefore

/-- Theorem: The total movement of all fault lines is the sum of their individual movements -/
theorem total_fault_movement (faultA faultB faultC : FaultMovement) :
  totalMovement faultA + totalMovement faultB + totalMovement faultC =
  faultA.pastYear + faultA.yearBefore +
  faultB.pastYear + faultB.yearBefore +
  faultC.pastYear + faultC.yearBefore := by
  sorry

/-- Given fault movements -/
def faultA : FaultMovement := { pastYear := 1.25, yearBefore := 5.25 }
def faultB : FaultMovement := { pastYear := 2.5, yearBefore := 3.0 }
def faultC : FaultMovement := { pastYear := 0.75, yearBefore := 4.5 }

/-- Theorem: The total movement of the given fault lines is 17.25 inches -/
theorem total_movement_is_17_25 :
  totalMovement faultA + totalMovement faultB + totalMovement faultC = 17.25 := by
  sorry

end NUMINAMATH_CALUDE_total_fault_movement_total_movement_is_17_25_l1015_101582


namespace NUMINAMATH_CALUDE_distribute_nine_computers_to_three_schools_l1015_101558

/-- The number of ways to distribute computers to schools -/
def distribute_computers (total_computers : ℕ) (num_schools : ℕ) (min_computers : ℕ) : ℕ :=
  -- The actual implementation is not provided here
  sorry

/-- Theorem: There are 10 ways to distribute 9 computers to 3 schools with at least 2 per school -/
theorem distribute_nine_computers_to_three_schools : 
  distribute_computers 9 3 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_distribute_nine_computers_to_three_schools_l1015_101558


namespace NUMINAMATH_CALUDE_loop_structure_and_body_l1015_101561

/-- Represents an algorithmic structure -/
structure AlgorithmicStructure where
  repeatedExecution : Bool
  conditionalExecution : Bool

/-- Represents a processing step in an algorithm -/
structure ProcessingStep where
  isRepeated : Bool

/-- Definition of a loop structure -/
def isLoopStructure (s : AlgorithmicStructure) : Prop :=
  s.repeatedExecution ∧ s.conditionalExecution

/-- Definition of a loop body -/
def isLoopBody (p : ProcessingStep) : Prop :=
  p.isRepeated

/-- Theorem stating the relationship between loop structures and loop bodies -/
theorem loop_structure_and_body 
    (s : AlgorithmicStructure) 
    (p : ProcessingStep) 
    (h1 : s.repeatedExecution) 
    (h2 : s.conditionalExecution) 
    (h3 : p.isRepeated) : 
  isLoopStructure s ∧ isLoopBody p := by
  sorry


end NUMINAMATH_CALUDE_loop_structure_and_body_l1015_101561


namespace NUMINAMATH_CALUDE_min_value_a_solution_set_l1015_101500

-- Define the function f(x)
def f (x a : ℝ) : ℝ := |x - 4| + |x - a|

-- Theorem for the minimum value of a
theorem min_value_a :
  ∃ (a : ℝ), ∀ (x : ℝ), f x a ≥ a ∧ (∃ (x₀ : ℝ), f x₀ a = a) ∧ a = 2 :=
sorry

-- Theorem for the solution set of f(x) ≤ 5
theorem solution_set :
  ∃ (a : ℝ), a = 2 ∧ {x : ℝ | f x a ≤ 5} = {x : ℝ | 1/2 ≤ x ∧ x ≤ 11/2} :=
sorry

end NUMINAMATH_CALUDE_min_value_a_solution_set_l1015_101500


namespace NUMINAMATH_CALUDE_additional_cars_problem_solution_l1015_101551

theorem additional_cars (front_initial : Nat) (back_initial : Nat) (total_end : Nat) : Nat :=
  let total_initial := front_initial + back_initial
  total_end - total_initial

theorem problem_solution : 
  let front_initial := 100
  let back_initial := 2 * front_initial
  let total_end := 700
  additional_cars front_initial back_initial total_end = 400 := by
sorry

end NUMINAMATH_CALUDE_additional_cars_problem_solution_l1015_101551


namespace NUMINAMATH_CALUDE_smallest_perimeter_600_smallest_perimeter_144_l1015_101564

/-- A triangle with integer side lengths -/
structure IntTriangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

/-- The perimeter of a triangle -/
def perimeter (t : IntTriangle) : ℕ := t.a + t.b + t.c

/-- The product of side lengths of a triangle -/
def product (t : IntTriangle) : ℕ := t.a * t.b * t.c

theorem smallest_perimeter_600 :
  ∀ t : IntTriangle, product t = 600 →
  perimeter t ≥ perimeter ⟨10, 10, 6, sorry⟩ := by sorry

theorem smallest_perimeter_144 :
  ∀ t : IntTriangle, product t = 144 →
  perimeter t ≥ perimeter ⟨4, 6, 6, sorry⟩ := by sorry

end NUMINAMATH_CALUDE_smallest_perimeter_600_smallest_perimeter_144_l1015_101564


namespace NUMINAMATH_CALUDE_repeating_decimal_difference_l1015_101548

/-- Proves that the difference between the repeating decimals 0.353535... and 0.777777... is equal to -14/33 -/
theorem repeating_decimal_difference : 
  (35 : ℚ) / 99 - (7 : ℚ) / 9 = -14 / 33 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_difference_l1015_101548


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1015_101503

theorem quadratic_equation_roots : ∃! x : ℝ, x^2 - 4*x + 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1015_101503


namespace NUMINAMATH_CALUDE_u_n_eq_2n_minus_1_l1015_101552

/-- 
Given a positive integer n, u_n is the smallest positive integer such that 
for any odd integer d, the number of integers in any u_n consecutive odd integers 
that are divisible by d is at least as many as the number of integers among 
1, 3, 5, ..., 2n-1 that are divisible by d.
-/
def u_n (n : ℕ+) : ℕ := sorry

/-- The main theorem stating that u_n is equal to 2n - 1 -/
theorem u_n_eq_2n_minus_1 (n : ℕ+) : u_n n = 2 * n - 1 := by sorry

end NUMINAMATH_CALUDE_u_n_eq_2n_minus_1_l1015_101552


namespace NUMINAMATH_CALUDE_x_squared_minus_5x_is_quadratic_l1015_101532

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 - 5x = 0 -/
def f (x : ℝ) : ℝ := x^2 - 5*x

/-- Theorem: x^2 - 5x = 0 is a quadratic equation -/
theorem x_squared_minus_5x_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_5x_is_quadratic_l1015_101532


namespace NUMINAMATH_CALUDE_marble_selection_ways_l1015_101583

theorem marble_selection_ways (total_marbles : ℕ) (special_marbles : ℕ) (selection_size : ℕ) 
  (h1 : total_marbles = 15)
  (h2 : special_marbles = 6)
  (h3 : selection_size = 5) :
  (special_marbles : ℕ) * Nat.choose (total_marbles - special_marbles) (selection_size - 1) = 756 := by
  sorry

end NUMINAMATH_CALUDE_marble_selection_ways_l1015_101583


namespace NUMINAMATH_CALUDE_average_difference_l1015_101510

theorem average_difference (x y z w : ℝ) : 
  (x + y + z) / 3 = (y + z + w) / 3 + 10 → w = x - 30 := by
sorry

end NUMINAMATH_CALUDE_average_difference_l1015_101510


namespace NUMINAMATH_CALUDE_total_winter_clothing_l1015_101533

def scarves_boxes : ℕ := 4
def scarves_per_box : ℕ := 8
def mittens_boxes : ℕ := 3
def mittens_per_box : ℕ := 6
def hats_boxes : ℕ := 2
def hats_per_box : ℕ := 5
def jackets_boxes : ℕ := 1
def jackets_per_box : ℕ := 3

theorem total_winter_clothing :
  scarves_boxes * scarves_per_box +
  mittens_boxes * mittens_per_box +
  hats_boxes * hats_per_box +
  jackets_boxes * jackets_per_box = 63 := by
sorry

end NUMINAMATH_CALUDE_total_winter_clothing_l1015_101533


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_one_second_l1015_101596

-- Define the height function
def h (t : ℝ) : ℝ := -4.9 * t^2 + 4.8 * t + 11

-- Define the velocity function as the derivative of the height function
def v (t : ℝ) : ℝ := -9.8 * t + 4.8

-- Theorem statement
theorem instantaneous_velocity_at_one_second :
  v 1 = -5 := by sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_one_second_l1015_101596


namespace NUMINAMATH_CALUDE_find_x_l1015_101523

theorem find_x (x y z : ℝ) 
  (hxy : x * y / (x + y) = 4)
  (hxz : x * z / (x + z) = 9)
  (hyz : y * z / (y + z) = 16)
  (hpos : x > 0 ∧ y > 0 ∧ z > 0)
  (hdist : x ≠ y ∧ y ≠ z ∧ x ≠ z) :
  x = 384 / 21 := by
sorry

end NUMINAMATH_CALUDE_find_x_l1015_101523


namespace NUMINAMATH_CALUDE_camera_profit_difference_l1015_101529

/-- Calculates the difference in profit between two camera sellers --/
theorem camera_profit_difference 
  (maddox_cameras : ℕ) (maddox_buy_price : ℚ) (maddox_sell_price : ℚ)
  (maddox_shipping : ℚ) (maddox_listing_fee : ℚ)
  (theo_cameras : ℕ) (theo_buy_price : ℚ) (theo_sell_price : ℚ)
  (theo_shipping : ℚ) (theo_listing_fee : ℚ)
  (h1 : maddox_cameras = 10)
  (h2 : maddox_buy_price = 35)
  (h3 : maddox_sell_price = 50)
  (h4 : maddox_shipping = 2)
  (h5 : maddox_listing_fee = 10)
  (h6 : theo_cameras = 15)
  (h7 : theo_buy_price = 30)
  (h8 : theo_sell_price = 40)
  (h9 : theo_shipping = 3)
  (h10 : theo_listing_fee = 15) :
  (maddox_cameras : ℚ) * maddox_sell_price - 
  (maddox_cameras : ℚ) * maddox_buy_price - 
  (maddox_cameras : ℚ) * maddox_shipping - 
  maddox_listing_fee -
  (theo_cameras : ℚ) * theo_sell_price + 
  (theo_cameras : ℚ) * theo_buy_price + 
  (theo_cameras : ℚ) * theo_shipping + 
  theo_listing_fee = 30 :=
by sorry

end NUMINAMATH_CALUDE_camera_profit_difference_l1015_101529


namespace NUMINAMATH_CALUDE_function_nature_l1015_101501

theorem function_nature (n : ℕ) (h : 30 * n = 30 * n) :
  let f : ℝ → ℝ := fun x ↦ x ^ n
  (f 1)^2 + (f (-1))^2 = 2 * ((f 1) + (f (-1)) - 1) →
  ∀ x : ℝ, f (-x) = f x :=
by sorry

end NUMINAMATH_CALUDE_function_nature_l1015_101501


namespace NUMINAMATH_CALUDE_tangent_line_length_l1015_101597

-- Define the circle C
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 6*y + 9 = 0

-- Define the point P
def P : ℝ × ℝ := (1, 0)

-- Theorem statement
theorem tangent_line_length :
  ∃ (t : ℝ × ℝ), 
    circle_equation t.1 t.2 ∧ 
    (t.1 - P.1)^2 + (t.2 - P.2)^2 = 8 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_length_l1015_101597


namespace NUMINAMATH_CALUDE_difference_of_squares_l1015_101518

theorem difference_of_squares (x y : ℝ) 
  (sum_eq : x + y = 20) 
  (diff_eq : x - y = 4) : 
  x^2 - y^2 = 80 := by
sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1015_101518


namespace NUMINAMATH_CALUDE_y1_greater_than_y2_l1015_101553

/-- Given a linear function y = 8x - 1 and two points P₁(3, y₁) and P₂(2, y₂) on its graph,
    prove that y₁ > y₂. -/
theorem y1_greater_than_y2 (y₁ y₂ : ℝ) : y₁ > y₂ :=
  by
  -- Define the linear function
  have h1 : ∀ x y, y = 8 * x - 1 → (x, y) ∈ {(x, y) | y = 8 * x - 1} := by sorry
  
  -- P₁(3, y₁) lies on the graph
  have h2 : (3, y₁) ∈ {(x, y) | y = 8 * x - 1} := by sorry
  
  -- P₂(2, y₂) lies on the graph
  have h3 : (2, y₂) ∈ {(x, y) | y = 8 * x - 1} := by sorry
  
  sorry -- Proof goes here

end NUMINAMATH_CALUDE_y1_greater_than_y2_l1015_101553


namespace NUMINAMATH_CALUDE_fuel_usage_proof_l1015_101560

theorem fuel_usage_proof (x : ℝ) : 
  x > 0 ∧ x + 0.8 * x = 27 → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_fuel_usage_proof_l1015_101560


namespace NUMINAMATH_CALUDE_area_of_inscribed_rectangle_l1015_101559

/-- Rectangle ABCD inscribed in triangle EFG with the following properties:
    - Side AD of the rectangle is on side EG of the triangle
    - Triangle's altitude from F to side EG is 7 inches
    - EG = 10 inches
    - Length of segment AB is equal to half the length of segment AD -/
structure InscribedRectangle where
  EG : ℝ
  altitude : ℝ
  AB : ℝ
  AD : ℝ
  h_EG : EG = 10
  h_altitude : altitude = 7
  h_AB_AD : AB = AD / 2

/-- The area of the inscribed rectangle ABCD is 1225/72 square inches -/
theorem area_of_inscribed_rectangle (rect : InscribedRectangle) :
  rect.AB * rect.AD = 1225 / 72 := by
  sorry

end NUMINAMATH_CALUDE_area_of_inscribed_rectangle_l1015_101559


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1015_101581

theorem inequality_solution_set (x : ℝ) :
  (Set.Ioo 2 3 : Set ℝ) = {x | (x - 2) * (x - 3) / (x^2 + 1) < 0} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1015_101581


namespace NUMINAMATH_CALUDE_percentage_division_equality_l1015_101506

theorem percentage_division_equality : 
  (208 / 100 * 1265) / 6 = 438.53333333333336 := by sorry

end NUMINAMATH_CALUDE_percentage_division_equality_l1015_101506


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l1015_101535

theorem quadratic_roots_problem (α β : ℝ) (h1 : α^2 - α - 2021 = 0)
                                         (h2 : β^2 - β - 2021 = 0)
                                         (h3 : α > β) : 
  let A := α^2 - 2*β^2 + 2*α*β + 3*β + 7
  ⌊A⌋ = -5893 := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l1015_101535


namespace NUMINAMATH_CALUDE_student_arrangement_count_l1015_101542

theorem student_arrangement_count : ℕ := by
  -- Define the total number of students
  let total_students : ℕ := 7
  
  -- Define the condition that A and B are adjacent
  let adjacent_pair : ℕ := 1
  
  -- Define the condition that C and D are not adjacent
  let non_adjacent_pair : ℕ := 2
  
  -- Define the number of entities to arrange after bundling A and B
  let entities : ℕ := total_students - adjacent_pair
  
  -- Define the number of gaps after arranging the entities
  let gaps : ℕ := entities + 1
  
  -- Calculate the total number of arrangements
  let arrangements : ℕ := 
    (Nat.factorial entities) *    -- Arrange entities
    (gaps * (gaps - 1)) *         -- Place C and D in gaps
    2                             -- Arrange A and B within their bundle
  
  -- Prove that the number of arrangements is 960
  sorry

end NUMINAMATH_CALUDE_student_arrangement_count_l1015_101542


namespace NUMINAMATH_CALUDE_angle_CDE_value_l1015_101587

-- Define the points
variable (A B C D E : Point)

-- Define the angles
variable (angleA angleB angleC angleAEB angleBED angleAED angleADE angleCDE : Real)

-- State the given conditions
axiom right_angles : angleA = 90 ∧ angleB = 90 ∧ angleC = 90
axiom angle_AEB : angleAEB = 50
axiom angle_BED : angleBED = 45
axiom isosceles_ADE : angleAED = angleADE

-- State the theorem to be proved
theorem angle_CDE_value : angleCDE = 112.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_CDE_value_l1015_101587


namespace NUMINAMATH_CALUDE_odd_function_half_period_zero_l1015_101520

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the smallest positive period T
variable (T : ℝ)

-- Define the oddness property of f
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the periodicity property of f with period T
def has_period (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x

-- Define that T is the smallest positive period
def is_smallest_positive_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  T > 0 ∧ has_period f T ∧ ∀ S, 0 < S ∧ S < T → ¬(has_period f S)

-- State the theorem
theorem odd_function_half_period_zero
  (h_odd : is_odd f)
  (h_period : is_smallest_positive_period f T) :
  f (-T/2) = 0 :=
sorry

end NUMINAMATH_CALUDE_odd_function_half_period_zero_l1015_101520


namespace NUMINAMATH_CALUDE_remainder_problem_l1015_101530

theorem remainder_problem (n : ℤ) (h : n % 11 = 4) : (4 * n - 9) % 11 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1015_101530


namespace NUMINAMATH_CALUDE_savings_calculation_l1015_101556

def folder_price : ℝ := 2.50
def num_folders : ℕ := 5
def discount_rate : ℝ := 0.20

theorem savings_calculation :
  let original_total := folder_price * num_folders
  let discounted_total := original_total * (1 - discount_rate)
  original_total - discounted_total = 2.50 := by
sorry

end NUMINAMATH_CALUDE_savings_calculation_l1015_101556


namespace NUMINAMATH_CALUDE_problem_solution_l1015_101570

theorem problem_solution : ∀ A B Y : ℤ,
  A = 3009 / 3 →
  B = A / 3 →
  Y = A - B →
  Y = 669 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1015_101570


namespace NUMINAMATH_CALUDE_coral_reef_number_conversion_l1015_101502

/-- Converts an octal number to decimal --/
def octal_to_decimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to hexadecimal --/
def decimal_to_hex (n : ℕ) : String := sorry

theorem coral_reef_number_conversion :
  let octal_num := 732
  let decimal_num := octal_to_decimal octal_num
  decimal_num = 474 ∧ decimal_to_hex decimal_num = "1DA" := by sorry

end NUMINAMATH_CALUDE_coral_reef_number_conversion_l1015_101502


namespace NUMINAMATH_CALUDE_line_through_two_points_l1015_101591

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The line equation derived from two points -/
def lineEquation (p₁ p₂ : Point) (p : Point) : Prop :=
  (p.x - p₁.x) * (p₂.y - p₁.y) = (p.y - p₁.y) * (p₂.x - p₁.x)

theorem line_through_two_points (p₁ p₂ : Point) (h : p₁ ≠ p₂) :
  ∃! l : Line, Point.onLine p₁ l ∧ Point.onLine p₂ l ∧
  ∀ p, Point.onLine p l ↔ lineEquation p₁ p₂ p :=
sorry

end NUMINAMATH_CALUDE_line_through_two_points_l1015_101591


namespace NUMINAMATH_CALUDE_curve_intersection_implies_a_equals_one_l1015_101557

/-- Curve C₁ in polar coordinates -/
def C₁ (ρ θ : ℝ) (a : ℝ) : Prop :=
  ρ^2 - 2*ρ*(Real.sin θ) + 1 - a^2 = 0 ∧ a > 0

/-- Curve C₂ in polar coordinates -/
def C₂ (ρ θ : ℝ) : Prop :=
  ρ = 4*(Real.cos θ)

/-- Line C₃ in polar coordinates -/
def C₃ (θ : ℝ) : Prop :=
  ∃ α₀, θ = α₀ ∧ Real.tan α₀ = 2

/-- Common points of C₁ and C₂ lie on C₃ -/
def common_points_on_C₃ (a : ℝ) : Prop :=
  ∀ ρ θ, C₁ ρ θ a ∧ C₂ ρ θ → C₃ θ

theorem curve_intersection_implies_a_equals_one :
  ∀ a, common_points_on_C₃ a → a = 1 :=
sorry

end NUMINAMATH_CALUDE_curve_intersection_implies_a_equals_one_l1015_101557


namespace NUMINAMATH_CALUDE_factorial_fraction_l1015_101589

theorem factorial_fraction (N : ℕ) (h : N > 2) :
  (Nat.factorial (N - 2) * (N - 1)) / Nat.factorial N = 1 / N := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_l1015_101589


namespace NUMINAMATH_CALUDE_prob_one_white_one_black_l1015_101554

/-- The probability of drawing one white ball and one black ball in two draws -/
theorem prob_one_white_one_black (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) 
  (h1 : total_balls = white_balls + black_balls)
  (h2 : total_balls > 0)
  (h3 : white_balls = 7)
  (h4 : black_balls = 3) :
  (white_balls : ℚ) / total_balls * (black_balls : ℚ) / total_balls + 
  (black_balls : ℚ) / total_balls * (white_balls : ℚ) / total_balls = 
  (7 : ℚ) / 10 * (3 : ℚ) / 10 + (3 : ℚ) / 10 * (7 : ℚ) / 10 :=
sorry

end NUMINAMATH_CALUDE_prob_one_white_one_black_l1015_101554


namespace NUMINAMATH_CALUDE_train_speed_l1015_101509

/-- The speed of a train crossing a bridge -/
theorem train_speed (train_length bridge_length time_to_cross : ℝ) 
  (h1 : train_length = 110)
  (h2 : bridge_length = 170)
  (h3 : time_to_cross = 13.998880089592832) :
  (train_length + bridge_length) / time_to_cross = 20.0014286607 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1015_101509


namespace NUMINAMATH_CALUDE_range_of_a_l1015_101588

def p (x : ℝ) := |4*x - 3| ≤ 1

def q (x a : ℝ) := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

theorem range_of_a :
  (∀ x, q x a → p x) ∧
  (∃ x, p x ∧ ¬q x a) →
  a ∈ Set.Icc (0 : ℝ) (1/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1015_101588


namespace NUMINAMATH_CALUDE_triangle_side_sum_l1015_101545

theorem triangle_side_sum (a b c : ℝ) (h1 : a + b + c = 180) 
  (h2 : a = 60) (h3 : b = 30) (h4 : c = 90) (h5 : 9 = a.sin * 18) :
  18 + 9 * Real.sqrt 3 = 18 + b.sin * 18 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_sum_l1015_101545


namespace NUMINAMATH_CALUDE_percentage_difference_l1015_101575

theorem percentage_difference (A B C : ℝ) (hpos : 0 < C ∧ C < B ∧ B < A) :
  let x := 100 * (A - B) / B
  let y := 100 * (A - C) / C
  (∃ k, A = B * (1 + k/100) → x = 100 * (A - B) / B) ∧
  (∃ m, A = C * (1 + m/100) → y = 100 * (A - C) / C) := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l1015_101575


namespace NUMINAMATH_CALUDE_baker_cupcake_distribution_l1015_101515

/-- The number of cupcakes left over when distributing cupcakes equally -/
def cupcakes_left_over (total : ℕ) (children : ℕ) : ℕ :=
  total % children

/-- Theorem: When distributing 17 cupcakes among 3 children equally, 2 cupcakes are left over -/
theorem baker_cupcake_distribution :
  cupcakes_left_over 17 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_baker_cupcake_distribution_l1015_101515


namespace NUMINAMATH_CALUDE_best_of_three_match_probability_l1015_101504

/-- The probability of player A winning a single set -/
def p : ℝ := 0.6

/-- The probability of player A winning the match in a best-of-three format -/
def prob_A_wins_match : ℝ := p^2 + 2 * p^2 * (1 - p)

theorem best_of_three_match_probability :
  prob_A_wins_match = 0.648 := by
  sorry

end NUMINAMATH_CALUDE_best_of_three_match_probability_l1015_101504


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1015_101590

theorem inequality_solution_range (k : ℝ) : 
  (1 : ℝ)^2 * k^2 - 6 * k * (1 : ℝ) + 8 ≥ 0 → k ≤ 2 ∨ k ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1015_101590
