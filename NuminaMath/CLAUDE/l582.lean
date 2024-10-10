import Mathlib

namespace number_of_trucks_l582_58223

/-- The number of trucks used in transportation -/
def x : ℕ := sorry

/-- The total profit from Qingxi to Shenzhen in yuan -/
def total_profit : ℕ := 11560

/-- The profit per truck from Qingxi to Guangzhou in yuan -/
def profit_qingxi_guangzhou : ℕ := 480

/-- The initial profit per truck from Guangzhou to Shenzhen in yuan -/
def initial_profit_guangzhou_shenzhen : ℕ := 520

/-- The decrease in profit for each additional truck in yuan -/
def profit_decrease : ℕ := 20

/-- The profit from Guangzhou to Shenzhen as a function of the number of trucks -/
def profit_guangzhou_shenzhen (n : ℕ) : ℤ :=
  initial_profit_guangzhou_shenzhen * n - profit_decrease * (n - 1)

theorem number_of_trucks : x = 10 := by
  have h1 : profit_qingxi_guangzhou * x + profit_guangzhou_shenzhen x = total_profit := by sorry
  sorry

end number_of_trucks_l582_58223


namespace philosophers_more_numerous_than_mathematicians_l582_58200

theorem philosophers_more_numerous_than_mathematicians 
  (x : ℕ) -- x represents the number of people who are both mathematicians and philosophers
  (h_positive : x > 0) -- assumption that at least one person belongs to either group
  : 9 * x > 7 * x := by
  sorry

end philosophers_more_numerous_than_mathematicians_l582_58200


namespace exponent_multiplication_l582_58240

theorem exponent_multiplication (a : ℝ) : a^4 * a^3 = a^7 := by
  sorry

end exponent_multiplication_l582_58240


namespace inscribed_circle_radius_l582_58276

theorem inscribed_circle_radius (r : ℝ) :
  r > 0 →
  ∃ (R : ℝ), R > 0 ∧ R = 4 →
  r + r * Real.sqrt 2 = R →
  r = 4 * Real.sqrt 2 - 4 :=
by sorry

end inscribed_circle_radius_l582_58276


namespace bryans_bookshelves_l582_58272

/-- Given that Bryan has 42 books in total and each bookshelf contains 2 books,
    prove that the number of bookshelves he has is 21. -/
theorem bryans_bookshelves :
  let total_books : ℕ := 42
  let books_per_shelf : ℕ := 2
  let num_shelves : ℕ := total_books / books_per_shelf
  num_shelves = 21 := by
  sorry

end bryans_bookshelves_l582_58272


namespace rosemary_leaves_count_rosemary_leaves_solution_l582_58227

theorem rosemary_leaves_count : ℕ → Prop :=
  fun r : ℕ =>
    let basil_pots : ℕ := 3
    let rosemary_pots : ℕ := 9
    let thyme_pots : ℕ := 6
    let basil_leaves_per_plant : ℕ := 4
    let thyme_leaves_per_plant : ℕ := 30
    let total_leaves : ℕ := 354
    
    basil_pots * basil_leaves_per_plant + 
    rosemary_pots * r + 
    thyme_pots * thyme_leaves_per_plant = total_leaves →
    r = 18

theorem rosemary_leaves_solution : rosemary_leaves_count 18 := by
  sorry

end rosemary_leaves_count_rosemary_leaves_solution_l582_58227


namespace no_two_roots_in_interval_l582_58294

theorem no_two_roots_in_interval (a b c : ℝ) (ha : a > 0) (hcond : 12 * a + 5 * b + 2 * c > 0) :
  ¬∃ (x y : ℝ), 2 < x ∧ x < 3 ∧ 2 < y ∧ y < 3 ∧
  x ≠ y ∧
  a * x^2 + b * x + c = 0 ∧
  a * y^2 + b * y + c = 0 :=
by sorry

end no_two_roots_in_interval_l582_58294


namespace hikers_speed_hikers_speed_specific_l582_58292

/-- The problem of determining a hiker's speed given specific conditions involving a cyclist -/
theorem hikers_speed (cyclist_speed : ℝ) (cyclist_travel_time : ℝ) (hiker_catch_up_time : ℝ) : ℝ :=
  let hiker_speed := (cyclist_speed * cyclist_travel_time) / hiker_catch_up_time
  by
    -- Assuming:
    -- 1. The hiker walks at a constant rate.
    -- 2. A cyclist passes the hiker, traveling in the same direction at 'cyclist_speed'.
    -- 3. The cyclist stops after 'cyclist_travel_time'.
    -- 4. The hiker continues walking at her constant rate.
    -- 5. The cyclist waits 'hiker_catch_up_time' until the hiker catches up.
    
    -- Prove: hiker_speed = 20/3

    sorry

/-- The specific instance of the hiker's speed problem -/
theorem hikers_speed_specific : hikers_speed 20 (1/12) (1/4) = 20/3 :=
  by sorry

end hikers_speed_hikers_speed_specific_l582_58292


namespace arrangement_theorem_l582_58219

/-- The number of ways to arrange 9 distinct objects in a row with specific conditions -/
def arrangement_count : ℕ := 2880

/-- The total number of objects -/
def total_objects : ℕ := 9

/-- The number of objects that must be at the ends -/
def end_objects : ℕ := 2

/-- The number of objects that must be adjacent -/
def adjacent_objects : ℕ := 2

/-- The number of remaining objects -/
def remaining_objects : ℕ := total_objects - end_objects - adjacent_objects

theorem arrangement_theorem :
  arrangement_count = 
    2 * -- ways to arrange end objects
    (remaining_objects + 1) * -- ways to place adjacent objects
    2 * -- ways to arrange adjacent objects
    remaining_objects! -- ways to arrange remaining objects
  := by sorry

end arrangement_theorem_l582_58219


namespace bowling_tournament_orders_l582_58254

/-- A tournament structure with players and games. -/
structure Tournament :=
  (num_players : ℕ)
  (num_games : ℕ)
  (outcomes_per_game : ℕ)

/-- The number of possible prize distribution orders in a tournament. -/
def prize_distribution_orders (t : Tournament) : ℕ := t.outcomes_per_game ^ t.num_games

/-- The specific tournament described in the problem. -/
def bowling_tournament : Tournament :=
  { num_players := 6,
    num_games := 5,
    outcomes_per_game := 2 }

/-- Theorem stating that the number of possible prize distribution orders
    in the bowling tournament is 32. -/
theorem bowling_tournament_orders :
  prize_distribution_orders bowling_tournament = 32 := by
  sorry


end bowling_tournament_orders_l582_58254


namespace range_of_product_l582_58260

theorem range_of_product (a b : ℝ) (ha : |a| ≤ 1) (hab : |a + b| ≤ 1) :
  -2 ≤ (a + 1) * (b + 1) ∧ (a + 1) * (b + 1) ≤ 9/4 := by
  sorry

end range_of_product_l582_58260


namespace replacement_cost_theorem_l582_58214

/-- A rectangular plot with specific dimensions and fencing cost -/
structure RectangularPlot where
  short_side : ℝ
  long_side : ℝ
  perimeter : ℝ
  cost_per_foot : ℝ
  long_side_relation : long_side = 3 * short_side
  perimeter_equation : perimeter = 2 * short_side + 2 * long_side

/-- The cost to replace one short side of the fence -/
def replacement_cost (plot : RectangularPlot) : ℝ :=
  plot.short_side * plot.cost_per_foot

/-- Theorem stating the replacement cost for the given conditions -/
theorem replacement_cost_theorem (plot : RectangularPlot) 
  (h_perimeter : plot.perimeter = 640)
  (h_cost : plot.cost_per_foot = 5) :
  replacement_cost plot = 400 := by
  sorry

end replacement_cost_theorem_l582_58214


namespace union_of_sets_l582_58250

theorem union_of_sets : 
  let A : Set ℕ := {1, 2}
  let B : Set ℕ := {2, 3}
  A ∪ B = {1, 2, 3} := by
sorry

end union_of_sets_l582_58250


namespace stratified_sample_theorem_l582_58255

/-- Calculates the number of samples to be drawn from a subgroup in stratified sampling -/
def stratified_sample_size (total_population : ℕ) (total_sample_size : ℕ) (subgroup_size : ℕ) : ℕ :=
  (total_sample_size * subgroup_size) / total_population

/-- Theorem: In a stratified sampling scenario with a total population of 1000 and a sample size of 50,
    the number of samples drawn from a subgroup of 200 is equal to 10 -/
theorem stratified_sample_theorem :
  stratified_sample_size 1000 50 200 = 10 := by
  sorry

end stratified_sample_theorem_l582_58255


namespace pencil_length_theorem_l582_58220

def pencil_length_after_sharpening (original_length sharpened_off : ℕ) : ℕ :=
  original_length - sharpened_off

theorem pencil_length_theorem (original_length sharpened_off : ℕ) 
  (h1 : original_length = 31)
  (h2 : sharpened_off = 17) :
  pencil_length_after_sharpening original_length sharpened_off = 14 := by
sorry

end pencil_length_theorem_l582_58220


namespace subcommittee_formation_ways_l582_58216

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem subcommittee_formation_ways :
  let total_republicans : ℕ := 10
  let total_democrats : ℕ := 8
  let subcommittee_republicans : ℕ := 4
  let subcommittee_democrats : ℕ := 3
  (choose total_republicans subcommittee_republicans) *
  (choose total_democrats subcommittee_democrats) = 11760 := by
  sorry

end subcommittee_formation_ways_l582_58216


namespace infinite_sum_equality_l582_58245

/-- Given a positive real number t satisfying t^3 + 3/7*t - 1 = 0,
    the infinite sum t^3 + 2t^6 + 3t^9 + 4t^12 + ... equals (49/9)*t -/
theorem infinite_sum_equality (t : ℝ) (ht : t > 0) (heq : t^3 + 3/7*t - 1 = 0) :
  ∑' n, (n : ℝ) * t^(3*n) = 49/9 * t := by
  sorry

end infinite_sum_equality_l582_58245


namespace calculation_proof_l582_58241

theorem calculation_proof : 2^2 - Real.tan (60 * π / 180) + |Real.sqrt 3 - 1| - (3 - Real.pi)^0 = 2 := by
  sorry

end calculation_proof_l582_58241


namespace rectangular_field_area_l582_58248

/-- Represents a rectangular field -/
structure RectangularField where
  width : ℝ
  length : ℝ

/-- Calculates the perimeter of a rectangular field -/
def perimeter (field : RectangularField) : ℝ :=
  2 * (field.width + field.length)

/-- Calculates the area of a rectangular field -/
def area (field : RectangularField) : ℝ :=
  field.width * field.length

/-- Theorem: The area of a rectangular field with perimeter 120 meters and width 20 meters is 800 square meters -/
theorem rectangular_field_area :
  ∀ (field : RectangularField),
    field.width = 20 →
    perimeter field = 120 →
    area field = 800 := by
  sorry

end rectangular_field_area_l582_58248


namespace dad_steps_l582_58235

/-- Represents the number of steps taken by each person -/
structure Steps where
  dad : ℕ
  masha : ℕ
  yasha : ℕ

/-- Defines the relationship between steps taken by Dad and Masha -/
def dad_masha_ratio (s : Steps) : Prop :=
  5 * s.dad = 3 * s.masha

/-- Defines the relationship between steps taken by Masha and Yasha -/
def masha_yasha_ratio (s : Steps) : Prop :=
  5 * s.masha = 3 * s.yasha

/-- States that Masha and Yasha together took 400 steps -/
def total_masha_yasha (s : Steps) : Prop :=
  s.masha + s.yasha = 400

/-- Theorem stating that given the conditions, Dad took 90 steps -/
theorem dad_steps :
  ∀ s : Steps,
  dad_masha_ratio s →
  masha_yasha_ratio s →
  total_masha_yasha s →
  s.dad = 90 :=
by
  sorry


end dad_steps_l582_58235


namespace true_discount_calculation_l582_58279

/-- Calculates the true discount given the banker's gain, interest rate, and time period. -/
def true_discount (bankers_gain : ℚ) (interest_rate : ℚ) (time : ℚ) : ℚ :=
  (bankers_gain * 100) / (interest_rate * time)

/-- Theorem stating that under the given conditions, the true discount is 55. -/
theorem true_discount_calculation :
  let bankers_gain : ℚ := 6.6
  let interest_rate : ℚ := 12
  let time : ℚ := 1
  true_discount bankers_gain interest_rate time = 55 := by
sorry

end true_discount_calculation_l582_58279


namespace stamp_redistribution_l582_58290

/-- Represents the stamp redistribution problem -/
theorem stamp_redistribution (
  initial_albums : ℕ)
  (initial_pages_per_album : ℕ)
  (initial_stamps_per_page : ℕ)
  (new_stamps_per_page : ℕ)
  (filled_albums : ℕ)
  (h1 : initial_albums = 10)
  (h2 : initial_pages_per_album = 50)
  (h3 : initial_stamps_per_page = 7)
  (h4 : new_stamps_per_page = 12)
  (h5 : filled_albums = 6) :
  (initial_albums * initial_pages_per_album * initial_stamps_per_page) % new_stamps_per_page = 8 := by
  sorry

#check stamp_redistribution

end stamp_redistribution_l582_58290


namespace fraction_equals_zero_l582_58251

theorem fraction_equals_zero (x : ℝ) : 
  (x - 5) / (6 * x + 12) = 0 ↔ x = 5 ∧ 6 * x + 12 ≠ 0 :=
by sorry

end fraction_equals_zero_l582_58251


namespace negative_max_inverse_is_max_of_negative_inverses_l582_58274

/-- Given a non-empty set A of real numbers not containing zero,
    with a negative maximum value a, -a⁻¹ is the maximum value
    of the set {-x⁻¹ | x ∈ A}. -/
theorem negative_max_inverse_is_max_of_negative_inverses
  (A : Set ℝ)
  (hA_nonempty : A.Nonempty)
  (hA_no_zero : 0 ∉ A)
  (a : ℝ)
  (ha_max : ∀ x ∈ A, x ≤ a)
  (ha_neg : a < 0) :
  ∀ y ∈ {-x⁻¹ | x ∈ A}, y ≤ -a⁻¹ :=
by sorry

end negative_max_inverse_is_max_of_negative_inverses_l582_58274


namespace sum_of_continuity_points_l582_58267

/-- Piecewise function f(x) defined by n -/
noncomputable def f (n : ℝ) (x : ℝ) : ℝ :=
  if x < n then x^2 + 2*x + 3 else 3*x + 6

/-- Theorem stating that the sum of all values of n that make f(x) continuous is 2 -/
theorem sum_of_continuity_points (n : ℝ) :
  (∀ x : ℝ, ContinuousAt (f n) x) →
  (∃ n₁ n₂ : ℝ, n₁ ≠ n₂ ∧ 
    (∀ x : ℝ, ContinuousAt (f n₁) x) ∧ 
    (∀ x : ℝ, ContinuousAt (f n₂) x) ∧
    n₁ + n₂ = 2) :=
by sorry

end sum_of_continuity_points_l582_58267


namespace union_of_A_and_B_l582_58293

def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | 1 < x ∧ x < 4}

theorem union_of_A_and_B : A ∪ B = {x | x > 1} := by sorry

end union_of_A_and_B_l582_58293


namespace restaurant_bill_proof_l582_58246

theorem restaurant_bill_proof (total_friends : Nat) (paying_friends : Nat) (extra_payment : ℚ) : 
  total_friends = 10 →
  paying_friends = 8 →
  extra_payment = 3 →
  ∃ (total_bill : ℚ), total_bill = 120 ∧ 
    paying_friends * (total_bill / total_friends + extra_payment) = total_bill :=
by sorry

end restaurant_bill_proof_l582_58246


namespace mary_towel_count_l582_58239

/-- Proves that Mary has 4 towels given the conditions of the problem --/
theorem mary_towel_count :
  ∀ (mary_towel_count frances_towel_count : ℕ)
    (total_weight mary_towel_weight frances_towel_weight : ℚ),
  mary_towel_count = 4 * frances_towel_count →
  total_weight = 60 →
  frances_towel_weight = 128 / 16 →
  total_weight = mary_towel_weight + frances_towel_weight →
  mary_towel_weight = mary_towel_count * (frances_towel_weight / frances_towel_count) →
  mary_towel_count = 4 :=
by
  sorry


end mary_towel_count_l582_58239


namespace arithmetic_sequence_property_l582_58273

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) (h : ArithmeticSequence a) 
    (h1 : a 4 + a 8 = -2) : 
  a 6 * (a 2 + 2 * a 6 + a 10) = 4 := by
  sorry

end arithmetic_sequence_property_l582_58273


namespace minutes_in_three_and_half_hours_l582_58252

/-- The number of minutes in one hour -/
def minutes_per_hour : ℕ := 60

/-- The number of hours -/
def hours : ℚ := 3.5

/-- Theorem: The number of minutes in 3.5 hours is 210 -/
theorem minutes_in_three_and_half_hours : 
  (hours * minutes_per_hour : ℚ) = 210 := by sorry

end minutes_in_three_and_half_hours_l582_58252


namespace x_value_proof_l582_58257

theorem x_value_proof (x y z a b d : ℝ) 
  (h1 : x * y / (x + y) = a) 
  (h2 : x * z / (x + z) = b) 
  (h3 : y * z / (y - z) = d) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hd : d ≠ 0) : 
  x = a * b / (a + b) := by
sorry

end x_value_proof_l582_58257


namespace first_digit_of_87_base_5_l582_58280

def base_5_representation (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

def first_digit_base_5 (n : ℕ) : ℕ :=
  match base_5_representation n with
  | [] => 0  -- This case should never occur for a valid input
  | d::_ => d

theorem first_digit_of_87_base_5 :
  first_digit_base_5 87 = 3 := by
  sorry

end first_digit_of_87_base_5_l582_58280


namespace quadratic_linear_common_solution_l582_58202

theorem quadratic_linear_common_solution
  (a d : ℝ) (x₁ x₂ e : ℝ) 
  (ha : a ≠ 0)
  (hd : d ≠ 0)
  (hx : x₁ ≠ x₂)
  (h_common : d * x₁ + e = 0)
  (h_unique : ∃! x, a * (x - x₁) * (x - x₂) + d * x + e = 0) :
  a * (x₂ - x₁) = d := by
sorry

end quadratic_linear_common_solution_l582_58202


namespace union_of_sets_l582_58262

open Set

theorem union_of_sets (A B : Set ℝ) : 
  A = {x : ℝ | -1 < x ∧ x < 3} → 
  B = {x : ℝ | x ≥ 1} → 
  A ∪ B = {x : ℝ | x > -1} := by
  sorry

end union_of_sets_l582_58262


namespace train_travel_time_l582_58296

/-- Proves that a train traveling at 120 kmph for 80 km takes 40 minutes -/
theorem train_travel_time (speed : ℝ) (distance : ℝ) (time : ℝ) :
  speed = 120 →
  distance = 80 →
  time = distance / speed * 60 →
  time = 40 := by
  sorry

end train_travel_time_l582_58296


namespace month_days_l582_58264

theorem month_days (days_taken : ℕ) (days_forgotten : ℕ) : 
  days_taken = 27 → days_forgotten = 4 → days_taken + days_forgotten = 31 := by
sorry

end month_days_l582_58264


namespace completing_square_result_l582_58229

theorem completing_square_result (x : ℝ) :
  x^2 - 4*x - 1 = 0 → (x - 2)^2 = 5 :=
by
  sorry

end completing_square_result_l582_58229


namespace sons_age_l582_58230

theorem sons_age (son_age father_age : ℕ) : 
  father_age = son_age + 18 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 16 := by
sorry

end sons_age_l582_58230


namespace modular_congruence_existence_l582_58247

theorem modular_congruence_existence (a c : ℕ) (b : ℤ) :
  ∃ x : ℕ, (a ^ x + x : ℤ) ≡ b [ZMOD c] := by
  sorry

end modular_congruence_existence_l582_58247


namespace cubic_sum_over_product_l582_58242

theorem cubic_sum_over_product (x y z : ℂ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 30)
  (h_eq : (x - y)^2 + (y - z)^2 + (z - x)^2 = 2*x*y*z) :
  (x^3 + y^3 + z^3) / (x*y*z) = 33 := by
sorry

end cubic_sum_over_product_l582_58242


namespace prob_at_least_one_prob_exactly_one_l582_58213

/-- Probability of event A occurring -/
def probA : ℚ := 4/5

/-- Probability of event B occurring -/
def probB : ℚ := 3/5

/-- Probability of event C occurring -/
def probC : ℚ := 2/5

/-- Events A, B, and C are independent -/
axiom independence : True

/-- Probability of at least one event occurring -/
theorem prob_at_least_one : 
  1 - (1 - probA) * (1 - probB) * (1 - probC) = 119/125 := by sorry

/-- Probability of exactly one event occurring -/
theorem prob_exactly_one :
  probA * (1 - probB) * (1 - probC) + 
  (1 - probA) * probB * (1 - probC) + 
  (1 - probA) * (1 - probB) * probC = 37/125 := by sorry

end prob_at_least_one_prob_exactly_one_l582_58213


namespace lighting_power_increase_l582_58238

/-- Proves that the increase in lighting power is 60 BT given the initial and final power values. -/
theorem lighting_power_increase (N_before N_after : ℝ) 
  (h1 : N_before = 240)
  (h2 : N_after = 300) :
  N_after - N_before = 60 := by
  sorry

end lighting_power_increase_l582_58238


namespace parabola_f_value_l582_58226

/-- A parabola with equation y = dx^2 + ex + f -/
structure Parabola where
  d : ℝ
  e : ℝ
  f : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_coord (p : Parabola) (x : ℝ) : ℝ :=
  p.d * x^2 + p.e * x + p.f

theorem parabola_f_value (p : Parabola) :
  p.y_coord (-1) = -2 →  -- vertex condition
  p.y_coord 0 = -1.5 →   -- point condition
  p.f = -1.5 := by
sorry

end parabola_f_value_l582_58226


namespace apple_sales_proof_l582_58203

/-- The number of apples sold by Reginald --/
def apples_sold : ℕ := 20

/-- The price of each apple in dollars --/
def apple_price : ℚ := 1.25

/-- The cost of Reginald's bike in dollars --/
def bike_cost : ℚ := 80

/-- The fraction of the bike cost that the repairs cost --/
def repair_cost_fraction : ℚ := 1/4

/-- The fraction of earnings remaining after repairs --/
def remaining_earnings_fraction : ℚ := 1/5

theorem apple_sales_proof :
  apples_sold = 20 ∧
  apple_price = 1.25 ∧
  bike_cost = 80 ∧
  repair_cost_fraction = 1/4 ∧
  remaining_earnings_fraction = 1/5 ∧
  (apples_sold : ℚ) * apple_price - bike_cost * repair_cost_fraction = 
    remaining_earnings_fraction * ((apples_sold : ℚ) * apple_price) :=
by sorry

end apple_sales_proof_l582_58203


namespace max_volume_right_prism_l582_58288

/-- 
Given a right prism with triangular bases where:
- Base triangle sides are a, b, b
- a = 2b
- Angle between sides a and b is π/2
- Sum of areas of two lateral faces and one base is 30
The maximum volume of the prism is 2.5√5.
-/
theorem max_volume_right_prism (a b h : ℝ) :
  a = 2 * b →
  4 * b * h + b^2 = 30 →
  (∀ h' : ℝ, 4 * b * h' + b^2 = 30 → b^2 * h / 2 ≤ b^2 * h' / 2) →
  b^2 * h / 2 = 2.5 * Real.sqrt 5 :=
by sorry

end max_volume_right_prism_l582_58288


namespace girls_in_class_l582_58285

theorem girls_in_class (total : ℕ) (prob : ℚ) (boys : ℕ) (girls : ℕ) : 
  total = 25 →
  prob = 3 / 25 →
  boys + girls = total →
  (boys.choose 2 : ℚ) / (total.choose 2 : ℚ) = prob →
  girls = 16 :=
sorry

end girls_in_class_l582_58285


namespace practice_for_five_months_l582_58208

/-- Calculates the total piano practice hours over a given number of months -/
def total_practice_hours (weekly_hours : ℕ) (months : ℕ) : ℕ :=
  weekly_hours * 4 * months

/-- Theorem stating that practicing 4 hours weekly for 5 months results in 80 total hours -/
theorem practice_for_five_months : 
  total_practice_hours 4 5 = 80 := by
  sorry

end practice_for_five_months_l582_58208


namespace window_area_ratio_l582_58286

theorem window_area_ratio :
  let AB : ℝ := 36
  let AD : ℝ := AB * (5/3)
  let circle_area : ℝ := Real.pi * (AB/2)^2
  let rectangle_area : ℝ := AD * AB
  let square_area : ℝ := AB^2
  rectangle_area / (circle_area + square_area) = 2160 / (324 * Real.pi + 1296) :=
by sorry

end window_area_ratio_l582_58286


namespace sum_of_solutions_quadratic_l582_58277

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (36 - 18*x - x^2 = 0) → 
  (∃ r s : ℝ, (36 - 18*r - r^2 = 0) ∧ (36 - 18*s - s^2 = 0) ∧ (r + s = -18)) :=
by sorry

end sum_of_solutions_quadratic_l582_58277


namespace toothpick_pattern_l582_58244

/-- 
Given an arithmetic sequence where:
- The first term is 5
- The common difference is 4
Prove that the 250th term is 1001
-/
theorem toothpick_pattern (n : ℕ) (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) :
  n = 250 → a₁ = 5 → d = 4 → aₙ = a₁ + (n - 1) * d → aₙ = 1001 := by
  sorry

end toothpick_pattern_l582_58244


namespace percent_of_sixty_l582_58222

theorem percent_of_sixty : (25 : ℚ) / 100 * 60 = 15 := by
  sorry

end percent_of_sixty_l582_58222


namespace arithmetic_to_geometric_sequence_l582_58268

/-- A sequence with common difference -/
def ArithmeticSequence (a₁ d : ℝ) (n : ℕ) := fun i : ℕ => a₁ + d * (i : ℝ)

/-- Condition for a sequence to be geometric -/
def IsGeometric (s : ℕ → ℝ) := ∀ i j k, s i * s k = s j * s j

/-- Removing one term from a sequence -/
def RemoveTerm (s : ℕ → ℝ) (k : ℕ) := fun i : ℕ => if i < k then s i else s (i + 1)

theorem arithmetic_to_geometric_sequence 
  (n : ℕ) (a₁ d : ℝ) (hn : n ≥ 4) (hd : d ≠ 0) :
  (∃ k, IsGeometric (RemoveTerm (ArithmeticSequence a₁ d n) k)) ↔ 
  (n = 4 ∧ (a₁ / d = -4 ∨ a₁ / d = 1)) := by
  sorry

end arithmetic_to_geometric_sequence_l582_58268


namespace polynomial_multiplication_correction_l582_58249

theorem polynomial_multiplication_correction (x a b : ℚ) : 
  (2*x-a)*(3*x+b) = 6*x^2 + 11*x - 10 →
  (2*x+a)*(x+b) = 2*x^2 - 9*x + 10 →
  (2*x+a)*(3*x+b) = 6*x^2 - 19*x + 10 := by
sorry

end polynomial_multiplication_correction_l582_58249


namespace non_monotonic_interval_implies_k_range_l582_58261

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 12*x

-- Define the property of non-monotonicity in an interval
def not_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x y, a < x ∧ x < y ∧ y < b ∧ (f x < f y ∧ ∃ z, x < z ∧ z < y ∧ f z < f x)

-- State the theorem
theorem non_monotonic_interval_implies_k_range (k : ℝ) :
  not_monotonic f (k - 1) (k + 1) → (-3 < k ∧ k < -1) ∨ (1 < k ∧ k < 3) :=
sorry

end non_monotonic_interval_implies_k_range_l582_58261


namespace jim_distance_l582_58215

/-- Represents the distance covered by a person in a certain number of steps -/
structure StepDistance where
  steps : ℕ
  distance : ℝ

/-- Carly's step distance -/
def carly_step : ℝ := 0.5

/-- The relationship between Carly's and Jim's steps for the same distance -/
def step_ratio : ℚ := 3 / 4

/-- Number of Jim's steps we want to calculate the distance for -/
def jim_steps : ℕ := 24

/-- Theorem stating that Jim travels 9 metres in 24 steps -/
theorem jim_distance : 
  ∀ (carly : StepDistance) (jim : StepDistance),
  carly.steps = 3 ∧ 
  jim.steps = 4 ∧
  carly.distance = jim.distance ∧
  carly.distance = carly_step * carly.steps →
  (jim_steps : ℝ) * jim.distance / jim.steps = 9 := by
  sorry

end jim_distance_l582_58215


namespace max_value_abc_l582_58299

theorem max_value_abc (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 5)
  (hb : 0 ≤ b ∧ b ≤ 5)
  (hc : 0 ≤ c ∧ c ≤ 5)
  (h_sum : 2 * a + b + c = 10) :
  a + 2 * b + 3 * c ≤ 25 :=
by sorry

end max_value_abc_l582_58299


namespace smallest_integer_with_conditions_l582_58284

/-- Represents a natural number as a list of its digits in reverse order -/
def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  have : n / 10 < n := sorry
  (n % 10) :: digits (n / 10)

/-- Checks if the digits of a number are in strictly increasing order -/
def increasing_digits (n : ℕ) : Prop :=
  List.Pairwise (· < ·) (digits n)

/-- Calculates the sum of squares of digits of a number -/
def sum_of_squares_of_digits (n : ℕ) : ℕ :=
  (digits n).map (λ d => d * d) |> List.sum

/-- Calculates the product of digits of a number -/
def product_of_digits (n : ℕ) : ℕ :=
  (digits n).prod

/-- The main theorem -/
theorem smallest_integer_with_conditions :
  ∃ n : ℕ,
    (∀ m : ℕ, m < n →
      (sum_of_squares_of_digits m ≠ 85 ∨
       ¬increasing_digits m)) ∧
    sum_of_squares_of_digits n = 85 ∧
    increasing_digits n ∧
    product_of_digits n = 18 :=
sorry

end smallest_integer_with_conditions_l582_58284


namespace uncle_dave_nieces_l582_58271

theorem uncle_dave_nieces (ice_cream_per_niece : ℝ) (total_ice_cream : ℕ) 
  (h1 : ice_cream_per_niece = 143.0)
  (h2 : total_ice_cream = 1573) :
  (total_ice_cream : ℝ) / ice_cream_per_niece = 11 := by
  sorry

end uncle_dave_nieces_l582_58271


namespace no_simultaneous_cubes_l582_58265

theorem no_simultaneous_cubes (n : ℕ) : 
  ¬(∃ (a b : ℤ), (2^(n+1) - 1 = a^3) ∧ (2^(n-1) * (2^n - 1) = b^3)) := by
  sorry

end no_simultaneous_cubes_l582_58265


namespace square_cut_from_rectangle_l582_58243

theorem square_cut_from_rectangle (a b x : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : x > 0) (h4 : x ≤ min a b) :
  (2 * (a + b) + 2 * x = a * b) ∧ (a * b - x^2 = 2 * (a + b)) → x = 2 := by
  sorry

end square_cut_from_rectangle_l582_58243


namespace polynomial_sum_l582_58231

theorem polynomial_sum (a b x y : ℝ) 
  (h1 : a * x + b * y = 5)
  (h2 : a * x^2 + b * y^2 = 11)
  (h3 : a * x^3 + b * y^3 = 24)
  (h4 : a * x^4 + b * y^4 = 58) :
  a * x^5 + b * y^5 = 262.88 := by
  sorry

end polynomial_sum_l582_58231


namespace round_trip_distance_approx_l582_58266

/-- Represents the total distance traveled in John's round trip --/
def total_distance (city_speed outbound_highway_speed return_highway_speed : ℝ)
  (outbound_city_time outbound_highway_time return_highway_time1 return_highway_time2 return_city_time : ℝ) : ℝ :=
  let outbound_city_distance := city_speed * outbound_city_time
  let outbound_highway_distance := outbound_highway_speed * outbound_highway_time
  let return_highway_distance := return_highway_speed * (return_highway_time1 + return_highway_time2)
  let return_city_distance := city_speed * return_city_time
  outbound_city_distance + outbound_highway_distance + return_highway_distance + return_city_distance

/-- Theorem stating that the total round trip distance is approximately 166.67 km --/
theorem round_trip_distance_approx : 
  ∀ (ε : ℝ), ε > 0 → 
  ∃ (city_speed outbound_highway_speed return_highway_speed : ℝ)
    (outbound_city_time outbound_highway_time return_highway_time1 return_highway_time2 return_city_time : ℝ),
  city_speed = 40 ∧ 
  outbound_highway_speed = 80 ∧
  return_highway_speed = 100 ∧
  outbound_city_time = 1/3 ∧
  outbound_highway_time = 2/3 ∧
  return_highway_time1 = 1/2 ∧
  return_highway_time2 = 1/6 ∧
  return_city_time = 1/3 ∧
  |total_distance city_speed outbound_highway_speed return_highway_speed
    outbound_city_time outbound_highway_time return_highway_time1 return_highway_time2 return_city_time - 166.67| < ε :=
by
  sorry


end round_trip_distance_approx_l582_58266


namespace sum_of_roots_bounds_l582_58289

theorem sum_of_roots_bounds (x y z : ℝ) (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) 
  (h_sum : x^2 + y^2 + z^2 = 10) : 
  Real.sqrt 6 + Real.sqrt 2 ≤ Real.sqrt (6 - x^2) + Real.sqrt (6 - y^2) + Real.sqrt (6 - z^2) 
  ∧ Real.sqrt (6 - x^2) + Real.sqrt (6 - y^2) + Real.sqrt (6 - z^2) ≤ 2 * Real.sqrt 6 :=
by sorry

end sum_of_roots_bounds_l582_58289


namespace range_of_m_l582_58258

theorem range_of_m (x y m : ℝ) : 
  (2 * x + y = -4 * m + 5) →
  (x + 2 * y = m + 4) →
  (x - y > -6) →
  (x + y < 8) →
  (-5 < m ∧ m < 7/5) :=
by sorry

end range_of_m_l582_58258


namespace line_inclination_angle_l582_58236

theorem line_inclination_angle (x y : ℝ) :
  x + y - Real.sqrt 3 = 0 → ∃ θ : ℝ, θ = 135 * π / 180 ∧ Real.tan θ = -1 := by
  sorry

end line_inclination_angle_l582_58236


namespace fraction_equality_l582_58287

theorem fraction_equality : (5 * 7) / 8 = 4 + 3 / 8 := by
  sorry

end fraction_equality_l582_58287


namespace arithmetic_sequence_formula_l582_58211

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_formula 
  (a : ℕ → ℤ) 
  (h_arith : arithmetic_sequence a) 
  (h_3 : a 3 = -2) 
  (h_7 : a 7 = -10) : 
  ∀ n : ℕ, n > 0 → a n = -2 * n + 4 := by
sorry

end arithmetic_sequence_formula_l582_58211


namespace matrix_tripler_uniqueness_l582_58256

theorem matrix_tripler_uniqueness (A M : Matrix (Fin 2) (Fin 2) ℝ) :
  (∀ (i j : Fin 2), (M • A) i j = 3 * A i j) ↔ M = ![![3, 0], ![0, 3]] := by
sorry

end matrix_tripler_uniqueness_l582_58256


namespace gcd_of_four_numbers_l582_58297

theorem gcd_of_four_numbers : Nat.gcd 84 (Nat.gcd 108 (Nat.gcd 132 156)) = 12 := by
  sorry

end gcd_of_four_numbers_l582_58297


namespace complex_expansion_l582_58234

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_expansion : i * (1 + i)^2 = -2 := by
  sorry

end complex_expansion_l582_58234


namespace roses_to_tulips_ratio_l582_58201

/-- Represents the number of flowers of each type in the shop -/
structure FlowerShop where
  carnations : ℕ
  violets : ℕ
  tulips : ℕ
  roses : ℕ

/-- Conditions for the flower shop inventory -/
def validFlowerShop (shop : FlowerShop) : Prop :=
  shop.violets = shop.carnations / 3 ∧
  shop.tulips = shop.violets / 4 ∧
  shop.carnations = 2 * (shop.carnations + shop.violets + shop.tulips + shop.roses) / 3

/-- Theorem stating that in a valid flower shop, the ratio of roses to tulips is 1:1 -/
theorem roses_to_tulips_ratio (shop : FlowerShop) (h : validFlowerShop shop) :
  shop.roses = shop.tulips := by
  sorry

#check roses_to_tulips_ratio

end roses_to_tulips_ratio_l582_58201


namespace initial_trees_count_l582_58212

/-- The number of walnut trees in the park after planting -/
def total_trees : ℕ := 55

/-- The number of walnut trees planted today -/
def planted_trees : ℕ := 33

/-- The initial number of walnut trees in the park -/
def initial_trees : ℕ := total_trees - planted_trees

theorem initial_trees_count : initial_trees = 22 := by
  sorry

end initial_trees_count_l582_58212


namespace common_roots_product_l582_58269

theorem common_roots_product (A B : ℝ) : 
  (∃ p q r s : ℂ, 
    (p^3 + A*p + 10 = 0) ∧ 
    (q^3 + A*q + 10 = 0) ∧ 
    (r^3 + A*r + 10 = 0) ∧
    (p^3 + B*p^2 + 50 = 0) ∧ 
    (q^3 + B*q^2 + 50 = 0) ∧ 
    (s^3 + B*s^2 + 50 = 0) ∧
    (p ≠ q) ∧ (p ≠ r) ∧ (q ≠ r) ∧ (p ≠ s) ∧ (q ≠ s)) →
  (∃ p q : ℂ, 
    (p^3 + A*p + 10 = 0) ∧ 
    (q^3 + A*q + 10 = 0) ∧
    (p^3 + B*p^2 + 50 = 0) ∧ 
    (q^3 + B*q^2 + 50 = 0) ∧
    (p*q = 5 * (4^(1/3)))) := by
sorry

end common_roots_product_l582_58269


namespace pencil_distribution_l582_58228

theorem pencil_distribution (total_pens : ℕ) (total_pencils : ℕ) (max_students : ℕ) :
  total_pens = 891 →
  max_students = 81 →
  total_pens % max_students = 0 →
  total_pencils % max_students = 0 :=
by sorry

end pencil_distribution_l582_58228


namespace polar_to_rectangular_equivalence_l582_58204

/-- Prove that the polar curve equation ρ = √2 cos(θ - π/4) is equivalent to the rectangular coordinate equation (x - 1/2)² + (y - 1/2)² = 1/2 -/
theorem polar_to_rectangular_equivalence (x y ρ θ : ℝ) :
  (ρ = Real.sqrt 2 * Real.cos (θ - π / 4)) ∧
  (x = ρ * Real.cos θ) ∧
  (y = ρ * Real.sin θ) →
  (x - 1 / 2) ^ 2 + (y - 1 / 2) ^ 2 = 1 / 2 := by
sorry

end polar_to_rectangular_equivalence_l582_58204


namespace student_goldfish_difference_l582_58283

/-- The number of fourth-grade classrooms -/
def num_classrooms : ℕ := 5

/-- The number of students in each fourth-grade classroom -/
def students_per_classroom : ℕ := 20

/-- The number of goldfish in each fourth-grade classroom -/
def goldfish_per_classroom : ℕ := 3

/-- The theorem stating the difference between the total number of students and goldfish -/
theorem student_goldfish_difference :
  num_classrooms * students_per_classroom - num_classrooms * goldfish_per_classroom = 85 := by
  sorry

end student_goldfish_difference_l582_58283


namespace clothing_business_profit_l582_58209

/-- Represents the daily profit function for a clothing business -/
def daily_profit (x : ℝ) : ℝ :=
  (40 - x) * (20 + 2 * x)

theorem clothing_business_profit :
  (∃ x : ℝ, x ≥ 0 ∧ daily_profit x = 1200) ∧
  (∀ y : ℝ, y ≥ 0 → daily_profit y ≠ 1800) := by
  sorry

end clothing_business_profit_l582_58209


namespace wall_length_calculation_l582_58224

theorem wall_length_calculation (mirror_side : ℝ) (wall_width : ℝ) :
  mirror_side = 18 →
  wall_width = 32 →
  (mirror_side * mirror_side) * 2 = wall_width * (20.25 : ℝ) := by
  sorry

end wall_length_calculation_l582_58224


namespace rope_length_satisfies_conditions_l582_58259

/-- The length of the rope in feet -/
def rope_length : ℝ := 10

/-- The length of the rope hanging down from the top of the pillar to the ground in feet -/
def hanging_length : ℝ := 4

/-- The distance from the base of the pillar to where the rope reaches the ground when pulled taut in feet -/
def ground_distance : ℝ := 8

/-- Theorem stating that the rope length satisfies the given conditions -/
theorem rope_length_satisfies_conditions :
  rope_length ^ 2 = (rope_length - hanging_length) ^ 2 + ground_distance ^ 2 :=
by sorry

end rope_length_satisfies_conditions_l582_58259


namespace problem_1_problem_2_problem_3_l582_58295

-- Problem 1
theorem problem_1 : -23 + 58 - (-5) = 40 := by sorry

-- Problem 2
theorem problem_2 : (5/8 + 1/6 - 3/4) * 24 = 1 := by sorry

-- Problem 3
theorem problem_3 : -3^2 - (-5 - 0.2 / (4/5) * (-2)^2) = -3 := by sorry

end problem_1_problem_2_problem_3_l582_58295


namespace smallest_odd_four_primes_l582_58275

def is_prime_factor (p n : ℕ) : Prop := Nat.Prime p ∧ p ∣ n

theorem smallest_odd_four_primes : 
  ∀ n : ℕ, 
    n % 2 = 1 → 
    (∃ p₁ p₂ p₃ p₄ : ℕ, 
      p₁ < p₂ ∧ p₂ < p₃ ∧ p₃ < p₄ ∧
      3 < p₁ ∧
      is_prime_factor p₁ n ∧
      is_prime_factor p₂ n ∧
      is_prime_factor p₃ n ∧
      is_prime_factor p₄ n ∧
      n = p₁ * p₂ * p₃ * p₄) →
    5005 ≤ n :=
by sorry

end smallest_odd_four_primes_l582_58275


namespace mangoes_per_kilogram_l582_58237

theorem mangoes_per_kilogram (total_harvest : ℕ) (sold_to_market : ℕ) (remaining_mangoes : ℕ) :
  total_harvest = 60 →
  sold_to_market = 20 →
  remaining_mangoes = 160 →
  ∃ (sold_to_community : ℕ),
    sold_to_community = (total_harvest - sold_to_market) / 2 ∧
    remaining_mangoes = (total_harvest - sold_to_market - sold_to_community) * 8 :=
by
  sorry

end mangoes_per_kilogram_l582_58237


namespace first_term_exceeding_2020_l582_58298

theorem first_term_exceeding_2020 : 
  (∃ n : ℕ, 2 * n^2 ≥ 2020 ∧ ∀ m : ℕ, m < n → 2 * m^2 < 2020) → 
  (∃ n : ℕ, 2 * n^2 ≥ 2020 ∧ ∀ m : ℕ, m < n → 2 * m^2 < 2020) ∧ 
  (∀ n : ℕ, (2 * n^2 ≥ 2020 ∧ ∀ m : ℕ, m < n → 2 * m^2 < 2020) → n = 32) :=
by sorry

end first_term_exceeding_2020_l582_58298


namespace paint_for_similar_statues_l582_58281

-- Define the height and paint amount for the original statue
def original_height : ℝ := 8
def original_paint : ℝ := 2

-- Define the height and number of new statues
def new_height : ℝ := 2
def num_new_statues : ℕ := 360

-- Theorem statement
theorem paint_for_similar_statues :
  let surface_area_ratio := (new_height / original_height) ^ 2
  let paint_per_new_statue := original_paint * surface_area_ratio
  let total_paint := num_new_statues * paint_per_new_statue
  total_paint = 45 := by sorry

end paint_for_similar_statues_l582_58281


namespace gina_collected_two_bags_l582_58221

/-- The number of bags Gina collected by herself -/
def gina_bags : ℕ := 2

/-- The number of bags collected by the rest of the neighborhood -/
def neighborhood_bags : ℕ := 82 * gina_bags

/-- The weight of each bag in pounds -/
def bag_weight : ℕ := 4

/-- The total weight of litter collected in pounds -/
def total_weight : ℕ := 664

/-- Theorem stating that Gina collected 2 bags of litter -/
theorem gina_collected_two_bags :
  gina_bags = 2 ∧
  neighborhood_bags = 82 * gina_bags ∧
  bag_weight = 4 ∧
  total_weight = 664 ∧
  total_weight = bag_weight * (gina_bags + neighborhood_bags) :=
by sorry

end gina_collected_two_bags_l582_58221


namespace music_school_enrollment_cost_l582_58218

/-- Calculates the total cost for music school enrollment for four siblings --/
theorem music_school_enrollment_cost :
  let regular_tuition : ℕ := 45
  let early_bird_discount : ℕ := 15
  let first_sibling_discount : ℕ := 15
  let additional_sibling_discount : ℕ := 10
  let weekend_class_extra : ℕ := 20
  let multi_instrument_discount : ℕ := 10

  let ali_cost : ℕ := regular_tuition - early_bird_discount
  let matt_cost : ℕ := regular_tuition - first_sibling_discount
  let jane_cost : ℕ := regular_tuition - additional_sibling_discount + weekend_class_extra - multi_instrument_discount
  let sarah_cost : ℕ := regular_tuition - additional_sibling_discount + weekend_class_extra - multi_instrument_discount

  ali_cost + matt_cost + jane_cost + sarah_cost = 150 :=
by
  sorry


end music_school_enrollment_cost_l582_58218


namespace conic_section_classification_l582_58207

/-- Given an interior angle θ of a triangle, vectors m and n, and their dot product,
    prove that the equation x²sin θ - y²cos θ = 1 represents an ellipse with foci on the y-axis. -/
theorem conic_section_classification (θ : ℝ) (m n : Fin 2 → ℝ) :
  (∃ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b ∧ θ = Real.arccos ((b^2 + c^2 - a^2) / (2*b*c))) →  -- θ is an interior angle of a triangle
  (m 0 = Real.sin θ ∧ m 1 = Real.cos θ) →  -- m⃗ = (sin θ, cos θ)
  (n 0 = 1 ∧ n 1 = 1) →  -- n⃗ = (1, 1)
  (m 0 * n 0 + m 1 * n 1 = 1/3) →  -- m⃗ · n⃗ = 1/3
  (∃ (x y : ℝ), x^2 * Real.sin θ - y^2 * Real.cos θ = 1) →  -- Equation: x²sin θ - y²cos θ = 1
  (∃ (a b : ℝ), 0 < a ∧ 0 < b ∧ a ≠ b ∧ 
    ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b) :=  -- The equation represents an ellipse with foci on the y-axis
by sorry

end conic_section_classification_l582_58207


namespace quadratic_inequality_solution_sets_l582_58278

theorem quadratic_inequality_solution_sets (p q : ℝ) :
  (∀ x : ℝ, x^2 + p*x + q < 0 ↔ -1/2 < x ∧ x < 1/3) →
  (∀ x : ℝ, q*x^2 + p*x + 1 > 0 ↔ -2 < x ∧ x < 3) :=
by sorry

end quadratic_inequality_solution_sets_l582_58278


namespace evaluate_x_squared_minus_y_squared_l582_58253

theorem evaluate_x_squared_minus_y_squared : 
  ∀ x y : ℝ, x + y = 10 → 2 * x + y = 13 → x^2 - y^2 = -40 := by
sorry

end evaluate_x_squared_minus_y_squared_l582_58253


namespace cos_2theta_plus_pi_3_l582_58233

theorem cos_2theta_plus_pi_3 (θ : Real) 
  (h1 : θ ∈ Set.Ioo (π / 2) π) 
  (h2 : 1 / Real.sin θ + 1 / Real.cos θ = 2 * Real.sqrt 2) : 
  Real.cos (2 * θ + π / 3) = Real.sqrt 3 / 2 := by
  sorry

end cos_2theta_plus_pi_3_l582_58233


namespace hank_total_donation_l582_58282

def carwash_earnings : ℝ := 100
def carwash_donation_percentage : ℝ := 0.90
def bake_sale_earnings : ℝ := 80
def bake_sale_donation_percentage : ℝ := 0.75
def lawn_mowing_earnings : ℝ := 50
def lawn_mowing_donation_percentage : ℝ := 1.00

def total_donation : ℝ := 
  carwash_earnings * carwash_donation_percentage +
  bake_sale_earnings * bake_sale_donation_percentage +
  lawn_mowing_earnings * lawn_mowing_donation_percentage

theorem hank_total_donation : total_donation = 200 := by
  sorry

end hank_total_donation_l582_58282


namespace hyperbola_focal_length_focal_length_specific_hyperbola_l582_58217

/-- The focal length of a hyperbola with equation x²/a² - y²/b² = 1 is 2c, where c² = a² + b² -/
theorem hyperbola_focal_length (a b c : ℝ) (h : a > 0) (h' : b > 0) :
  (a^2 = 10) → (b^2 = 2) → (c^2 = a^2 + b^2) →
  (2 * c = 4 * Real.sqrt 3) := by
  sorry

/-- The focal length of the hyperbola x²/10 - y²/2 = 1 is 4√3 -/
theorem focal_length_specific_hyperbola :
  ∃ (a b c : ℝ), (a > 0) ∧ (b > 0) ∧
  (a^2 = 10) ∧ (b^2 = 2) ∧ (c^2 = a^2 + b^2) ∧
  (2 * c = 4 * Real.sqrt 3) := by
  sorry

end hyperbola_focal_length_focal_length_specific_hyperbola_l582_58217


namespace consecutive_integers_sum_l582_58225

theorem consecutive_integers_sum (p q r s : ℤ) : 
  (q = p + 1 ∧ r = p + 2 ∧ s = p + 3) →  -- consecutive integers condition
  (p + s = 109) →                        -- given sum condition
  (q + r = 109) :=                       -- theorem to prove
by
  sorry


end consecutive_integers_sum_l582_58225


namespace geometric_sequence_sum_l582_58263

/-- Given a geometric sequence {a_n} where a_1 = 3 and 4a_1, 2a_2, a_3 form an arithmetic sequence,
    prove that a_3 + a_4 + a_5 = 84 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 3 →                            -- a_1 = 3
  4 * a 1 - 2 * a 2 = 2 * a 2 - a 3 →  -- arithmetic sequence condition
  a 3 + a 4 + a 5 = 84 := by
sorry

end geometric_sequence_sum_l582_58263


namespace smallest_angle_solution_l582_58291

theorem smallest_angle_solution (x : Real) : 
  (8 * Real.sin x * (Real.cos x)^6 - 8 * (Real.sin x)^6 * Real.cos x = 2) →
  (x ≥ 0) →
  (∀ y : Real, y > 0 ∧ y < x → 
    8 * Real.sin y * (Real.cos y)^6 - 8 * (Real.sin y)^6 * Real.cos y ≠ 2) →
  x = 11.25 * π / 180 := by
sorry

end smallest_angle_solution_l582_58291


namespace savings_equality_l582_58210

/-- Proves that A's savings equal B's savings given the problem conditions --/
theorem savings_equality (total_salary : ℝ) (a_salary : ℝ) (a_spend_rate : ℝ) (b_spend_rate : ℝ)
  (h1 : total_salary = 3000)
  (h2 : a_salary = 2250)
  (h3 : a_spend_rate = 0.95)
  (h4 : b_spend_rate = 0.85) :
  a_salary * (1 - a_spend_rate) = (total_salary - a_salary) * (1 - b_spend_rate) := by
  sorry

end savings_equality_l582_58210


namespace trivia_team_selection_l582_58270

/-- The number of students not picked for a trivia team --/
def students_not_picked (total : ℕ) (groups : ℕ) (per_group : ℕ) : ℕ :=
  total - (groups * per_group)

/-- Theorem: Given 65 total students, 8 groups, and 6 students per group,
    17 students were not picked for the trivia team --/
theorem trivia_team_selection :
  students_not_picked 65 8 6 = 17 := by
  sorry

end trivia_team_selection_l582_58270


namespace bike_wheel_radius_increase_l582_58206

theorem bike_wheel_radius_increase (initial_circumference final_circumference : ℝ) 
  (h1 : initial_circumference = 30)
  (h2 : final_circumference = 40) :
  (final_circumference / (2 * Real.pi)) - (initial_circumference / (2 * Real.pi)) = 5 / Real.pi := by
  sorry

end bike_wheel_radius_increase_l582_58206


namespace prob_rain_A_given_B_l582_58232

/-- The probability of rain in city A given rain in city B -/
theorem prob_rain_A_given_B (pA pB pAB : ℝ) : 
  pA = 0.2 → pB = 0.18 → pAB = 0.12 → pAB / pB = 2/3 := by
  sorry

end prob_rain_A_given_B_l582_58232


namespace sin_minus_abs_sin_range_l582_58205

theorem sin_minus_abs_sin_range :
  ∀ y : ℝ, (∃ x : ℝ, y = Real.sin x - |Real.sin x|) ↔ -2 ≤ y ∧ y ≤ 0 := by
  sorry

end sin_minus_abs_sin_range_l582_58205
