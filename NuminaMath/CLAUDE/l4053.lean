import Mathlib

namespace arithmetic_calculations_l4053_405326

theorem arithmetic_calculations :
  ((-8) + 10 - (-2) = 12) ∧
  (42 * (-2/3) + (-3/4) / (-0.25) = -25) ∧
  ((-2.5) / (-5/8) * (-0.25) = -1) ∧
  ((1 + 3/4 - 7/8 - 7/12) / (-7/8) = -1/3) := by
  sorry

end arithmetic_calculations_l4053_405326


namespace oplus_inequality_range_l4053_405361

def oplus (x y : ℝ) : ℝ := x * (2 - y)

theorem oplus_inequality_range (a : ℝ) :
  (∀ t : ℝ, oplus (t - a) (t + a) < 1) ↔ (0 < a ∧ a < 2) := by
  sorry

end oplus_inequality_range_l4053_405361


namespace order_cost_l4053_405340

/-- The cost of the order given the prices and quantities of pencils and erasers -/
theorem order_cost (pencil_price eraser_price : ℕ) (total_cartons pencil_cartons : ℕ) : 
  pencil_price = 6 →
  eraser_price = 3 →
  total_cartons = 100 →
  pencil_cartons = 20 →
  pencil_price * pencil_cartons + eraser_price * (total_cartons - pencil_cartons) = 360 := by
sorry

end order_cost_l4053_405340


namespace valid_permutation_exists_valid_permutation_32_valid_permutation_100_l4053_405321

/-- A permutation of numbers from 1 to n satisfying the required property -/
def ValidPermutation (n : ℕ) : Type :=
  { p : Fin n → Fin n // Function.Bijective p ∧
    ∀ i j k, i < j → j < k →
      (p i).val + (p k).val ≠ 2 * (p j).val }

/-- The theorem stating the existence of a valid permutation for any n -/
theorem valid_permutation_exists (n : ℕ) : Nonempty (ValidPermutation n) := by
  sorry

/-- The specific cases for n = 32 and n = 100 -/
theorem valid_permutation_32 : Nonempty (ValidPermutation 32) := by
  exact valid_permutation_exists 32

theorem valid_permutation_100 : Nonempty (ValidPermutation 100) := by
  exact valid_permutation_exists 100

end valid_permutation_exists_valid_permutation_32_valid_permutation_100_l4053_405321


namespace second_month_sale_l4053_405323

/-- Calculates the sale in the second month given sales for other months and the average --/
def calculate_second_month_sale (first_month : ℕ) (third_month : ℕ) (fourth_month : ℕ) 
  (fifth_month : ℕ) (sixth_month : ℕ) (average : ℕ) : ℕ :=
  6 * average - (first_month + third_month + fourth_month + fifth_month + sixth_month)

/-- Theorem stating that the sale in the second month is 5660 --/
theorem second_month_sale : 
  calculate_second_month_sale 5420 6200 6350 6500 6470 6100 = 5660 := by
  sorry

end second_month_sale_l4053_405323


namespace f_properties_l4053_405344

def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem f_properties :
  (∀ x : ℝ, f x ≥ 3 ↔ x ≤ -3/2 ∨ x ≥ 3/2) ∧
  (∀ a : ℝ, (∀ x : ℝ, f x ≥ a^2 - a) ↔ a ∈ Set.Icc (-1) 2) := by
  sorry

end f_properties_l4053_405344


namespace min_value_geometric_sequence_l4053_405396

theorem min_value_geometric_sequence (b₁ b₂ b₃ : ℝ) :
  b₁ = 2 →
  (∃ r : ℝ, b₂ = b₁ * r ∧ b₃ = b₂ * r) →
  ∀ b₂' b₃' : ℝ, (∃ r' : ℝ, b₂' = 2 * r' ∧ b₃' = b₂' * r') →
  3 * b₂ + 4 * b₃ ≥ 3 * b₂' + 4 * b₃' →
  3 * b₂ + 4 * b₃ ≥ -9/8 :=
by sorry

end min_value_geometric_sequence_l4053_405396


namespace fraction_equality_l4053_405315

theorem fraction_equality (a b : ℝ) (h : a / (a + 2 * b) = 3 / 5) : a / b = 3 := by
  sorry

end fraction_equality_l4053_405315


namespace blue_parrots_count_l4053_405312

theorem blue_parrots_count (total : ℕ) (green_fraction : ℚ) : 
  total = 160 → 
  green_fraction = 5/8 → 
  (1 - green_fraction) * total = 60 := by
sorry

end blue_parrots_count_l4053_405312


namespace f_is_odd_l4053_405345

def f (x : ℝ) : ℝ := x^3 - x

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by sorry

end f_is_odd_l4053_405345


namespace orthogonal_projection_magnitude_l4053_405332

/-- Given two vectors a and b in ℝ², prove that the magnitude of the orthogonal projection of a onto b is √5 -/
theorem orthogonal_projection_magnitude (a b : ℝ × ℝ) (h1 : a = (3, -1)) (h2 : b = (1, -2)) :
  ‖(((a.1 * b.1 + a.2 * b.2) / (b.1^2 + b.2^2)) • b)‖ = Real.sqrt 5 := by
  sorry

end orthogonal_projection_magnitude_l4053_405332


namespace fermat_like_equation_exponent_l4053_405324

theorem fermat_like_equation_exponent (x y p n k : ℕ) : 
  x^n + y^n = p^k →
  n > 1 →
  Odd n →
  Nat.Prime p →
  Odd p →
  ∃ l : ℕ, n = p^l := by
sorry

end fermat_like_equation_exponent_l4053_405324


namespace division_problem_l4053_405317

theorem division_problem : (107.8 : ℝ) / 11 = 9.8 := by
  sorry

end division_problem_l4053_405317


namespace equation_solutions_l4053_405391

theorem equation_solutions :
  (∀ x : ℝ, 4 * (x - 2)^2 = 9 ↔ x = 7/2 ∨ x = 1/2) ∧
  (∀ x : ℝ, x^2 + 6*x - 1 = 0 ↔ x = -3 + Real.sqrt 10 ∨ x = -3 - Real.sqrt 10) :=
by sorry

end equation_solutions_l4053_405391


namespace max_value_of_x_plus_inverse_l4053_405307

theorem max_value_of_x_plus_inverse (x : ℝ) (h : 7 = x^2 + 1/x^2) :
  ∃ (y : ℝ), y = x + 1/x ∧ y ≤ 3 ∧ ∃ (z : ℝ), z = x + 1/x ∧ z = 3 :=
sorry

end max_value_of_x_plus_inverse_l4053_405307


namespace oliver_bath_frequency_l4053_405349

def bucket_capacity : ℕ := 120
def buckets_to_fill : ℕ := 14
def buckets_removed : ℕ := 3
def weekly_water_usage : ℕ := 9240

theorem oliver_bath_frequency :
  let full_tub := bucket_capacity * buckets_to_fill
  let water_removed := bucket_capacity * buckets_removed
  let water_per_bath := full_tub - water_removed
  weekly_water_usage / water_per_bath = 7 := by sorry

end oliver_bath_frequency_l4053_405349


namespace probability_of_choosing_quarter_l4053_405352

def quarter_value : ℚ := 25 / 100
def nickel_value : ℚ := 5 / 100
def penny_value : ℚ := 1 / 100

def total_value_per_coin_type : ℚ := 10

theorem probability_of_choosing_quarter :
  let num_quarters := total_value_per_coin_type / quarter_value
  let num_nickels := total_value_per_coin_type / nickel_value
  let num_pennies := total_value_per_coin_type / penny_value
  let total_coins := num_quarters + num_nickels + num_pennies
  (num_quarters / total_coins : ℚ) = 1 / 31 := by
sorry

end probability_of_choosing_quarter_l4053_405352


namespace james_coins_value_l4053_405376

theorem james_coins_value (p n : ℕ) : 
  p + n = 15 →
  p - 1 = n →
  p * 1 + n * 5 = 43 :=
by sorry

end james_coins_value_l4053_405376


namespace group_size_l4053_405327

/-- The number of people in a group, given weight changes. -/
theorem group_size (weight_increase_per_person : ℝ) (new_person_weight : ℝ) (replaced_person_weight : ℝ) :
  weight_increase_per_person * 10 = new_person_weight - replaced_person_weight →
  10 = (new_person_weight - replaced_person_weight) / weight_increase_per_person :=
by
  sorry

#check group_size 7.2 137 65

end group_size_l4053_405327


namespace geometric_sequence_second_term_l4053_405300

theorem geometric_sequence_second_term (a : ℝ) : 
  a > 0 ∧ 
  (∃ r : ℝ, r ≠ 0 ∧ 25 * r = a ∧ a * r = 8/5) → 
  a = 2 * Real.sqrt 10 := by
sorry

end geometric_sequence_second_term_l4053_405300


namespace quadratic_solution_l4053_405341

-- Define the quadratic equation
def quadratic_equation (p q x : ℝ) : Prop := x^2 + p*x + q = 0

-- Define the conditions
theorem quadratic_solution (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) :
  (quadratic_equation p q (2*p) ∧ quadratic_equation p q (q/2)) →
  p = 1 ∧ q = -6 := by sorry

end quadratic_solution_l4053_405341


namespace toy_problem_solution_l4053_405393

/-- Represents the toy purchasing and pricing problem -/
structure ToyProblem where
  total_toys : ℕ
  total_cost : ℕ
  purchase_price_A : ℕ
  purchase_price_B : ℕ
  original_price_A : ℕ
  original_daily_sales : ℕ
  sales_increase_rate : ℕ
  desired_daily_profit : ℕ

/-- The solution to the toy problem -/
structure ToySolution where
  num_A : ℕ
  num_B : ℕ
  new_price_A : ℕ

/-- Theorem stating the correct solution for the given problem -/
theorem toy_problem_solution (p : ToyProblem) 
  (h1 : p.total_toys = 50)
  (h2 : p.total_cost = 1320)
  (h3 : p.purchase_price_A = 28)
  (h4 : p.purchase_price_B = 24)
  (h5 : p.original_price_A = 40)
  (h6 : p.original_daily_sales = 8)
  (h7 : p.sales_increase_rate = 1)
  (h8 : p.desired_daily_profit = 96) :
  ∃ (s : ToySolution), 
    s.num_A = 30 ∧ 
    s.num_B = 20 ∧ 
    s.new_price_A = 36 ∧
    s.num_A + s.num_B = p.total_toys ∧
    s.num_A * p.purchase_price_A + s.num_B * p.purchase_price_B = p.total_cost ∧
    (s.new_price_A - p.purchase_price_A) * (p.original_daily_sales + (p.original_price_A - s.new_price_A) * p.sales_increase_rate) = p.desired_daily_profit :=
by
  sorry


end toy_problem_solution_l4053_405393


namespace charlottes_phone_usage_l4053_405389

/-- Charlotte's daily phone usage problem -/
theorem charlottes_phone_usage 
  (social_media_time : ℝ) 
  (weekly_social_media : ℝ) 
  (h1 : social_media_time = weekly_social_media / 7)
  (h2 : weekly_social_media = 56)
  (h3 : social_media_time = daily_phone_time / 2) : 
  daily_phone_time = 16 :=
sorry

end charlottes_phone_usage_l4053_405389


namespace gcd_180_270_l4053_405388

theorem gcd_180_270 : Nat.gcd 180 270 = 90 := by
  sorry

end gcd_180_270_l4053_405388


namespace margie_change_l4053_405301

/-- The change received when buying apples -/
def change_received (num_apples : ℕ) (cost_per_apple : ℚ) (amount_paid : ℚ) : ℚ :=
  amount_paid - (num_apples : ℚ) * cost_per_apple

/-- Theorem: Margie's change when buying apples -/
theorem margie_change : 
  let num_apples : ℕ := 3
  let cost_per_apple : ℚ := 50 / 100
  let amount_paid : ℚ := 5
  change_received num_apples cost_per_apple amount_paid = 7 / 2 := by
sorry

end margie_change_l4053_405301


namespace two_blue_gumballs_probability_l4053_405328

/-- The probability of drawing a pink gumball from the jar -/
def prob_pink : ℝ := 0.5714285714285714

/-- The probability of drawing a blue gumball from the jar -/
def prob_blue : ℝ := 1 - prob_pink

/-- The probability of drawing two blue gumballs in a row -/
def prob_two_blue : ℝ := prob_blue * prob_blue

theorem two_blue_gumballs_probability :
  prob_two_blue = 0.1836734693877551 := by sorry

end two_blue_gumballs_probability_l4053_405328


namespace fifteenth_set_sum_l4053_405392

def first_element (n : ℕ) : ℕ := 
  1 + (n - 1) * n / 2

def last_element (n : ℕ) : ℕ := 
  first_element n + n - 1

def set_sum (n : ℕ) : ℕ := 
  n * (first_element n + last_element n) / 2

theorem fifteenth_set_sum : set_sum 15 = 1695 := by
  sorry

end fifteenth_set_sum_l4053_405392


namespace parallel_line_equation_l4053_405384

/-- Given two parallel lines with a distance of 2 between them, where one line has the equation 5x - 12y + 6 = 0, prove that the equation of the other line is either 5x - 12y + 32 = 0 or 5x - 12y - 20 = 0 -/
theorem parallel_line_equation (x y : ℝ) :
  let l₁ : ℝ → ℝ → Prop := λ x y ↦ 5 * x - 12 * y + 6 = 0
  let l : ℝ → ℝ → Prop := λ x y ↦ 5 * x - 12 * y + 32 = 0 ∨ 5 * x - 12 * y - 20 = 0
  let parallel : Prop := ∃ (k : ℝ), k ≠ 0 ∧ ∀ x y, l₁ x y ↔ l x y
  let distance : ℝ := 2
  parallel → (∀ x y, l x y ↔ (5 * x - 12 * y + 32 = 0 ∨ 5 * x - 12 * y - 20 = 0)) :=
by
  sorry

end parallel_line_equation_l4053_405384


namespace earnings_before_car_purchase_l4053_405366

/-- Calculates the total earnings before saving enough to buy a car. -/
def totalEarningsBeforePurchase (monthlyEarnings : ℕ) (monthlySavings : ℕ) (carCost : ℕ) : ℕ :=
  (carCost / monthlySavings) * monthlyEarnings

/-- Theorem stating the total earnings before saving enough to buy the car. -/
theorem earnings_before_car_purchase :
  totalEarningsBeforePurchase 4000 500 45000 = 360000 := by
  sorry

end earnings_before_car_purchase_l4053_405366


namespace cosecant_330_degrees_l4053_405387

theorem cosecant_330_degrees :
  let csc (θ : ℝ) := 1 / Real.sin θ
  let π : ℝ := Real.pi
  ∀ (θ : ℝ), Real.sin (2 * π - θ) = -Real.sin θ
  → Real.sin (π / 6) = 1 / 2
  → csc (11 * π / 6) = -2 := by
  sorry

end cosecant_330_degrees_l4053_405387


namespace sqrt_eight_minus_sqrt_two_equals_sqrt_two_l4053_405313

theorem sqrt_eight_minus_sqrt_two_equals_sqrt_two :
  Real.sqrt 8 - Real.sqrt 2 = Real.sqrt 2 := by
  sorry

end sqrt_eight_minus_sqrt_two_equals_sqrt_two_l4053_405313


namespace circles_A_B_intersect_l4053_405356

/-- Circle A is defined by the equation x^2 + y^2 + 4x + 2y + 1 = 0 -/
def circle_A (x y : ℝ) : Prop := x^2 + y^2 + 4*x + 2*y + 1 = 0

/-- Circle B is defined by the equation x^2 + y^2 - 2x - 6y + 1 = 0 -/
def circle_B (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 6*y + 1 = 0

/-- Two circles are intersecting if there exists a point that satisfies both circle equations -/
def circles_intersect (c1 c2 : ℝ → ℝ → Prop) : Prop :=
  ∃ x y : ℝ, c1 x y ∧ c2 x y

/-- Theorem stating that circle A and circle B are intersecting -/
theorem circles_A_B_intersect : circles_intersect circle_A circle_B := by
  sorry

end circles_A_B_intersect_l4053_405356


namespace order_of_special_roots_l4053_405314

theorem order_of_special_roots : ∃ (a b c : ℝ), 
  a = (2 : ℝ) ^ (1/2) ∧ 
  b = Real.exp (1/Real.exp 1) ∧ 
  c = (3 : ℝ) ^ (1/3) ∧ 
  a < c ∧ c < b := by
  sorry

end order_of_special_roots_l4053_405314


namespace function_minimum_and_tangent_line_l4053_405367

/-- The function f(x) = x³ - x² + ax -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - x^2 + a*x

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*x + a

theorem function_minimum_and_tangent_line (a : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a 1 ≤ f a x) →
  a = -1 ∧
  (∃ x₀ : ℝ, f a x₀ - (-1) = f' a x₀ * (x₀ - (-1)) ∧
              (x₀ = 1 ∨ 4 * x₀ - f a x₀ + 3 = 0)) :=
sorry

end function_minimum_and_tangent_line_l4053_405367


namespace aprons_to_sew_tomorrow_l4053_405304

def total_aprons : ℕ := 150
def aprons_before_today : ℕ := 13
def aprons_today : ℕ := 3 * aprons_before_today

def aprons_sewn_so_far : ℕ := aprons_before_today + aprons_today
def remaining_aprons : ℕ := total_aprons - aprons_sewn_so_far
def aprons_tomorrow : ℕ := remaining_aprons / 2

theorem aprons_to_sew_tomorrow : aprons_tomorrow = 49 := by
  sorry

end aprons_to_sew_tomorrow_l4053_405304


namespace roger_toys_theorem_l4053_405358

def max_toys_buyable (initial_amount : ℕ) (spent_amount : ℕ) (toy_cost : ℕ) : ℕ :=
  (initial_amount - spent_amount) / toy_cost

theorem roger_toys_theorem : 
  max_toys_buyable 63 48 3 = 5 := by
  sorry

end roger_toys_theorem_l4053_405358


namespace sufficient_not_necessary_condition_l4053_405334

theorem sufficient_not_necessary_condition (a : ℝ) :
  (a ≥ 5) →
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0) ∧
  ¬(∀ a : ℝ, (∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0) → a ≥ 5) :=
by sorry

end sufficient_not_necessary_condition_l4053_405334


namespace nature_reserve_count_l4053_405343

theorem nature_reserve_count (total_heads total_legs : ℕ) 
  (h1 : total_heads = 300)
  (h2 : total_legs = 688) : ∃ (birds mammals reptiles : ℕ),
  birds + mammals + reptiles = total_heads ∧
  2 * birds + 4 * mammals + 6 * reptiles = total_legs ∧
  birds = 234 := by
  sorry

end nature_reserve_count_l4053_405343


namespace unique_solution_l4053_405377

theorem unique_solution : ∃! x : ℝ, ((52 + x) * 3 - 60) / 8 = 15 := by
  sorry

end unique_solution_l4053_405377


namespace absolute_value_sum_l4053_405342

theorem absolute_value_sum (a : ℝ) (h : -2 < a ∧ a < 0) : |a| + |a+2| = 2 := by
  sorry

end absolute_value_sum_l4053_405342


namespace complex_number_equality_l4053_405318

theorem complex_number_equality (z : ℂ) (h : z / (1 - Complex.I) = Complex.I) : z = 1 + Complex.I := by
  sorry

end complex_number_equality_l4053_405318


namespace waiter_tips_sum_l4053_405355

/-- Represents the tips received by a waiter during a lunch shift -/
def WaiterTips : Type := List Float

/-- The number of customers served during the lunch shift -/
def totalCustomers : Nat := 10

/-- The number of customers who left a tip -/
def tippingCustomers : Nat := 5

/-- The list of tips received from the customers who left tips -/
def tipsList : WaiterTips := [1.50, 2.75, 3.25, 4.00, 5.00]

/-- Theorem stating that the sum of tips received by the waiter is $16.50 -/
theorem waiter_tips_sum :
  tipsList.length = tippingCustomers ∧
  totalCustomers = tippingCustomers + (totalCustomers - tippingCustomers) →
  tipsList.sum = 16.50 := by
  sorry

end waiter_tips_sum_l4053_405355


namespace pig_year_paintings_distribution_l4053_405363

theorem pig_year_paintings_distribution (n : ℕ) (k : ℕ) (h1 : n = 4) (h2 : k = 3) :
  let total_outcomes := k^n
  let favorable_outcomes := (n.choose 2) * (k.factorial)
  (favorable_outcomes : ℚ) / total_outcomes = 4/9 := by
  sorry

end pig_year_paintings_distribution_l4053_405363


namespace stock_yield_inconsistency_l4053_405380

theorem stock_yield_inconsistency (price : ℝ) (price_pos : price > 0) :
  ¬(∃ (dividend : ℝ), 
    dividend = 0.2 * price ∧ 
    dividend / price = 0.1) :=
by
  sorry

#check stock_yield_inconsistency

end stock_yield_inconsistency_l4053_405380


namespace sector_inscribed_circle_area_ratio_l4053_405329

theorem sector_inscribed_circle_area_ratio (α : Real) :
  let R := 1  -- We can set R to 1 without loss of generality
  let r := (R * Real.sin (α / 2)) / (1 + Real.sin (α / 2))
  let sector_area := (1 / 2) * R^2 * α
  let inscribed_circle_area := Real.pi * r^2
  sector_area / inscribed_circle_area = (2 * α * (Real.cos (Real.pi / 4 - α / 4))^2) / (Real.pi * (Real.sin (α / 2))^2) :=
by sorry

end sector_inscribed_circle_area_ratio_l4053_405329


namespace test_probabilities_l4053_405339

/-- Given probabilities of answering questions correctly on a test, 
    calculate the probability of answering neither question correctly. -/
theorem test_probabilities (p_first p_second p_both : ℝ) 
  (h1 : p_first = 0.75)
  (h2 : p_second = 0.55)
  (h3 : p_both = 0.50) :
  1 - (p_first + p_second - p_both) = 0.20 := by
  sorry

end test_probabilities_l4053_405339


namespace complex_fraction_equals_two_l4053_405374

theorem complex_fraction_equals_two (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = a*b) :
  (a^6 + b^6) / (a + b)^6 = 2 := by
  sorry

end complex_fraction_equals_two_l4053_405374


namespace exam_questions_count_l4053_405357

/-- Prove that the number of questions in an exam is 50, given the following conditions:
1. Sylvia had one-fifth of incorrect answers
2. Sergio got 4 mistakes
3. Sergio has 6 more correct answers than Sylvia
-/
theorem exam_questions_count : ∃ Q : ℕ,
  (Q : ℚ) > 0 ∧
  let sylvia_correct := (4 : ℚ) / 5 * Q
  let sergio_correct := Q - 4
  sergio_correct = sylvia_correct + 6 ∧
  Q = 50 := by
  sorry

#check exam_questions_count

end exam_questions_count_l4053_405357


namespace concert_attendance_l4053_405319

/-- The number of buses used for the concert trip -/
def number_of_buses : ℕ := 8

/-- The number of students each bus can carry -/
def students_per_bus : ℕ := 45

/-- The total number of students who went to the concert -/
def total_students : ℕ := number_of_buses * students_per_bus

/-- Theorem stating that the total number of students who went to the concert is 360 -/
theorem concert_attendance : total_students = 360 := by
  sorry

end concert_attendance_l4053_405319


namespace even_function_range_l4053_405353

def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 2

theorem even_function_range (a b : ℝ) :
  (∀ x ∈ Set.Icc (1 + a) 2, f a b x = f a b (-x)) →
  Set.range (f a b) = Set.Icc (-10) 2 := by
  sorry

end even_function_range_l4053_405353


namespace max_principals_in_period_l4053_405386

/-- Represents the number of years in a principal's term -/
def term_length : ℕ := 4

/-- Represents the total period in years -/
def total_period : ℕ := 10

/-- Represents the maximum number of principals that can serve during the total period -/
def max_principals : ℕ := 4

/-- Theorem stating that given a 10-year period and principals serving 4-year terms,
    the maximum number of principals that can serve during this period is 4 -/
theorem max_principals_in_period :
  ∀ (n : ℕ), n ≤ max_principals →
  n * term_length > total_period →
  (n - 1) * term_length ≤ total_period :=
sorry

end max_principals_in_period_l4053_405386


namespace car_wash_goal_proof_l4053_405330

def car_wash_goal (families_10 : ℕ) (amount_10 : ℕ) (families_5 : ℕ) (amount_5 : ℕ) (more_needed : ℕ) : Prop :=
  let earned_10 := families_10 * amount_10
  let earned_5 := families_5 * amount_5
  let total_earned := earned_10 + earned_5
  let goal := total_earned + more_needed
  goal = 150

theorem car_wash_goal_proof :
  car_wash_goal 3 10 15 5 45 := by
  sorry

end car_wash_goal_proof_l4053_405330


namespace train_speed_calculation_l4053_405369

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 100 →
  bridge_length = 150 →
  crossing_time = 49.9960003199744 →
  ∃ (speed : ℝ), (abs (speed - 18) < 0.1 ∧ 
    speed = (train_length + bridge_length) / crossing_time * 3.6) := by
  sorry


end train_speed_calculation_l4053_405369


namespace min_journey_cost_l4053_405398

-- Define the cities and distances
def XY : ℝ := 3500
def XZ : ℝ := 4000

-- Define the cost functions
def train_cost (distance : ℝ) : ℝ := 0.20 * distance
def taxi_cost (distance : ℝ) : ℝ := 150 + 0.15 * distance

-- Define the theorem
theorem min_journey_cost :
  let YZ : ℝ := Real.sqrt (XZ^2 - XY^2)
  let XY_cost : ℝ := min (train_cost XY) (taxi_cost XY)
  let YZ_cost : ℝ := min (train_cost YZ) (taxi_cost YZ)
  let ZX_cost : ℝ := min (train_cost XZ) (taxi_cost XZ)
  XY_cost + YZ_cost + ZX_cost = 1812.30 := by sorry

end min_journey_cost_l4053_405398


namespace distance_between_cities_l4053_405397

/-- The distance between two cities A and B, where two trains traveling towards each other meet. -/
theorem distance_between_cities (v1 v2 t1 t2 : ℝ) (h1 : v1 = 60) (h2 : v2 = 75) (h3 : t1 = 4) (h4 : t2 = 3) : v1 * t1 + v2 * t2 = 465 := by
  sorry

end distance_between_cities_l4053_405397


namespace tangent_line_implies_function_values_l4053_405337

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem tangent_line_implies_function_values :
  (∀ y, y = f 5 ↔ y = -5 + 8) →  -- Tangent line equation at x = 5
  (f 5 = 3 ∧ deriv f 5 = -1) :=
by sorry

end tangent_line_implies_function_values_l4053_405337


namespace parakeets_per_cage_l4053_405378

theorem parakeets_per_cage 
  (num_cages : ℝ)
  (parrots_per_cage : ℝ)
  (total_birds : ℕ)
  (h1 : num_cages = 6.0)
  (h2 : parrots_per_cage = 6.0)
  (h3 : total_birds = 48) :
  (total_birds : ℝ) - num_cages * parrots_per_cage = num_cages * 2 :=
by sorry

end parakeets_per_cage_l4053_405378


namespace luka_age_when_max_born_l4053_405336

/-- Proves Luka's age when Max was born -/
theorem luka_age_when_max_born (luka_aubrey_age_diff : ℕ) 
  (aubrey_age_at_max_6 : ℕ) (max_age_at_aubrey_8 : ℕ) :
  luka_aubrey_age_diff = 2 →
  aubrey_age_at_max_6 = 8 →
  max_age_at_aubrey_8 = 6 →
  aubrey_age_at_max_6 - max_age_at_aubrey_8 + luka_aubrey_age_diff = 4 :=
by sorry

end luka_age_when_max_born_l4053_405336


namespace paco_cookies_l4053_405309

/-- The number of cookies Paco initially had -/
def initial_cookies : ℕ := sorry

/-- The number of cookies Paco gave to his friend -/
def cookies_given : ℕ := 9

/-- The number of cookies Paco ate -/
def cookies_eaten : ℕ := 18

/-- The difference between cookies eaten and given -/
def cookies_difference : ℕ := 9

theorem paco_cookies : 
  initial_cookies = cookies_given + cookies_eaten ∧
  cookies_eaten = cookies_given + cookies_difference ∧
  initial_cookies = 27 := by sorry

end paco_cookies_l4053_405309


namespace comparison_of_expressions_l4053_405308

theorem comparison_of_expressions (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  ¬ (∀ x y z : ℝ, 
    (x = (a + 1/a) * (b + 1/b) ∧ 
     y = (Real.sqrt (a * b) + 1 / Real.sqrt (a * b))^2 ∧ 
     z = ((a + b)/2 + 2/(a + b))^2) →
    (x > y ∧ x > z) ∨ (y > x ∧ y > z) ∨ (z > x ∧ z > y)) :=
by sorry


end comparison_of_expressions_l4053_405308


namespace derivative_f_at_zero_l4053_405338

noncomputable def f (θ : ℝ) : ℝ := (Real.sin θ) / (2 + Real.cos θ)

theorem derivative_f_at_zero :
  deriv f 0 = 1/3 := by sorry

end derivative_f_at_zero_l4053_405338


namespace parallelogram_area_v_w_l4053_405360

/-- The area of a parallelogram formed by two 2D vectors -/
def parallelogramArea (v w : Fin 2 → ℝ) : ℝ :=
  |v 0 * w 1 - v 1 * w 0|

/-- Vectors v and w -/
def v : Fin 2 → ℝ := ![4, -6]
def w : Fin 2 → ℝ := ![7, -1]

/-- Theorem stating that the area of the parallelogram formed by v and w is 38 -/
theorem parallelogram_area_v_w : parallelogramArea v w = 38 := by
  sorry

end parallelogram_area_v_w_l4053_405360


namespace triangle_angle_sum_l4053_405335

theorem triangle_angle_sum (a b c : ℝ) (h1 : b = 30)
    (h2 : c = 3 * b) : a = 60 := by
  sorry

end triangle_angle_sum_l4053_405335


namespace sum_of_integers_l4053_405394

theorem sum_of_integers (a b : ℕ+) (h1 : a - b = 8) (h2 : a * b = 65) : a + b = 18 := by
  sorry

end sum_of_integers_l4053_405394


namespace inequality_solution_l4053_405306

theorem inequality_solution (x : ℝ) : 
  (9*x^2 + 27*x - 40) / ((3*x - 4)*(x + 5)) < 5 ↔ 
  (x > -6 ∧ x < -5) ∨ (x > 4/3 ∧ x < 5/3) :=
by sorry

end inequality_solution_l4053_405306


namespace two_is_sup_of_satisfying_set_l4053_405303

/-- A sequence of positive integers satisfying the given inequality -/
def SatisfyingSequence (r : ℝ) (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, (a n ≤ a (n + 2)) ∧ ((a (n + 2))^2 ≤ (a n)^2 + r * (a (n + 1)))

/-- The property that a sequence eventually becomes constant -/
def EventuallyConstant (a : ℕ → ℕ) : Prop :=
  ∃ M : ℕ, ∀ n ≥ M, a (n + 2) = a n

/-- The set of real numbers r satisfying the given condition -/
def SatisfyingSet : Set ℝ :=
  {r : ℝ | ∀ a : ℕ → ℕ, SatisfyingSequence r a → EventuallyConstant a}

/-- The main theorem: 2 is the supremum of the satisfying set -/
theorem two_is_sup_of_satisfying_set : 
  IsLUB SatisfyingSet 2 := by sorry

end two_is_sup_of_satisfying_set_l4053_405303


namespace house_development_l4053_405382

theorem house_development (total houses garage pool neither : ℕ) : 
  total = 70 → 
  garage = 50 → 
  pool = 40 → 
  neither = 15 → 
  ∃ both : ℕ, both = garage + pool - (total - neither) :=
by
  sorry

end house_development_l4053_405382


namespace ladder_problem_l4053_405316

theorem ladder_problem (ladder_length height : ℝ) 
  (h1 : ladder_length = 13)
  (h2 : height = 12) :
  ∃ (base : ℝ), base^2 + height^2 = ladder_length^2 ∧ base = 5 :=
sorry

end ladder_problem_l4053_405316


namespace systematic_sampling_interval_l4053_405381

theorem systematic_sampling_interval 
  (total : ℕ) 
  (samples : ℕ) 
  (h1 : total = 231) 
  (h2 : samples = 22) :
  let adjusted_total := total - (total % samples)
  adjusted_total / samples = 10 :=
sorry

end systematic_sampling_interval_l4053_405381


namespace meals_per_day_l4053_405333

theorem meals_per_day (people : ℕ) (total_plates : ℕ) (days : ℕ) (plates_per_meal : ℕ)
  (h1 : people = 6)
  (h2 : total_plates = 144)
  (h3 : days = 4)
  (h4 : plates_per_meal = 2)
  : (total_plates / (people * days * plates_per_meal) : ℚ) = 3 := by
  sorry

end meals_per_day_l4053_405333


namespace orangeade_pricing_l4053_405311

/-- Orangeade pricing problem -/
theorem orangeade_pricing
  (orange_juice : ℝ)  -- Amount of orange juice (same for both days)
  (water_day1 : ℝ)    -- Amount of water on day 1
  (water_day2 : ℝ)    -- Amount of water on day 2
  (price_day1 : ℝ)    -- Price per glass on day 1
  (h1 : water_day1 = orange_juice)        -- Equal amounts of orange juice and water on day 1
  (h2 : water_day2 = 2 * orange_juice)    -- Twice the amount of water on day 2
  (h3 : price_day1 = 0.48)                -- Price per glass on day 1 is $0.48
  (h4 : (orange_juice + water_day1) * price_day1 = 
        (orange_juice + water_day2) * price_day2) -- Same revenue on both days
  : price_day2 = 0.32 :=
by sorry

end orangeade_pricing_l4053_405311


namespace largest_prime_factor_of_expression_l4053_405383

theorem largest_prime_factor_of_expression : 
  ∃ p : ℕ, p.Prime ∧ p ∣ (20^3 + 15^4 - 10^5) ∧ 
  ∀ q : ℕ, q.Prime → q ∣ (20^3 + 15^4 - 10^5) → q ≤ p :=
by
  -- The proof goes here
  sorry

end largest_prime_factor_of_expression_l4053_405383


namespace wickets_in_last_match_is_five_l4053_405375

/-- Represents the bowling statistics of a cricket player -/
structure BowlingStats where
  initialAverage : ℝ
  runsLastMatch : ℕ
  averageDecrease : ℝ
  wicketsBeforeLastMatch : ℕ

/-- Calculates the number of wickets taken in the last match -/
def wicketsInLastMatch (stats : BowlingStats) : ℕ :=
  sorry

/-- Theorem stating that given the specific bowling statistics, the number of wickets in the last match is 5 -/
theorem wickets_in_last_match_is_five (stats : BowlingStats) 
  (h1 : stats.initialAverage = 12.4)
  (h2 : stats.runsLastMatch = 26)
  (h3 : stats.averageDecrease = 0.4)
  (h4 : stats.wicketsBeforeLastMatch = 85) :
  wicketsInLastMatch stats = 5 := by
  sorry

end wickets_in_last_match_is_five_l4053_405375


namespace office_age_problem_l4053_405371

theorem office_age_problem (total_persons : Nat) (avg_age_all : Nat) (num_group1 : Nat) 
  (avg_age_group1 : Nat) (num_group2 : Nat) (age_person15 : Nat) :
  total_persons = 19 →
  avg_age_all = 15 →
  num_group1 = 9 →
  avg_age_group1 = 16 →
  num_group2 = 5 →
  age_person15 = 71 →
  (((total_persons * avg_age_all) - (num_group1 * avg_age_group1) - age_person15) / num_group2) = 14 := by
  sorry

#check office_age_problem

end office_age_problem_l4053_405371


namespace fraction_calculation_l4053_405322

theorem fraction_calculation : 
  (2 + 1/4 + 0.25) / (2 + 3/4 - 1/2) + (2 * 0.5) / (2 + 1/5 - 2/5) = 5/3 := by
  sorry

end fraction_calculation_l4053_405322


namespace unique_solution_k_values_l4053_405362

theorem unique_solution_k_values (k : ℝ) :
  (∃! x : ℝ, 1 ≤ k * x^2 + 2 ∧ x + k ≤ 2) ↔ 
  (k = 1 + Real.sqrt 2 ∨ k = (1 - Real.sqrt 5) / 2) :=
by sorry

end unique_solution_k_values_l4053_405362


namespace no_reciprocal_implies_one_l4053_405310

/-- If a number minus 1 does not have a reciprocal, then that number equals 1 -/
theorem no_reciprocal_implies_one (a : ℝ) : (∀ x : ℝ, x * (a - 1) ≠ 1) → a = 1 := by
  sorry

end no_reciprocal_implies_one_l4053_405310


namespace smallest_solution_quartic_equation_l4053_405351

theorem smallest_solution_quartic_equation :
  ∃ (x : ℝ), x^4 - 50*x^2 + 625 = 0 ∧ 
  (∀ y : ℝ, y^4 - 50*y^2 + 625 = 0 → y ≥ x) ∧
  x = -5 := by
sorry

end smallest_solution_quartic_equation_l4053_405351


namespace no_nontrivial_solution_for_4n_plus_3_prime_l4053_405331

theorem no_nontrivial_solution_for_4n_plus_3_prime (a : ℕ) (x y z : ℤ) :
  Prime a →
  (∃ n : ℕ, a = 4 * n + 3) →
  x^2 + y^2 = a * z^2 →
  x = 0 ∧ y = 0 :=
by sorry

end no_nontrivial_solution_for_4n_plus_3_prime_l4053_405331


namespace rectangle_width_l4053_405365

/-- Given a rectangle with length 13 cm and perimeter 50 cm, prove its width is 12 cm. -/
theorem rectangle_width (length : ℝ) (perimeter : ℝ) (width : ℝ) : 
  length = 13 → perimeter = 50 → perimeter = 2 * (length + width) → width = 12 := by
  sorry

end rectangle_width_l4053_405365


namespace coin_value_difference_l4053_405379

/-- Represents the number of coins of each type -/
structure CoinCount where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ

/-- Calculates the total value in cents for a given coin count -/
def totalValue (coins : CoinCount) : ℕ :=
  coins.pennies + 5 * coins.nickels + 10 * coins.dimes

/-- Represents the problem constraints -/
def validCoinCount (coins : CoinCount) : Prop :=
  coins.pennies ≥ 1 ∧ coins.nickels ≥ 1 ∧ coins.dimes ≥ 1 ∧
  coins.pennies + coins.nickels + coins.dimes = 3030

theorem coin_value_difference :
  ∃ (maxCoins minCoins : CoinCount),
    validCoinCount maxCoins ∧
    validCoinCount minCoins ∧
    (∀ c, validCoinCount c → totalValue c ≤ totalValue maxCoins) ∧
    (∀ c, validCoinCount c → totalValue c ≥ totalValue minCoins) ∧
    totalValue maxCoins - totalValue minCoins = 21182 :=
sorry

end coin_value_difference_l4053_405379


namespace solutions_count_for_specific_n_l4053_405350

/-- Count of integer solutions for x^2 - y^2 = n^2 -/
def count_solutions (n : ℕ) : ℕ :=
  if n = 1992 then 90
  else if n = 1993 then 6
  else if n = 1994 then 6
  else 0

/-- Theorem stating the count of integer solutions for x^2 - y^2 = n^2 for specific n values -/
theorem solutions_count_for_specific_n :
  (count_solutions 1992 = 90) ∧
  (count_solutions 1993 = 6) ∧
  (count_solutions 1994 = 6) :=
by sorry

end solutions_count_for_specific_n_l4053_405350


namespace eggs_laid_per_dove_l4053_405354

/-- The number of eggs laid by each dove -/
def eggs_per_dove : ℕ := 3

/-- The initial number of female doves -/
def initial_doves : ℕ := 20

/-- The fraction of eggs that hatched -/
def hatch_rate : ℚ := 3/4

/-- The total number of doves after hatching -/
def total_doves : ℕ := 65

theorem eggs_laid_per_dove :
  eggs_per_dove * initial_doves * hatch_rate = total_doves - initial_doves :=
sorry

end eggs_laid_per_dove_l4053_405354


namespace sphere_surface_area_l4053_405373

theorem sphere_surface_area (d : ℝ) (h : d = 4) :
  4 * Real.pi * (d / 2)^2 = 16 * Real.pi :=
by sorry

end sphere_surface_area_l4053_405373


namespace growth_comparison_l4053_405370

theorem growth_comparison (x : ℝ) (h : x > 0) :
  (0 < x ∧ x < 1/2 → (fun y => y) x > (fun y => y^2) x) ∧
  (x > 1/2 → (fun y => y^2) x > (fun y => y) x) := by
sorry

end growth_comparison_l4053_405370


namespace library_schedule_lcm_l4053_405385

theorem library_schedule_lcm : Nat.lcm 5 (Nat.lcm 3 (Nat.lcm 9 8)) = 360 := by
  sorry

end library_schedule_lcm_l4053_405385


namespace mike_seashells_count_l4053_405364

/-- The number of seashells Mike found initially -/
def initial_seashells : ℝ := 6.0

/-- The number of seashells Mike found later -/
def later_seashells : ℝ := 4.0

/-- The total number of seashells Mike found -/
def total_seashells : ℝ := initial_seashells + later_seashells

theorem mike_seashells_count : total_seashells = 10.0 := by sorry

end mike_seashells_count_l4053_405364


namespace club_co_presidents_selection_l4053_405305

/-- The number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of members in the club --/
def club_members : ℕ := 18

/-- The number of co-presidents to be chosen --/
def co_presidents : ℕ := 3

theorem club_co_presidents_selection :
  choose club_members co_presidents = 816 := by
  sorry

end club_co_presidents_selection_l4053_405305


namespace plane_perpendicularity_l4053_405359

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (perp_planes : Plane → Plane → Prop)

-- State the theorem
theorem plane_perpendicularity 
  (m l : Line) (α β : Plane)
  (h1 : parallel m l)
  (h2 : perpendicular l β)
  (h3 : subset m α) :
  perp_planes α β :=
sorry

end plane_perpendicularity_l4053_405359


namespace joes_total_lift_weight_l4053_405325

/-- The total weight of Joe's two lifts is 600 pounds -/
theorem joes_total_lift_weight :
  ∀ (first_lift second_lift : ℕ),
  first_lift = 300 →
  2 * first_lift = second_lift + 300 →
  first_lift + second_lift = 600 :=
by
  sorry

end joes_total_lift_weight_l4053_405325


namespace total_outfits_count_l4053_405320

def total_shirts : ℕ := 8
def total_ties : ℕ := 6
def special_shirt_matches : ℕ := 3

def outfits_with_special_shirt : ℕ := special_shirt_matches
def outfits_with_other_shirts : ℕ := (total_shirts - 1) * total_ties

theorem total_outfits_count :
  outfits_with_special_shirt + outfits_with_other_shirts = 45 :=
by sorry

end total_outfits_count_l4053_405320


namespace slag_transport_theorem_l4053_405348

/-- Represents the daily transport capacity of a team in tons -/
structure TransportCapacity where
  daily : ℝ

/-- Represents a construction team -/
structure Team where
  capacity : TransportCapacity

/-- Represents the project parameters -/
structure Project where
  totalSlag : ℝ
  teamA : Team
  teamB : Team
  transportCost : ℝ

/-- The main theorem to prove -/
theorem slag_transport_theorem (p : Project) 
  (h1 : p.teamA.capacity.daily = p.teamB.capacity.daily * (5/3))
  (h2 : 4000 / p.teamA.capacity.daily + 2 = 3000 / p.teamB.capacity.daily)
  (h3 : p.totalSlag = 7000)
  (h4 : ∃ m : ℝ, 
    (p.teamA.capacity.daily + m) * 7 + 
    (p.teamB.capacity.daily + m/300) * 9 = p.totalSlag) :
  p.teamA.capacity.daily = 500 ∧ 
  (p.teamB.capacity.daily + (50/300)) * 9 * p.transportCost = 157500 := by
  sorry

#check slag_transport_theorem

end slag_transport_theorem_l4053_405348


namespace people_in_room_l4053_405368

theorem people_in_room (total_chairs : ℚ) (occupied_chairs : ℚ) (empty_chairs : ℚ) 
  (h1 : empty_chairs = 5)
  (h2 : occupied_chairs = (2/3) * total_chairs)
  (h3 : empty_chairs = (1/3) * total_chairs)
  (h4 : occupied_chairs = 10) :
  ∃ (total_people : ℚ), total_people = 50/3 ∧ (3/5) * total_people = occupied_chairs := by
  sorry

end people_in_room_l4053_405368


namespace set_operations_l4053_405346

-- Define the sets A and B
def A : Set ℝ := {y | -1 < y ∧ y < 4}
def B : Set ℝ := {y | 0 < y ∧ y < 5}

-- Theorem statements
theorem set_operations :
  (Set.univ \ B = {y | y ≤ 0 ∨ y ≥ 5}) ∧
  (A ∪ B = {y | -1 < y ∧ y < 5}) ∧
  (A ∩ B = {y | 0 < y ∧ y < 4}) ∧
  (A ∩ (Set.univ \ B) = {y | -1 < y ∧ y ≤ 0}) ∧
  ((Set.univ \ A) ∩ (Set.univ \ B) = {y | y ≤ -1 ∨ y ≥ 5}) := by
  sorry

end set_operations_l4053_405346


namespace equation_solution_l4053_405347

theorem equation_solution :
  ∃! x : ℝ, 
    x > 10 ∧
    (8 / (Real.sqrt (x - 10) - 10) + 
     2 / (Real.sqrt (x - 10) - 5) + 
     9 / (Real.sqrt (x - 10) + 5) + 
     15 / (Real.sqrt (x - 10) + 10) = 0) ∧
    x = 35 := by
  sorry

end equation_solution_l4053_405347


namespace five_people_arrangement_l4053_405399

/-- The number of ways to arrange n people in a line -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n-1 people in a line -/
def arrangements_without_youngest (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The number of valid arrangements for n people where the youngest cannot be first or last -/
def validArrangements (n : ℕ) : ℕ :=
  totalArrangements n - 2 * arrangements_without_youngest n

theorem five_people_arrangement :
  validArrangements 5 = 72 := by
  sorry

end five_people_arrangement_l4053_405399


namespace symmetric_point_y_axis_coordinates_l4053_405302

/-- A point in a 2D coordinate system. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines symmetry with respect to the y-axis. -/
def symmetricPointYAxis (p : Point) : Point :=
  { x := -p.x, y := p.y }

theorem symmetric_point_y_axis_coordinates :
  let B : Point := { x := -3, y := 4 }
  let A : Point := symmetricPointYAxis B
  A.x = 3 ∧ A.y = 4 := by
  sorry

end symmetric_point_y_axis_coordinates_l4053_405302


namespace arithmetic_sequence_difference_l4053_405372

def arithmetic_sequence (a : ℕ → ℝ) := 
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_difference 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_sum1 : a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 81)
  (h_sum2 : a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 = 171) :
  ∃ d : ℝ, d = 10 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by sorry

end arithmetic_sequence_difference_l4053_405372


namespace equation_solutions_l4053_405395

theorem equation_solutions :
  (∃ x : ℚ, 4 - 3 * x = 6 - 5 * x ∧ x = 1) ∧
  (∃ x : ℚ, 7 - 3 * (x - 1) = -x ∧ x = 5) ∧
  (∃ x : ℚ, (3 * x - 1) / 2 = 1 - (x - 1) / 6 ∧ x = 1) ∧
  (∃ x : ℚ, (2 * x - 1) / 3 - x = (2 * x + 1) / 4 - 1 ∧ x = 1/2) :=
by sorry

end equation_solutions_l4053_405395


namespace probability_of_two_boys_l4053_405390

theorem probability_of_two_boys (total : ℕ) (boys : ℕ) (girls : ℕ) 
  (h1 : total = 12) 
  (h2 : boys = 8) 
  (h3 : girls = 4) 
  (h4 : total = boys + girls) : 
  (Nat.choose boys 2 : ℚ) / (Nat.choose total 2) = 14/33 :=
by sorry

end probability_of_two_boys_l4053_405390
