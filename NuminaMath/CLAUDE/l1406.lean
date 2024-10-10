import Mathlib

namespace fraction_modification_l1406_140677

theorem fraction_modification (p q r s x : ℚ) : 
  p ≠ q → q ≠ 0 → p = 3 → q = 5 → r = 7 → s = 9 → (p + x) / (q - x) = r / s → x = 1/2 := by
  sorry

end fraction_modification_l1406_140677


namespace carl_josh_wage_ratio_l1406_140689

/-- Represents the hourly wage ratio between Carl and Josh -/
def wage_ratio : ℚ := 1 / 2

theorem carl_josh_wage_ratio : 
  let josh_hours_per_day : ℕ := 8
  let josh_days_per_week : ℕ := 5
  let josh_weeks_per_month : ℕ := 4
  let carl_hours_less_per_day : ℕ := 2
  let josh_hourly_wage : ℚ := 9
  let total_monthly_pay : ℚ := 1980
  
  let josh_monthly_hours : ℕ := josh_hours_per_day * josh_days_per_week * josh_weeks_per_month
  let carl_monthly_hours : ℕ := (josh_hours_per_day - carl_hours_less_per_day) * josh_days_per_week * josh_weeks_per_month
  let josh_monthly_pay : ℚ := josh_hourly_wage * josh_monthly_hours
  let carl_monthly_pay : ℚ := total_monthly_pay - josh_monthly_pay
  let carl_hourly_wage : ℚ := carl_monthly_pay / carl_monthly_hours

  carl_hourly_wage / josh_hourly_wage = wage_ratio :=
by
  sorry

#check carl_josh_wage_ratio

end carl_josh_wage_ratio_l1406_140689


namespace dog_meal_amount_proof_l1406_140672

/-- The amount of food a dog eats at each meal, in pounds -/
def dog_meal_amount : ℝ := 4

/-- The number of puppies -/
def num_puppies : ℕ := 4

/-- The number of dogs -/
def num_dogs : ℕ := 3

/-- The number of times a dog eats per day -/
def dog_meals_per_day : ℕ := 3

/-- The total amount of food eaten by dogs and puppies in a day, in pounds -/
def total_food_per_day : ℝ := 108

theorem dog_meal_amount_proof :
  dog_meal_amount * num_dogs * dog_meals_per_day + 
  (dog_meal_amount / 2) * num_puppies * (3 * dog_meals_per_day) = total_food_per_day :=
by sorry

end dog_meal_amount_proof_l1406_140672


namespace milk_tea_sales_ratio_l1406_140657

theorem milk_tea_sales_ratio (total_sales : ℕ) (okinawa_ratio : ℚ) (chocolate_sales : ℕ) : 
  total_sales = 50 →
  okinawa_ratio = 3 / 10 →
  chocolate_sales = 15 →
  (total_sales - (okinawa_ratio * total_sales).num - chocolate_sales) * 5 = total_sales * 2 := by
  sorry

end milk_tea_sales_ratio_l1406_140657


namespace quadratic_inequality_solution_l1406_140608

theorem quadratic_inequality_solution (x : ℝ) : x^2 + 5*x < 10 ↔ -5 < x ∧ x < 2 := by
  sorry

end quadratic_inequality_solution_l1406_140608


namespace stocking_price_calculation_l1406_140616

/-- The original price of a stocking before discount -/
def original_price : ℝ := 122.22

/-- The number of stockings ordered -/
def num_stockings : ℕ := 9

/-- The discount rate applied to the stockings -/
def discount_rate : ℝ := 0.1

/-- The cost of monogramming per stocking -/
def monogram_cost : ℝ := 5

/-- The total cost after discount and including monogramming -/
def total_cost : ℝ := 1035

/-- Theorem stating that the calculated original price satisfies the given conditions -/
theorem stocking_price_calculation :
  total_cost = num_stockings * (original_price * (1 - discount_rate) + monogram_cost) :=
by sorry

end stocking_price_calculation_l1406_140616


namespace bus_passing_time_l1406_140622

theorem bus_passing_time (distance : ℝ) (time : ℝ) (bus_length : ℝ) : 
  distance = 12 → time = 5 → bus_length = 200 →
  (bus_length / (distance * 1000 / (time * 60))) = 5 := by
  sorry

end bus_passing_time_l1406_140622


namespace sqrt_eighteen_minus_sqrt_eight_equals_sqrt_two_l1406_140687

theorem sqrt_eighteen_minus_sqrt_eight_equals_sqrt_two :
  Real.sqrt 18 - Real.sqrt 8 = Real.sqrt 2 := by
  sorry

end sqrt_eighteen_minus_sqrt_eight_equals_sqrt_two_l1406_140687


namespace plant_arrangement_count_l1406_140601

/-- Represents the number of basil plants -/
def basil_count : ℕ := 5

/-- Represents the number of tomato plants -/
def tomato_count : ℕ := 5

/-- Represents the total number of plant positions (basil + tomato block) -/
def total_positions : ℕ := basil_count + 1

/-- Calculates the number of ways to arrange the plants with given constraints -/
def plant_arrangements : ℕ :=
  (Nat.factorial total_positions) * (Nat.factorial tomato_count)

theorem plant_arrangement_count :
  plant_arrangements = 86400 := by sorry

end plant_arrangement_count_l1406_140601


namespace algebraic_expression_value_l1406_140649

theorem algebraic_expression_value (a b : ℝ) (h : a = b + 1) :
  a^2 - 2*a*b + b^2 + 2 = 3 := by
  sorry

end algebraic_expression_value_l1406_140649


namespace chocolate_milk_probability_l1406_140650

theorem chocolate_milk_probability : 
  let n : ℕ := 7  -- total number of days
  let k : ℕ := 5  -- number of days with chocolate milk
  let p : ℚ := 1/2  -- probability of bottling chocolate milk each day
  (n.choose k) * p^k * (1-p)^(n-k) = 21/128 := by
sorry

end chocolate_milk_probability_l1406_140650


namespace train_passing_length_l1406_140626

/-- The length of a train passing another train in opposite direction -/
theorem train_passing_length (v1 v2 : ℝ) (t : ℝ) (h1 : v1 = 50) (h2 : v2 = 62) (h3 : t = 9) :
  let relative_speed := (v1 + v2) * (1000 / 3600)
  let train_length := relative_speed * t
  ∃ ε > 0, |train_length - 280| < ε :=
by sorry

end train_passing_length_l1406_140626


namespace hotel_bill_friends_count_prove_hotel_bill_friends_count_l1406_140638

theorem hotel_bill_friends_count : ℕ → Prop :=
  fun total_friends =>
    let standard_pay := 100
    let extra_pay := 100
    let actual_extra_pay := 220
    let standard_payers := 5
    let total_bill := standard_payers * standard_pay + extra_pay
    let share_per_friend := total_bill / total_friends
    total_friends = standard_payers + 1 ∧
    share_per_friend * total_friends = total_bill ∧
    share_per_friend + extra_pay = actual_extra_pay

theorem prove_hotel_bill_friends_count : 
  ∃ (n : ℕ), hotel_bill_friends_count n ∧ n = 6 := by
  sorry

end hotel_bill_friends_count_prove_hotel_bill_friends_count_l1406_140638


namespace concentric_circles_angle_l1406_140619

theorem concentric_circles_angle (r₁ r₂ r₃ : ℝ) (shaded_area unshaded_area : ℝ) (θ : ℝ) : 
  r₁ = 4 →
  r₂ = 3 →
  r₃ = 2 →
  shaded_area = (3/4) * unshaded_area →
  shaded_area + unshaded_area = 29 * π →
  shaded_area = 11 * θ + 9 * π →
  θ = 6 * π / 77 :=
by sorry

end concentric_circles_angle_l1406_140619


namespace racket_price_l1406_140611

theorem racket_price (total_spent : ℚ) (h1 : total_spent = 90) : ∃ (original_price : ℚ),
  original_price + original_price / 2 = total_spent ∧ original_price = 60 :=
by
  sorry

end racket_price_l1406_140611


namespace no_real_solutions_for_equation_l1406_140631

theorem no_real_solutions_for_equation : 
  ¬ ∃ x : ℝ, (x + 4)^2 = 3*(x - 2) := by
sorry

end no_real_solutions_for_equation_l1406_140631


namespace sum_of_a_and_b_l1406_140624

theorem sum_of_a_and_b (a b : ℝ) 
  (h1 : a^2 = 16) 
  (h2 : b^3 = -27) 
  (h3 : |a - b| = a - b) : 
  a + b = 1 := by
sorry

end sum_of_a_and_b_l1406_140624


namespace baker_remaining_cakes_l1406_140636

/-- The number of cakes Baker initially made -/
def initial_cakes : ℕ := 48

/-- The number of cakes Baker sold -/
def sold_cakes : ℕ := 44

/-- Theorem: Baker still has 4 cakes -/
theorem baker_remaining_cakes : initial_cakes - sold_cakes = 4 := by
  sorry

end baker_remaining_cakes_l1406_140636


namespace prob_adjacent_20_3_l1406_140686

/-- The number of people sitting at the round table -/
def n : ℕ := 20

/-- The number of specific people we're interested in -/
def k : ℕ := 3

/-- The probability of at least two out of three specific people sitting next to each other
    in a random seating arrangement of n people at a round table -/
def prob_adjacent (n k : ℕ) : ℚ :=
  17/57

/-- Theorem stating the probability for the given problem -/
theorem prob_adjacent_20_3 : prob_adjacent n k = 17/57 := by
  sorry

end prob_adjacent_20_3_l1406_140686


namespace intersection_point_l1406_140640

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 9*x + 15

theorem intersection_point (a b : ℝ) :
  (∀ x ≠ a, f x ≠ f a) ∧ 
  f a = b ∧ 
  f b = a ∧ 
  (∀ x y, f x = y ∧ f y = x → x = a ∧ y = b) →
  a = -1 ∧ b = -1 := by
sorry

end intersection_point_l1406_140640


namespace unique_solution_condition_l1406_140690

/-- The equation (3x+8)(x-6) = -52 + kx has exactly one real solution if and only if k = 4√3 - 10 or k = -4√3 - 10 -/
theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (3*x+8)*(x-6) = -52 + k*x) ↔ (k = 4*Real.sqrt 3 - 10 ∨ k = -4*Real.sqrt 3 - 10) := by
sorry


end unique_solution_condition_l1406_140690


namespace bridget_initial_skittles_l1406_140647

/-- Proves that Bridget initially has 4 Skittles given the problem conditions. -/
theorem bridget_initial_skittles : 
  ∀ (bridget_initial henry_skittles bridget_final : ℕ),
  henry_skittles = 4 →
  bridget_final = bridget_initial + henry_skittles →
  bridget_final = 8 →
  bridget_initial = 4 := by sorry

end bridget_initial_skittles_l1406_140647


namespace car_original_price_l1406_140661

/-- Given a car sold at a 15% loss and then resold with a 20% gain for Rs. 54000,
    prove that the original cost price of the car was Rs. 52,941.18 (rounded to two decimal places). -/
theorem car_original_price (loss_percent : ℝ) (gain_percent : ℝ) (final_price : ℝ) :
  loss_percent = 15 →
  gain_percent = 20 →
  final_price = 54000 →
  ∃ (original_price : ℝ),
    (1 - loss_percent / 100) * original_price * (1 + gain_percent / 100) = final_price ∧
    (round (original_price * 100) / 100 : ℝ) = 52941.18 := by
  sorry

end car_original_price_l1406_140661


namespace am_gm_inequality_l1406_140646

theorem am_gm_inequality (x : ℝ) (h : x > 0) : x + 1/x ≥ 2 := by
  sorry

end am_gm_inequality_l1406_140646


namespace quadratic_equal_roots_l1406_140681

theorem quadratic_equal_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 + (m-2)*x + m + 1 = 0 ∧ 
   ∀ y : ℝ, y^2 + (m-2)*y + m + 1 = 0 → y = x) ↔ 
  m = 0 ∨ m = 8 := by
sorry

end quadratic_equal_roots_l1406_140681


namespace digit_sum_to_100_l1406_140637

def digits : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9]

def insert_operators (ds : List Nat) : List (Option Bool) :=
  [none, some true, some true, some false, some false, some false, some false, some true, some false]

def evaluate (ds : List Nat) (ops : List (Option Bool)) : Int :=
  match ds, ops with
  | [], _ => 0
  | d :: ds', none :: ops' => d * 100 + evaluate ds' ops'
  | d :: ds', some true :: ops' => d + evaluate ds' ops'
  | d :: ds', some false :: ops' => -d + evaluate ds' ops'
  | _, _ => 0

theorem digit_sum_to_100 :
  ∃ (ops : List (Option Bool)), evaluate digits ops = 100 :=
sorry

end digit_sum_to_100_l1406_140637


namespace bus_seats_columns_l1406_140643

/-- The number of rows in each bus -/
def rows : ℕ := 10

/-- The number of buses -/
def buses : ℕ := 6

/-- The total number of students that can be accommodated -/
def total_students : ℕ := 240

/-- The number of columns of seats in each bus -/
def columns : ℕ := 4

theorem bus_seats_columns :
  columns * rows * buses = total_students :=
sorry

end bus_seats_columns_l1406_140643


namespace expand_product_l1406_140648

theorem expand_product (x : ℝ) : (x + 4) * (x - 7) = x^2 - 3*x - 28 := by
  sorry

end expand_product_l1406_140648


namespace parabola_properties_l1406_140693

-- Define the parabola
def parabola (x : ℝ) : ℝ := (x - 4)^2 - 5

-- Theorem statement
theorem parabola_properties :
  (∃ (x y : ℝ), y = parabola x ∧ ∀ (x' : ℝ), parabola x' ≥ y) ∧
  (∀ (x₁ x₂ : ℝ), x₁ < 4 ∧ x₂ > 4 → parabola x₁ > parabola 4 ∧ parabola x₂ > parabola 4) :=
by sorry

end parabola_properties_l1406_140693


namespace symmetric_angle_set_l1406_140691

/-- Given α = π/6 and the terminal side of angle β is symmetric to the terminal side of α
    with respect to the line y=x, prove that the set of all possible values for β
    is {β | β = 2kπ + π/3, k ∈ ℤ}. -/
theorem symmetric_angle_set (α β : Real) (k : ℤ) :
  α = π / 6 →
  (∃ (f : Real → Real), f β = α ∧ f (π / 4) = π / 4 ∧ ∀ x, f (f x) = x) →
  (β = 2 * π * k + π / 3) :=
sorry

end symmetric_angle_set_l1406_140691


namespace proposition_implication_l1406_140667

theorem proposition_implication (P : ℕ → Prop) 
  (h1 : ∀ k : ℕ, k > 0 → (P k → P (k + 1)))
  (h2 : ¬ P 5) : 
  ¬ P 4 :=
sorry

end proposition_implication_l1406_140667


namespace probability_two_female_volunteers_l1406_140694

/-- The probability of selecting 2 female volunteers from a group of 3 female and 2 male volunteers (5 in total) is 3/10. -/
theorem probability_two_female_volunteers :
  let total_volunteers : ℕ := 5
  let female_volunteers : ℕ := 3
  let male_volunteers : ℕ := 2
  let selected_volunteers : ℕ := 2
  let total_combinations := Nat.choose total_volunteers selected_volunteers
  let female_combinations := Nat.choose female_volunteers selected_volunteers
  (female_combinations : ℚ) / total_combinations = 3 / 10 := by
  sorry

end probability_two_female_volunteers_l1406_140694


namespace max_sum_constrained_l1406_140618

theorem max_sum_constrained (x y : ℝ) : 
  x^2 + y^2 = 100 → xy = 40 → x + y ≤ 6 * Real.sqrt 5 := by
  sorry

end max_sum_constrained_l1406_140618


namespace bargain_bin_books_l1406_140670

theorem bargain_bin_books (initial_books : ℕ) : 
  initial_books - 3 + 10 = 11 → initial_books = 4 := by
  sorry

end bargain_bin_books_l1406_140670


namespace orange_juice_fraction_l1406_140645

theorem orange_juice_fraction : 
  let pitcher1_capacity : ℚ := 800
  let pitcher2_capacity : ℚ := 700
  let pitcher1_juice_fraction : ℚ := 1/4
  let pitcher2_juice_fraction : ℚ := 3/7
  let total_juice := pitcher1_capacity * pitcher1_juice_fraction + pitcher2_capacity * pitcher2_juice_fraction
  let total_volume := pitcher1_capacity + pitcher2_capacity
  total_juice / total_volume = 1/3 := by sorry

end orange_juice_fraction_l1406_140645


namespace solution_set_when_a_is_one_range_of_a_when_x_in_open_interval_l1406_140671

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| - |a * x - 1|

-- Part 1
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x > 1} = {x : ℝ | x > 1/2} := by sorry

-- Part 2
theorem range_of_a_when_x_in_open_interval :
  {a : ℝ | ∀ x ∈ Set.Ioo 0 1, f a x > x} = Set.Ioc 0 2 := by sorry

end solution_set_when_a_is_one_range_of_a_when_x_in_open_interval_l1406_140671


namespace divisible_by_eleven_l1406_140614

theorem divisible_by_eleven (n : ℕ) : n < 10 → (123 * 100000 + n * 1000 + 789) % 11 = 0 ↔ n = 10 % 11 := by
  sorry

end divisible_by_eleven_l1406_140614


namespace triangle_sine_sum_zero_l1406_140665

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  pos_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π
  law_of_sines : a / Real.sin A = b / Real.sin B
  
-- State the theorem
theorem triangle_sine_sum_zero (t : Triangle) : 
  t.a^3 * Real.sin (t.B - t.C) + t.b^3 * Real.sin (t.C - t.A) + t.c^3 * Real.sin (t.A - t.B) = 0 :=
sorry

end triangle_sine_sum_zero_l1406_140665


namespace least_n_divisibility_l1406_140627

theorem least_n_divisibility (n : ℕ) : n = 5 ↔ 
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 2*n → 
    (∃ m : ℕ, m ≥ 1 ∧ m ≤ 2*n ∧ (n^2 - n + m) % m = 0) ∧ 
    (∃ l : ℕ, l ≥ 1 ∧ l ≤ 2*n ∧ (n^2 - n + l) % l ≠ 0)) ∧
  (∀ m : ℕ, m < n → 
    ¬(∀ k : ℕ, 1 ≤ k ∧ k ≤ 2*m → 
      (∃ p : ℕ, p ≥ 1 ∧ p ≤ 2*m ∧ (m^2 - m + p) % p = 0) ∧ 
      (∃ q : ℕ, q ≥ 1 ∧ q ≤ 2*m ∧ (m^2 - m + q) % q ≠ 0))) :=
by sorry

end least_n_divisibility_l1406_140627


namespace f_min_at_inv_e_l1406_140653

open Real

noncomputable def f (x : ℝ) : ℝ := x * log x

theorem f_min_at_inv_e :
  ∀ x > 0, f (1 / Real.exp 1) ≤ f x :=
by sorry

end f_min_at_inv_e_l1406_140653


namespace equal_area_point_on_diagonal_l1406_140684

/-- A point inside a rectangle where lines through it parallel to the sides create equal-area subrectangles -/
structure EqualAreaPoint (a b : ℝ) where
  x : ℝ
  y : ℝ
  x_bounds : 0 < x ∧ x < a
  y_bounds : 0 < y ∧ y < b
  equal_areas : x * y = (a - x) * y ∧ x * (b - y) = (a - x) * (b - y)

/-- The diagonals of a rectangle -/
def rectangleDiagonals (a b : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ (p = (t * a, t * b) ∨ p = ((1 - t) * a, t * b))}

/-- Theorem: Points satisfying the equal area condition lie on the diagonals of the rectangle -/
theorem equal_area_point_on_diagonal (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
    (p : EqualAreaPoint a b) : (p.x, p.y) ∈ rectangleDiagonals a b := by
  sorry

end equal_area_point_on_diagonal_l1406_140684


namespace max_girls_for_five_boys_valid_arrangement_l1406_140669

/-- The maximum number of girls that can be arranged in a "Mathematical Ballet" -/
def max_girls (num_boys : ℕ) : ℕ :=
  (num_boys.choose 2) * 2

/-- Theorem stating the maximum number of girls for 5 boys -/
theorem max_girls_for_five_boys :
  max_girls 5 = 20 := by
  sorry

/-- Theorem proving the validity of the arrangement -/
theorem valid_arrangement (num_boys : ℕ) (num_girls : ℕ) :
  num_girls ≤ max_girls num_boys →
  ∃ (boy_positions : Fin num_boys → ℝ × ℝ)
    (girl_positions : Fin num_girls → ℝ × ℝ),
    ∀ (g : Fin num_girls),
      ∃ (b1 b2 : Fin num_boys),
        b1 ≠ b2 ∧
        dist (girl_positions g) (boy_positions b1) = 5 ∧
        dist (girl_positions g) (boy_positions b2) = 5 ∧
        ∀ (b : Fin num_boys),
          b ≠ b1 ∧ b ≠ b2 →
          dist (girl_positions g) (boy_positions b) ≠ 5 := by
  sorry


end max_girls_for_five_boys_valid_arrangement_l1406_140669


namespace h_increasing_implies_k_range_l1406_140617

def h (k : ℝ) (x : ℝ) : ℝ := 2 * x - k

theorem h_increasing_implies_k_range (k : ℝ) :
  (∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → h k x₁ < h k x₂) →
  k ∈ Set.Ici (-2) :=
sorry

end h_increasing_implies_k_range_l1406_140617


namespace sqrt_product_equality_l1406_140609

theorem sqrt_product_equality : 
  2 * Real.sqrt 3 * (1.5 ^ (1/3)) * (12 ^ (1/6)) = 6 := by sorry

end sqrt_product_equality_l1406_140609


namespace nigels_money_l1406_140633

theorem nigels_money (initial_amount : ℕ) (given_away : ℕ) (final_amount : ℕ) : 
  initial_amount = 45 →
  given_away = 25 →
  final_amount = 2 * initial_amount + 10 →
  final_amount - (initial_amount - given_away) = 80 :=
by
  sorry

end nigels_money_l1406_140633


namespace quadratic_equation_with_irrational_root_l1406_140632

theorem quadratic_equation_with_irrational_root :
  ∃ (a b c : ℚ), a ≠ 0 ∧ 
  (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = 2 * Real.sqrt 5 - 3) ∧
  a = 1 ∧ b = 6 ∧ c = -11 :=
sorry

end quadratic_equation_with_irrational_root_l1406_140632


namespace orange_groups_indeterminate_philips_collection_valid_philips_orange_groups_indeterminate_l1406_140623

/-- Represents a fruit collection with oranges and bananas -/
structure FruitCollection where
  oranges : ℕ
  bananas : ℕ
  banana_groups : ℕ
  bananas_per_group : ℕ

/-- Predicate to check if the banana distribution is valid -/
def valid_banana_distribution (fc : FruitCollection) : Prop :=
  fc.bananas = fc.banana_groups * fc.bananas_per_group

/-- Theorem stating that the number of orange groups cannot be determined -/
theorem orange_groups_indeterminate (fc : FruitCollection) 
  (h1 : fc.oranges > 0)
  (h2 : valid_banana_distribution fc) :
  ¬ ∃ (orange_groups : ℕ), orange_groups > 0 ∧ ∀ (oranges_per_group : ℕ), fc.oranges = orange_groups * oranges_per_group :=
by
  sorry

/-- Philip's fruit collection -/
def philips_collection : FruitCollection :=
  { oranges := 87
  , bananas := 290
  , banana_groups := 2
  , bananas_per_group := 145 }

/-- Proof that Philip's collection satisfies the conditions -/
theorem philips_collection_valid :
  valid_banana_distribution philips_collection :=
by
  sorry

/-- Application of the theorem to Philip's collection -/
theorem philips_orange_groups_indeterminate :
  ¬ ∃ (orange_groups : ℕ), orange_groups > 0 ∧ ∀ (oranges_per_group : ℕ), philips_collection.oranges = orange_groups * oranges_per_group :=
by
  apply orange_groups_indeterminate
  · simp [philips_collection]
  · exact philips_collection_valid

end orange_groups_indeterminate_philips_collection_valid_philips_orange_groups_indeterminate_l1406_140623


namespace article_choice_correct_l1406_140621

-- Define the possible article choices
inductive Article
  | A
  | The
  | None

-- Define the structure for an article combination
structure ArticleCombination where
  first : Article
  second : Article

-- Define the conditions of the problem
def is_general_reference (a : Article) : Prop :=
  a = Article.A

def is_specific_reference (a : Article) : Prop :=
  a = Article.The

-- Define the correct combination
def correct_combination : ArticleCombination :=
  { first := Article.A, second := Article.The }

-- Theorem to prove
theorem article_choice_correct
  (german_engineer_general : is_general_reference correct_combination.first)
  (car_invention_specific : is_specific_reference correct_combination.second) :
  correct_combination = { first := Article.A, second := Article.The } := by
  sorry

end article_choice_correct_l1406_140621


namespace installation_solution_l1406_140663

/-- Represents the number of installations of each type -/
structure Installations where
  type1 : ℕ
  type2 : ℕ
  type3 : ℕ

/-- Checks if the given installation numbers satisfy all conditions -/
def satisfiesConditions (i : Installations) : Prop :=
  i.type1 + i.type2 + i.type3 ≥ 100 ∧
  i.type2 = 4 * i.type1 ∧
  ∃ k : ℕ, i.type3 = k * i.type1 ∧
  5 * i.type3 = i.type2 + 22

theorem installation_solution :
  ∃ i : Installations, satisfiesConditions i ∧ i.type1 = 22 ∧ i.type2 = 88 ∧ i.type3 = 22 :=
by sorry

end installation_solution_l1406_140663


namespace largest_whole_number_satisfying_inequality_l1406_140615

theorem largest_whole_number_satisfying_inequality :
  ∀ x : ℤ, (1/4 : ℚ) + (x : ℚ)/8 < 1 → x ≤ 5 ∧
  ((1/4 : ℚ) + (5 : ℚ)/8 < 1 ∧ ∀ y : ℤ, y > 5 → (1/4 : ℚ) + (y : ℚ)/8 ≥ 1) :=
by sorry

end largest_whole_number_satisfying_inequality_l1406_140615


namespace alien_rock_count_l1406_140630

/-- Converts a three-digit number in base 7 to base 10 --/
def base7ToBase10 (hundreds tens units : ℕ) : ℕ :=
  hundreds * 7^2 + tens * 7^1 + units * 7^0

/-- The number of rocks seen by the alien --/
def alienRocks : ℕ := base7ToBase10 3 5 1

theorem alien_rock_count : alienRocks = 183 := by sorry

end alien_rock_count_l1406_140630


namespace m_range_l1406_140660

theorem m_range (m : ℝ) : 
  (∀ x : ℝ, x^2 + m*x + 1 ≠ 0 ∨ ∃ y : ℝ, y ≠ x ∧ y^2 + m*y + 1 = 0 → False) →
  (∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0) →
  1 < m ∧ m ≤ 2 :=
sorry

end m_range_l1406_140660


namespace perfect_square_base_l1406_140662

theorem perfect_square_base : ∃! (d : ℕ), d > 1 ∧ ∃ (n : ℕ), d^4 + d^3 + d^2 + d + 1 = n^2 :=
sorry

end perfect_square_base_l1406_140662


namespace line_equation_l1406_140652

-- Define the lines
def line1 (x y : ℝ) : Prop := x - 2*y + 4 = 0
def line2 (x y : ℝ) : Prop := x + y - 2 = 0
def line3 (x y : ℝ) : Prop := x + 3*y + 5 = 0

-- Define the intersection point
def intersection_point : ℝ × ℝ :=
  (0, 2)

-- Define the perpendicularity condition
def perpendicular (m1 m2 : ℝ) : Prop :=
  m1 * m2 = -1

-- Theorem statement
theorem line_equation : 
  ∃ (A B C : ℝ), 
    (A ≠ 0 ∨ B ≠ 0) ∧ 
    (∀ x y : ℝ, A*x + B*y + C = 0 ↔ 
      (line1 x y ∧ line2 x y) ∨
      (x = intersection_point.1 ∧ y = intersection_point.2) ∨
      (∃ m : ℝ, perpendicular m (-1/3) ∧ y - intersection_point.2 = m * (x - intersection_point.1))) ∧
    A = 3 ∧ B = -1 ∧ C = 2 :=
  sorry

end line_equation_l1406_140652


namespace modulus_of_z_l1406_140658

-- Define the complex number z
def z : ℂ := 3 - 4 * Complex.I

-- Theorem statement
theorem modulus_of_z : Complex.abs z = 5 := by sorry

end modulus_of_z_l1406_140658


namespace tan_value_of_sequences_l1406_140666

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

def is_arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

theorem tan_value_of_sequences (a b : ℕ → ℝ) :
  is_geometric_sequence a →
  is_arithmetic_sequence b →
  a 1 - a 6 - a 11 = -3 * Real.sqrt 3 →
  b 1 + b 6 + b 11 = 7 * Real.pi →
  Real.tan ((b 3 + b 9) / (1 - a 4 - a 3)) = -Real.sqrt 3 :=
by sorry

end tan_value_of_sequences_l1406_140666


namespace total_days_2010_to_2014_l1406_140641

def days_in_year (year : ℕ) : ℕ :=
  if year = 2012 then 366 else 365

def years_range : List ℕ := [2010, 2011, 2012, 2013, 2014]

theorem total_days_2010_to_2014 :
  (years_range.map days_in_year).sum = 1826 := by sorry

end total_days_2010_to_2014_l1406_140641


namespace inequality_proof_l1406_140604

theorem inequality_proof (a b : ℝ) (h : 1/a < 1/b ∧ 1/b < 0) : b/a + a/b > 2 := by
  sorry

end inequality_proof_l1406_140604


namespace expected_adjacent_red_pairs_l1406_140692

theorem expected_adjacent_red_pairs (total_cards : ℕ) (red_cards : ℕ) 
  (h1 : total_cards = 40) (h2 : red_cards = 20) :
  let prob_red_after_red := (red_cards - 1) / (total_cards - 1)
  let expected_pairs := red_cards * prob_red_after_red
  expected_pairs = 380 / 39 := by
  sorry

end expected_adjacent_red_pairs_l1406_140692


namespace root_difference_implies_k_value_l1406_140625

theorem root_difference_implies_k_value (k : ℝ) : 
  (∀ x₁ x₂, x₁^2 + k*x₁ + 10 = 0 → x₂^2 - k*x₂ + 10 = 0 → x₂ = x₁ + 3) →
  k = 3 := by
sorry

end root_difference_implies_k_value_l1406_140625


namespace actual_weekly_earnings_increase_l1406_140668

/-- Calculates the actual increase in weekly earnings given a raise, work hours, and housing benefit reduction. -/
theorem actual_weekly_earnings_increase
  (hourly_raise : ℝ)
  (weekly_hours : ℝ)
  (monthly_benefit_reduction : ℝ)
  (h1 : hourly_raise = 0.50)
  (h2 : weekly_hours = 40)
  (h3 : monthly_benefit_reduction = 60)
  : ∃ (actual_increase : ℝ), abs (actual_increase - 6.14) < 0.01 := by
  sorry

#check actual_weekly_earnings_increase

end actual_weekly_earnings_increase_l1406_140668


namespace quadratic_roots_are_x_intercepts_ac_sign_not_guaranteed_l1406_140642

/-- Represents a quadratic function f(x) = ax^2 + bx + c --/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The roots of a quadratic function --/
def roots (f : QuadraticFunction) : Set ℝ :=
  {x : ℝ | f.a * x^2 + f.b * x + f.c = 0}

/-- The x-intercepts of a quadratic function --/
def xIntercepts (f : QuadraticFunction) : Set ℝ :=
  {x : ℝ | f.a * x^2 + f.b * x + f.c = 0}

theorem quadratic_roots_are_x_intercepts (f : QuadraticFunction) :
  roots f = xIntercepts f := by sorry

theorem ac_sign_not_guaranteed (f : QuadraticFunction) :
  ¬∀ f : QuadraticFunction, f.a * f.c < 0 := by sorry

end quadratic_roots_are_x_intercepts_ac_sign_not_guaranteed_l1406_140642


namespace inequality_proof_l1406_140635

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum_squares : x^2 + y^2 + z^2 = 1) :
  (1/x + 1/y + 1/z) - (x + y + z) ≥ 2 * Real.sqrt 3 := by
  sorry

end inequality_proof_l1406_140635


namespace inequality_range_proof_l1406_140696

theorem inequality_range_proof : 
  {x : ℝ | ∀ t : ℝ, |t - 3| + |2*t + 1| ≥ |2*x - 1| + |x + 2|} = 
  {x : ℝ | -1/2 ≤ x ∧ x ≤ 5/6} := by sorry

end inequality_range_proof_l1406_140696


namespace same_height_time_l1406_140675

/-- Represents the height of a ball as a function of time -/
def ball_height (a : ℝ) (h : ℝ) (t : ℝ) : ℝ := a * (t - 1.2)^2 + h

theorem same_height_time :
  ∀ (a : ℝ) (h : ℝ),
  a ≠ 0 →
  ∃ (t : ℝ),
  t = 2.2 ∧
  ball_height a h t = ball_height a h (t - 2) :=
sorry

end same_height_time_l1406_140675


namespace range_of_a_l1406_140695

theorem range_of_a (a : ℝ) : 
  (∀ x, x^2 - 8*x - 20 > 0 → x^2 - 2*x + 1 - a^2 > 0) ∧ 
  (∃ x, x^2 - 2*x + 1 - a^2 > 0 ∧ x^2 - 8*x - 20 ≤ 0) ∧
  (a > 0) →
  3 ≤ a ∧ a ≤ 9 :=
sorry

end range_of_a_l1406_140695


namespace equal_probabilities_l1406_140610

/-- Represents a box containing colored balls -/
structure Box where
  red : ℕ
  green : ℕ

/-- The initial state of the boxes -/
def initial_state : Box × Box :=
  (⟨100, 0⟩, ⟨0, 100⟩)

/-- The state after transferring 8 red balls to the green box -/
def after_first_transfer (state : Box × Box) : Box × Box :=
  let (red_box, green_box) := state
  (⟨red_box.red - 8, red_box.green⟩, ⟨green_box.red + 8, green_box.green⟩)

/-- The final state after transferring 8 balls back to the red box -/
def final_state (state : Box × Box) : Box × Box :=
  let (red_box, green_box) := after_first_transfer state
  (⟨red_box.red + 8, red_box.green + 8⟩, ⟨green_box.red - 8, green_box.green - 8⟩)

/-- The probability of drawing a specific color from a box -/
def prob_draw (box : Box) (color : String) : ℚ :=
  if color = "red" then
    box.red / (box.red + box.green)
  else
    box.green / (box.red + box.green)

theorem equal_probabilities :
  let (final_red_box, final_green_box) := final_state initial_state
  prob_draw final_red_box "green" = prob_draw final_green_box "red" := by
  sorry

end equal_probabilities_l1406_140610


namespace shirt_cost_is_9_l1406_140655

/-- The cost of one pair of jeans -/
def jeans_cost : ℝ := sorry

/-- The cost of one shirt -/
def shirt_cost : ℝ := sorry

/-- First condition: 3 pairs of jeans and 2 shirts cost $69 -/
axiom condition1 : 3 * jeans_cost + 2 * shirt_cost = 69

/-- Second condition: 2 pairs of jeans and 3 shirts cost $61 -/
axiom condition2 : 2 * jeans_cost + 3 * shirt_cost = 61

/-- Theorem: The cost of one shirt is $9 -/
theorem shirt_cost_is_9 : shirt_cost = 9 := by sorry

end shirt_cost_is_9_l1406_140655


namespace intersection_condition_l1406_140699

-- Define the curves
def curve1 (b x y : ℝ) : Prop := x^2 + y^2 = 2 * b^2
def curve2 (b x y : ℝ) : Prop := y = x^2 - b

-- Define the intersection condition
def intersect_at_four_points (b : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ),
    (curve1 b x1 y1 ∧ curve2 b x1 y1) ∧
    (curve1 b x2 y2 ∧ curve2 b x2 y2) ∧
    (curve1 b x3 y3 ∧ curve2 b x3 y3) ∧
    (curve1 b x4 y4 ∧ curve2 b x4 y4) ∧
    (x1 ≠ x2 ∨ y1 ≠ y2) ∧
    (x1 ≠ x3 ∨ y1 ≠ y3) ∧
    (x1 ≠ x4 ∨ y1 ≠ y4) ∧
    (x2 ≠ x3 ∨ y2 ≠ y3) ∧
    (x2 ≠ x4 ∨ y2 ≠ y4) ∧
    (x3 ≠ x4 ∨ y3 ≠ y4) ∧
    ∀ (x y : ℝ), (curve1 b x y ∧ curve2 b x y) →
      ((x = x1 ∧ y = y1) ∨ (x = x2 ∧ y = y2) ∨ (x = x3 ∧ y = y3) ∨ (x = x4 ∧ y = y4))

-- State the theorem
theorem intersection_condition (b : ℝ) :
  intersect_at_four_points b ↔ b > 1/2 := by sorry

end intersection_condition_l1406_140699


namespace abs_plus_exp_zero_equals_three_l1406_140697

theorem abs_plus_exp_zero_equals_three :
  |(-2 : ℝ)| + (3 - Real.sqrt 5) ^ (0 : ℕ) = 3 := by
  sorry

end abs_plus_exp_zero_equals_three_l1406_140697


namespace line_canonical_to_general_equations_l1406_140620

/-- Given a line in 3D space defined by canonical equations, prove that the general equations are equivalent. -/
theorem line_canonical_to_general_equations :
  ∀ (x y z : ℝ),
  ((x - 2) / 3 = (y + 1) / 5 ∧ (x - 2) / 3 = (z - 3) / (-1)) ↔
  (5 * x - 3 * y = 13 ∧ x + 3 * z = 11) :=
by sorry

end line_canonical_to_general_equations_l1406_140620


namespace apple_percentage_after_adding_oranges_l1406_140679

def initial_apples : ℕ := 10
def initial_oranges : ℕ := 5
def added_oranges : ℕ := 5

def total_fruits : ℕ := initial_apples + initial_oranges + added_oranges

theorem apple_percentage_after_adding_oranges :
  (initial_apples : ℚ) / total_fruits * 100 = 50 := by
  sorry

end apple_percentage_after_adding_oranges_l1406_140679


namespace sum_of_extrema_x_l1406_140628

theorem sum_of_extrema_x (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x^2 + y^2 + z^2 = 11) :
  ∃ (m M : ℝ), (∀ x', ∃ y' z', x' + y' + z' = 5 ∧ x'^2 + y'^2 + z'^2 = 11 → m ≤ x' ∧ x' ≤ M) ∧
                m + M = 8/3 :=
sorry

end sum_of_extrema_x_l1406_140628


namespace tarantulas_in_egg_sac_l1406_140674

/-- The number of legs a tarantula has -/
def tarantula_legs : ℕ := 8

/-- The total number of baby tarantula legs in the egg sacs -/
def total_legs : ℕ := 32000

/-- The number of egg sacs containing the baby tarantulas -/
def num_egg_sacs : ℕ := 4

/-- The number of tarantulas in one egg sac -/
def tarantulas_per_sac : ℕ := total_legs / (tarantula_legs * num_egg_sacs)

theorem tarantulas_in_egg_sac : tarantulas_per_sac = 1000 := by
  sorry

end tarantulas_in_egg_sac_l1406_140674


namespace interview_probability_l1406_140651

/-- The number of students enrolled in at least one foreign language class -/
def total_students : ℕ := 25

/-- The number of students in the French class -/
def french_students : ℕ := 18

/-- The number of students in the Spanish class -/
def spanish_students : ℕ := 21

/-- The number of students to be chosen -/
def chosen_students : ℕ := 2

/-- The probability of selecting at least one student from French class
    and at least one student from Spanish class -/
def probability_both_classes : ℚ := 91 / 100

theorem interview_probability :
  let students_in_both := french_students + spanish_students - total_students
  let only_french := french_students - students_in_both
  let only_spanish := spanish_students - students_in_both
  probability_both_classes = 1 - (Nat.choose only_french chosen_students + Nat.choose only_spanish chosen_students : ℚ) / Nat.choose total_students chosen_students :=
by sorry

end interview_probability_l1406_140651


namespace no_triangle_solution_l1406_140606

theorem no_triangle_solution (A B C : Real) (a b c : Real) : 
  A = Real.pi / 3 →  -- 60 degrees in radians
  b = 4 → 
  a = 2 → 
  ¬ (∃ (B C : Real), 
      0 < B ∧ 0 < C ∧ 
      A + B + C = Real.pi ∧ 
      a / Real.sin A = b / Real.sin B ∧ 
      b / Real.sin B = c / Real.sin C) :=
by
  sorry


end no_triangle_solution_l1406_140606


namespace unique_zero_of_f_l1406_140688

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a^x else Real.log x / Real.log a

-- Theorem statement
theorem unique_zero_of_f (a : ℝ) (h : a > 0) :
  ∃! x, f a x = a :=
sorry

end unique_zero_of_f_l1406_140688


namespace pauls_chickens_l1406_140683

theorem pauls_chickens (neighbor_sale quick_sale remaining : ℕ) 
  (h1 : neighbor_sale = 12)
  (h2 : quick_sale = 25)
  (h3 : remaining = 43) :
  neighbor_sale + quick_sale + remaining = 80 :=
by sorry

end pauls_chickens_l1406_140683


namespace aaron_position_100_l1406_140656

/-- Represents a position on a 2D plane -/
structure Position :=
  (x : Int) (y : Int)

/-- Represents a direction -/
inductive Direction
  | East
  | North
  | West
  | South

/-- Defines Aaron's movement rules -/
def nextPosition (current : Position) (dir : Direction) (visited : List Position) : Position × Direction :=
  sorry

/-- Calculates Aaron's position after n moves -/
def aaronPosition (n : Nat) : Position :=
  sorry

/-- Theorem stating Aaron's position after 100 moves -/
theorem aaron_position_100 : aaronPosition 100 = Position.mk 22 (-6) := by
  sorry

end aaron_position_100_l1406_140656


namespace power_sqrt_abs_calculation_l1406_140698

theorem power_sqrt_abs_calculation : 2^0 + Real.sqrt 9 - |(-4)| = 0 := by
  sorry

end power_sqrt_abs_calculation_l1406_140698


namespace shaded_area_is_five_l1406_140634

/-- Given a parallelogram with regions labeled by areas, prove that the shaded region α has area 5 -/
theorem shaded_area_is_five (x y α : ℝ) 
  (h1 : 3 + α + y = 4 + α + x)
  (h2 : 1 + x + 3 + 3 + α + y + 4 + 1 = 2 * (4 + α + x)) : 
  α = 5 := by
  sorry

end shaded_area_is_five_l1406_140634


namespace quadratic_sum_value_l1406_140664

/-- 
Given two quadratic trinomials that differ by the interchange of the constant term 
and the second coefficient, if their sum has a unique root, then the value of their 
sum at x = 2 is either 8 or 32.
-/
theorem quadratic_sum_value (p q : ℝ) : 
  let f := fun x : ℝ => x^2 + p*x + q
  let g := fun x : ℝ => x^2 + q*x + p
  let sum := fun x : ℝ => f x + g x
  (∃! r : ℝ, sum r = 0) → (sum 2 = 8 ∨ sum 2 = 32) :=
by sorry

end quadratic_sum_value_l1406_140664


namespace students_in_c_class_l1406_140602

theorem students_in_c_class (a b c : ℕ) : 
  a = 44 ∧ a + 2 = b ∧ b = c + 1 → c = 45 := by
  sorry

end students_in_c_class_l1406_140602


namespace special_isosceles_sine_l1406_140600

/-- An isosceles triangle with a special property on inscribed rectangles -/
structure SpecialIsoscelesTriangle where
  -- The vertex angle of the isosceles triangle
  vertex_angle : ℝ
  -- The base and height of the isosceles triangle
  base : ℝ
  height : ℝ
  -- The isosceles property
  isosceles : base = height
  -- The property that all inscribed rectangles with two vertices on the base have the same perimeter
  constant_perimeter : ∀ (x : ℝ), 0 ≤ x → x ≤ base → 
    2 * (x + (base * (height - x)) / height) = base + height

/-- The main theorem stating that the sine of the vertex angle is 4/5 -/
theorem special_isosceles_sine (t : SpecialIsoscelesTriangle) : 
  Real.sin t.vertex_angle = 4/5 := by
  sorry

end special_isosceles_sine_l1406_140600


namespace exists_negative_irrational_greater_than_neg_four_l1406_140676

theorem exists_negative_irrational_greater_than_neg_four :
  ∃ x : ℝ, x < 0 ∧ Irrational x ∧ -4 < x := by
sorry

end exists_negative_irrational_greater_than_neg_four_l1406_140676


namespace triangle_sin_b_l1406_140654

theorem triangle_sin_b (A B C : Real) (AC BC : Real) (h1 : AC = 2) (h2 : BC = 3) (h3 : Real.cos A = 3/5) :
  Real.sin B = 8/15 := by
  sorry

end triangle_sin_b_l1406_140654


namespace fraction_sum_division_l1406_140673

theorem fraction_sum_division (a b c d e f g h : ℚ) :
  a = 3/7 →
  b = 5/8 →
  c = 5/12 →
  d = 2/9 →
  e = a + b →
  f = c + d →
  g = e / f →
  g = 531/322 :=
by
  sorry

end fraction_sum_division_l1406_140673


namespace y_minimum_range_l1406_140612

def y (x : ℝ) : ℝ := |x^2 - 1| + |2*x^2 - 1| + |3*x^2 - 1|

theorem y_minimum_range :
  ∀ x : ℝ, y x ≥ 1 ∧
  (y x = 1 ↔ (x ∈ Set.Icc (-Real.sqrt (1/2)) (-Real.sqrt (1/3)) ∪ 
              Set.Icc (Real.sqrt (1/3)) (Real.sqrt (1/2)))) :=
sorry

end y_minimum_range_l1406_140612


namespace box_min_height_l1406_140629

/-- Represents a rectangular box with square bases -/
structure Box where
  base_side : ℝ
  height : ℝ

/-- Calculates the surface area of a box -/
def surface_area (b : Box) : ℝ :=
  2 * b.base_side^2 + 4 * b.base_side * b.height

/-- The minimum height of a box satisfying the given conditions -/
def min_height : ℝ := 6

theorem box_min_height :
  ∀ (b : Box),
    b.height = b.base_side + 4 →
    surface_area b ≥ 120 →
    b.height ≥ min_height :=
by
  sorry

end box_min_height_l1406_140629


namespace range_of_m_l1406_140644

-- Define the sets A and B
def A := {x : ℝ | -2 ≤ x ∧ x ≤ 10}
def B (m : ℝ) := {x : ℝ | 1 - m ≤ x ∧ x ≤ 1 + m}

-- State the theorem
theorem range_of_m (m : ℝ) :
  (m > 0) →
  (∀ x : ℝ, x ∈ A → x ∈ B m) →
  (∃ x : ℝ, x ∈ B m ∧ x ∉ A) →
  m ≥ 9 :=
by sorry

end range_of_m_l1406_140644


namespace min_reciprocal_sum_min_reciprocal_sum_equality_l1406_140603

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 2) :
  (1 / x + 1 / y) ≥ 3 / 2 + Real.sqrt 2 :=
by sorry

theorem min_reciprocal_sum_equality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 2) :
  (1 / x + 1 / y = 3 / 2 + Real.sqrt 2) ↔ (x = 2 / 3 ∧ y = 2 / 3) :=
by sorry

end min_reciprocal_sum_min_reciprocal_sum_equality_l1406_140603


namespace axis_of_symmetry_sinusoid_l1406_140678

open Real

theorem axis_of_symmetry_sinusoid (x : ℝ) :
  let f := fun x => Real.sin (1/2 * x - π/6)
  ∃ k : ℤ, f (4*π/3 + x) = f (4*π/3 - x) :=
by sorry

end axis_of_symmetry_sinusoid_l1406_140678


namespace office_paper_cost_l1406_140607

/-- Represents a type of bond paper -/
structure BondPaper where
  sheets_per_ream : ℕ
  cost_per_ream : ℕ

/-- Calculates the number of reams needed, rounding up -/
def reams_needed (sheets_required : ℕ) (paper : BondPaper) : ℕ :=
  (sheets_required + paper.sheets_per_ream - 1) / paper.sheets_per_ream

/-- Calculates the cost for a given number of reams -/
def cost_for_reams (reams : ℕ) (paper : BondPaper) : ℕ :=
  reams * paper.cost_per_ream

theorem office_paper_cost :
  let type_a : BondPaper := ⟨500, 27⟩
  let type_b : BondPaper := ⟨400, 24⟩
  let type_c : BondPaper := ⟨300, 18⟩
  let total_sheets : ℕ := 5000
  let min_a_sheets : ℕ := 2500
  let min_b_sheets : ℕ := 1500
  let remaining_sheets : ℕ := total_sheets - min_a_sheets - min_b_sheets
  let reams_a : ℕ := reams_needed min_a_sheets type_a
  let reams_b : ℕ := reams_needed min_b_sheets type_b
  let reams_c : ℕ := reams_needed remaining_sheets type_c
  let total_cost : ℕ := cost_for_reams reams_a type_a +
                        cost_for_reams reams_b type_b +
                        cost_for_reams reams_c type_c
  total_cost = 303 := by
  sorry

end office_paper_cost_l1406_140607


namespace turtle_ratio_l1406_140682

theorem turtle_ratio : 
  ∀ (trey kris kristen : ℕ),
  kristen = 12 →
  kris = kristen / 4 →
  trey = kristen + 9 →
  trey / kris = 7 :=
by
  sorry

end turtle_ratio_l1406_140682


namespace simplify_expression_l1406_140613

theorem simplify_expression : (2^3002 * 3^3004) / 6^3003 = 3/4 := by
  sorry

end simplify_expression_l1406_140613


namespace tan_sin_identity_l1406_140680

theorem tan_sin_identity : 2 * Real.tan (10 * π / 180) + 3 * Real.sin (10 * π / 180) = 5 * Real.sin (10 * π / 180) := by
  sorry

end tan_sin_identity_l1406_140680


namespace bodyguard_hours_theorem_l1406_140685

/-- The number of hours per day Tim hires bodyguards -/
def hours_per_day (num_bodyguards : ℕ) (hourly_rate : ℕ) (weekly_payment : ℕ) (days_per_week : ℕ) : ℕ :=
  weekly_payment / (num_bodyguards * hourly_rate * days_per_week)

/-- Theorem stating that Tim hires bodyguards for 8 hours per day -/
theorem bodyguard_hours_theorem (num_bodyguards : ℕ) (hourly_rate : ℕ) (weekly_payment : ℕ) (days_per_week : ℕ)
  (h1 : num_bodyguards = 2)
  (h2 : hourly_rate = 20)
  (h3 : weekly_payment = 2240)
  (h4 : days_per_week = 7) :
  hours_per_day num_bodyguards hourly_rate weekly_payment days_per_week = 8 := by
  sorry

end bodyguard_hours_theorem_l1406_140685


namespace two_face_cubes_5x5x5_l1406_140639

/-- The number of unit cubes with exactly two faces on the surface of a 5x5x5 cube -/
def two_face_cubes (n : ℕ) : ℕ := 12 * (n - 2)

/-- Theorem stating that the number of unit cubes with exactly two faces
    on the surface of a 5x5x5 cube is 36 -/
theorem two_face_cubes_5x5x5 :
  two_face_cubes 5 = 36 := by
  sorry

end two_face_cubes_5x5x5_l1406_140639


namespace geometric_sequence_condition_l1406_140605

/-- Given a geometric sequence {a_n} with a_1 = 1, prove that a_2 = 4 is sufficient but not necessary for a_3 = 16 -/
theorem geometric_sequence_condition (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- Geometric sequence condition
  a 1 = 1 →                            -- First term is 1
  (a 2 = 4 → a 3 = 16) ∧               -- Sufficient condition
  ¬(a 3 = 16 → a 2 = 4)                -- Not necessary condition
  := by sorry

end geometric_sequence_condition_l1406_140605


namespace symmetric_point_wrt_origin_l1406_140659

/-- Given a point P(-3, 2), its symmetric point P' with respect to the origin O has coordinates (3, -2). -/
theorem symmetric_point_wrt_origin :
  let P : ℝ × ℝ := (-3, 2)
  let P' : ℝ × ℝ := (3, -2)
  let symmetric_wrt_origin (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)
  symmetric_wrt_origin P = P' := by sorry

end symmetric_point_wrt_origin_l1406_140659
