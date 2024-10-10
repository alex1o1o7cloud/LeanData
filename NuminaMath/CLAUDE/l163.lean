import Mathlib

namespace complex_roots_properties_l163_16370

noncomputable def z₁ : ℂ := Real.sqrt 2 * Complex.exp (Complex.I * Real.pi / 4)

theorem complex_roots_properties (a b : ℝ) :
  z₁^2 + a * z₁ + b = 0 →
  ∃ z₂ : ℂ,
    z₁ = 1 + Complex.I ∧
    a = -2 ∧
    b = 2 ∧
    z₂ = 1 - Complex.I ∧
    Complex.abs (z₁ * z₂) = Complex.abs z₁ * Complex.abs z₂ ∧
    Complex.abs (z₁ * z₂) = 2 := by
  sorry

end complex_roots_properties_l163_16370


namespace games_for_512_participants_l163_16336

/-- Represents a single-elimination tournament. -/
structure SingleEliminationTournament where
  participants : ℕ
  is_power_of_two : ∃ n : ℕ, participants = 2^n

/-- Calculates the number of games required to determine a winner in a single-elimination tournament. -/
def games_required (tournament : SingleEliminationTournament) : ℕ :=
  tournament.participants - 1

/-- Theorem stating that a single-elimination tournament with 512 participants requires 511 games. -/
theorem games_for_512_participants :
  ∀ (tournament : SingleEliminationTournament),
  tournament.participants = 512 →
  games_required tournament = 511 := by
  sorry

#eval games_required ⟨512, ⟨9, rfl⟩⟩

end games_for_512_participants_l163_16336


namespace f_continuous_iff_b_eq_zero_l163_16314

-- Define the piecewise function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x > 2 then x + 4 else 3 * x + b

-- State the theorem
theorem f_continuous_iff_b_eq_zero (b : ℝ) :
  Continuous (f b) ↔ b = 0 := by
  sorry

end f_continuous_iff_b_eq_zero_l163_16314


namespace probability_second_class_correct_l163_16306

/-- The probability of selecting at least one second-class product when
    randomly choosing 4 products from a batch of 100 products containing
    90 first-class and 10 second-class products. -/
def probability_second_class (total : ℕ) (first_class : ℕ) (second_class : ℕ) (selected : ℕ) : ℚ :=
  1 - (first_class / total) * ((first_class - 1) / (total - 1)) *
      ((first_class - 2) / (total - 2)) * ((first_class - 3) / (total - 3))

/-- The theorem stating that the probability of selecting at least one
    second-class product is correct for the given conditions. -/
theorem probability_second_class_correct :
  probability_second_class 100 90 10 4 = 1 - (90/100 * 89/99 * 88/98 * 87/97) :=
by sorry

end probability_second_class_correct_l163_16306


namespace system_solution_l163_16338

theorem system_solution (p q r s t : ℝ) :
  p^2 + q^2 + r^2 = 6 ∧ p * q - s^2 - t^2 = 3 →
  ((p = Real.sqrt 3 ∧ q = Real.sqrt 3 ∧ r = 0 ∧ s = 0 ∧ t = 0) ∨
   (p = -Real.sqrt 3 ∧ q = -Real.sqrt 3 ∧ r = 0 ∧ s = 0 ∧ t = 0)) := by
  sorry

end system_solution_l163_16338


namespace algebraic_expression_value_l163_16344

theorem algebraic_expression_value (a : ℝ) (h : a = Real.sqrt 3 - 1) :
  (1 - 3 / (a + 2)) / ((a^2 - 1) / (a + 2)) = Real.sqrt 3 / 3 := by
  sorry

end algebraic_expression_value_l163_16344


namespace not_perfect_square_l163_16305

theorem not_perfect_square : 
  (∃ x : ℕ, 6^2024 = x^2) ∧ 
  (∀ y : ℕ, 7^2025 ≠ y^2) ∧ 
  (∃ z : ℕ, 8^2026 = z^2) ∧ 
  (∃ w : ℕ, 9^2027 = w^2) ∧ 
  (∃ v : ℕ, 10^2028 = v^2) := by
  sorry

end not_perfect_square_l163_16305


namespace sum_of_solutions_l163_16368

/-- The equation we want to solve -/
def equation (x : ℝ) : Prop :=
  15 / (x * (35 - 8 * x^3)^(1/3)) = 2 * x + (35 - 8 * x^3)^(1/3)

/-- The set of all solutions to the equation -/
def solution_set : Set ℝ :=
  {x : ℝ | equation x ∧ x ≠ 0 ∧ 35 - 8 * x^3 > 0}

/-- The theorem stating that the sum of all solutions is 2.5 -/
theorem sum_of_solutions :
  ∃ (s : Finset ℝ), s.toSet = solution_set ∧ s.sum id = 2.5 :=
sorry

end sum_of_solutions_l163_16368


namespace smallest_composite_no_small_factors_l163_16384

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def has_no_prime_factors_less_than_20 (n : ℕ) : Prop :=
  ∀ p, p < 20 → p.Prime → ¬(p ∣ n)

theorem smallest_composite_no_small_factors :
  (is_composite 667) ∧
  (has_no_prime_factors_less_than_20 667) ∧
  (∀ m : ℕ, m < 667 →
    ¬(is_composite m ∧ has_no_prime_factors_less_than_20 m)) :=
sorry

end smallest_composite_no_small_factors_l163_16384


namespace binary_10101_is_21_l163_16365

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_10101_is_21 :
  binary_to_decimal [true, false, true, false, true] = 21 := by
  sorry

end binary_10101_is_21_l163_16365


namespace max_product_constraint_l163_16386

theorem max_product_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4*y = 1) :
  x * y ≤ 1/16 := by
  sorry

end max_product_constraint_l163_16386


namespace gcd_lcm_a_b_l163_16320

-- Define a and b
def a : Nat := 2 * 3 * 7
def b : Nat := 2 * 3 * 3 * 5

-- State the theorem
theorem gcd_lcm_a_b : Nat.gcd a b = 6 ∧ Nat.lcm a b = 630 := by
  sorry

end gcd_lcm_a_b_l163_16320


namespace line_intercept_ratio_l163_16352

theorem line_intercept_ratio (b : ℝ) (u v : ℝ) (h : b ≠ 0) : 
  (5 * u + b = 0) → (3 * v + b = 0) → u / v = 3 / 5 := by
  sorry

end line_intercept_ratio_l163_16352


namespace min_t_value_l163_16312

theorem min_t_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 2 * a + b = 1) :
  (∀ a b, a > 0 → b > 0 → 2 * a + b = 1 → 2 * Real.sqrt (a * b) - 4 * a^2 - b^2 ≤ t - 1/2) →
  t ≥ Real.sqrt 2 / 2 :=
by sorry

end min_t_value_l163_16312


namespace equation_rewrite_l163_16316

theorem equation_rewrite (x y : ℝ) : 
  (3 * x + y = 17) → (y = -3 * x + 17) := by
  sorry

end equation_rewrite_l163_16316


namespace least_possible_value_x_l163_16301

theorem least_possible_value_x (a b x : ℕ) 
  (h1 : x = 2 * a^5)
  (h2 : x = 3 * b^2)
  (h3 : 0 < a)
  (h4 : 0 < b) :
  ∀ y : ℕ, (∃ c d : ℕ, y = 2 * c^5 ∧ y = 3 * d^2 ∧ 0 < c ∧ 0 < d) → x ≤ y ∧ x = 15552 := by
  sorry

#check least_possible_value_x

end least_possible_value_x_l163_16301


namespace rabbits_eaten_potatoes_l163_16324

/-- The number of potatoes eaten by rabbits -/
def potatoesEaten (initial remaining : ℕ) : ℕ := initial - remaining

/-- Theorem: The number of potatoes eaten by rabbits is equal to the difference
    between the initial number of potatoes and the remaining number of potatoes -/
theorem rabbits_eaten_potatoes (initial remaining : ℕ) (h : remaining ≤ initial) :
  potatoesEaten initial remaining = initial - remaining := by
  sorry

#eval potatoesEaten 8 5  -- Should evaluate to 3

end rabbits_eaten_potatoes_l163_16324


namespace min_value_expression_l163_16364

theorem min_value_expression (x y z t : ℝ) 
  (h1 : x + 4*y = 4) 
  (h2 : y > 0) 
  (h3 : 0 < t) 
  (h4 : t < z) : 
  (4*z^2 / abs x) + (abs (x*z^2) / y) + (12 / (t*(z-t))) ≥ 24 := by
  sorry

end min_value_expression_l163_16364


namespace zoo_visitors_l163_16379

theorem zoo_visitors (monday_children monday_adults tuesday_children : ℕ)
  (child_ticket_price adult_ticket_price : ℕ)
  (total_revenue : ℕ) :
  monday_children = 7 →
  monday_adults = 5 →
  tuesday_children = 4 →
  child_ticket_price = 3 →
  adult_ticket_price = 4 →
  total_revenue = 61 →
  ∃ tuesday_adults : ℕ,
    total_revenue =
      monday_children * child_ticket_price +
      monday_adults * adult_ticket_price +
      tuesday_children * child_ticket_price +
      tuesday_adults * adult_ticket_price ∧
    tuesday_adults = 2 :=
by sorry

end zoo_visitors_l163_16379


namespace x_eq_3_sufficient_not_necessary_l163_16359

-- Define vectors a and b
def a (x : ℝ) : ℝ × ℝ := (2, x - 1)
def b (x : ℝ) : ℝ × ℝ := (x + 1, 4)

-- Define parallel condition for 2D vectors
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

-- Theorem statement
theorem x_eq_3_sufficient_not_necessary :
  (∀ x : ℝ, x = 3 → parallel (a x) (b x)) ∧
  ¬(∀ x : ℝ, parallel (a x) (b x) → x = 3) :=
sorry

end x_eq_3_sufficient_not_necessary_l163_16359


namespace cylinder_volume_l163_16337

/-- The volume of a cylinder with height 300 cm and circular base area of 9 square cm is 2700 cubic centimeters. -/
theorem cylinder_volume (h : ℝ) (base_area : ℝ) (volume : ℝ) 
  (h_val : h = 300)
  (base_area_val : base_area = 9)
  (volume_def : volume = base_area * h) : 
  volume = 2700 := by
  sorry

end cylinder_volume_l163_16337


namespace plot_length_is_75_l163_16362

/-- Proves that the length of a rectangular plot is 75 meters given the specified conditions -/
theorem plot_length_is_75 (breadth : ℝ) (length : ℝ) (perimeter : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) :
  length = breadth + 50 →
  perimeter = 2 * length + 2 * breadth →
  cost_per_meter = 26.50 →
  total_cost = 5300 →
  total_cost = perimeter * cost_per_meter →
  length = 75 := by
sorry

end plot_length_is_75_l163_16362


namespace chocolates_remaining_on_day_five_l163_16329

/-- Chocolates eaten per day --/
def chocolates_eaten (day : Nat) : Nat :=
  match day with
  | 1 => 4
  | 2 => 2 * 4 - 3
  | 3 => 4 - 2
  | 4 => (4 - 2) - 1
  | _ => 0

/-- Total chocolates eaten up to a given day --/
def total_eaten (day : Nat) : Nat :=
  match day with
  | 0 => 0
  | n + 1 => total_eaten n + chocolates_eaten (n + 1)

theorem chocolates_remaining_on_day_five : 
  24 - total_eaten 4 = 12 := by
  sorry

end chocolates_remaining_on_day_five_l163_16329


namespace apple_distribution_l163_16353

theorem apple_distribution (total_apples : ℕ) (given_to_father : ℕ) (num_friends : ℕ) :
  total_apples = 55 →
  given_to_father = 10 →
  num_friends = 4 →
  (total_apples - given_to_father) % (num_friends + 1) = 0 →
  (total_apples - given_to_father) / (num_friends + 1) = 9 :=
by sorry

end apple_distribution_l163_16353


namespace building_heights_sum_l163_16354

/-- Proves that the total height of three buildings is 340 feet -/
theorem building_heights_sum (middle_height : ℝ) (left_height : ℝ) (right_height : ℝ)
  (h1 : middle_height = 100)
  (h2 : left_height = 0.8 * middle_height)
  (h3 : right_height = left_height + middle_height - 20) :
  left_height + middle_height + right_height = 340 := by
  sorry

end building_heights_sum_l163_16354


namespace function_growth_l163_16373

/-- For any differentiable function f: ℝ → ℝ, if f'(x) > f(x) for all x ∈ ℝ,
    then f(a) > e^a * f(0) for any a > 0. -/
theorem function_growth (f : ℝ → ℝ) (hf : Differentiable ℝ f)
    (h : ∀ x, deriv f x > f x) (a : ℝ) (ha : a > 0) :
  f a > Real.exp a * f 0 := by
  sorry

end function_growth_l163_16373


namespace C_power_50_l163_16394

def C : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; -4, -1]

theorem C_power_50 : C^50 = !![101, 50; -200, -99] := by sorry

end C_power_50_l163_16394


namespace twenty_times_nineteen_plus_twenty_plus_nineteen_l163_16351

theorem twenty_times_nineteen_plus_twenty_plus_nineteen : 20 * 19 + 20 + 19 = 419 := by
  sorry

end twenty_times_nineteen_plus_twenty_plus_nineteen_l163_16351


namespace initial_wings_count_l163_16327

/-- The number of initially cooked chicken wings -/
def initial_wings : ℕ := sorry

/-- The number of friends -/
def num_friends : ℕ := 3

/-- The number of additional wings cooked -/
def additional_wings : ℕ := 10

/-- The number of wings each person got -/
def wings_per_person : ℕ := 6

/-- Theorem stating that the number of initially cooked wings is 8 -/
theorem initial_wings_count : initial_wings = 8 := by sorry

end initial_wings_count_l163_16327


namespace simplify_expression_l163_16308

-- Define the left-hand side of the equation
def lhs (y : ℝ) : ℝ := 3*y + 4*y^2 + 2 - (8 - 3*y - 4*y^2 + y^3)

-- Define the right-hand side of the equation
def rhs (y : ℝ) : ℝ := -y^3 + 8*y^2 + 6*y - 6

-- Theorem statement
theorem simplify_expression (y : ℝ) : lhs y = rhs y := by
  sorry

end simplify_expression_l163_16308


namespace digit_sequence_equality_l163_16363

def A (n : ℕ) : ℕ := (10^n - 1) / 9

theorem digit_sequence_equality (n : ℕ) (hn : n > 0) :
  Real.sqrt ((10^n + 1) * A n - 2 * A n) = 3 * A n :=
sorry

end digit_sequence_equality_l163_16363


namespace book_sale_loss_percentage_l163_16395

theorem book_sale_loss_percentage 
  (total_cost : ℝ) 
  (cost_book1 : ℝ) 
  (gain_percentage : ℝ) :
  total_cost = 300 →
  cost_book1 = 175 →
  gain_percentage = 19 →
  let cost_book2 := total_cost - cost_book1
  let selling_price := cost_book2 * (1 + gain_percentage / 100)
  let loss_amount := cost_book1 - selling_price
  let loss_percentage := (loss_amount / cost_book1) * 100
  loss_percentage = 15 := by sorry

end book_sale_loss_percentage_l163_16395


namespace range_of_a_l163_16317

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x + a * x + a = 0) → a ∈ Set.Iic 0 ∪ Set.Ici 4 := by
  sorry

end range_of_a_l163_16317


namespace article_price_l163_16303

theorem article_price (profit_percentage : ℝ) (profit_amount : ℝ) (original_price : ℝ) : 
  profit_percentage = 40 →
  profit_amount = 560 →
  original_price * (1 + profit_percentage / 100) - original_price = profit_amount →
  original_price = 1400 := by
sorry

end article_price_l163_16303


namespace gas_experiment_values_l163_16361

/-- Represents the state of a gas -/
structure GasState where
  volume : ℝ
  pressure : ℝ
  temperature : ℝ

/-- Represents the change in gas state -/
structure GasStateChange where
  Δp : ℝ
  ΔV : ℝ

/-- Theorem stating the values of a₁ and a₂ for the given gas experiments -/
theorem gas_experiment_values (initialState : GasState) 
  (h_volume : initialState.volume = 1)
  (h_pressure : initialState.pressure = 10^5)
  (h_temperature : initialState.temperature = 300)
  (experiment1 : GasStateChange → Bool)
  (experiment2 : GasStateChange → Bool)
  (h_exp1 : ∀ change, experiment1 change ↔ change.Δp / change.ΔV = -10^5)
  (h_exp2 : ∀ change, experiment2 change ↔ change.Δp / change.ΔV = -1.4 * 10^5)
  (h_cooling1 : ∀ change, experiment1 change → 
    (change.ΔV > 0 → initialState.temperature > initialState.temperature + change.ΔV) ∧
    (change.ΔV < 0 → initialState.temperature > initialState.temperature - change.ΔV))
  (h_heating2 : ∀ change, experiment2 change → 
    (change.ΔV > 0 → initialState.temperature < initialState.temperature + change.ΔV) ∧
    (change.ΔV < 0 → initialState.temperature < initialState.temperature - change.ΔV)) :
  ∃ (a₁ a₂ : ℝ), a₁ = -10^5 ∧ a₂ = -1.4 * 10^5 := by
  sorry


end gas_experiment_values_l163_16361


namespace vector_equation_solution_l163_16343

theorem vector_equation_solution :
  let a : ℚ := 23/7
  let b : ℚ := -1/7
  let v1 : Fin 2 → ℚ := ![1, 4]
  let v2 : Fin 2 → ℚ := ![3, -2]
  let result : Fin 2 → ℚ := ![2, 10]
  (a • v1 + b • v2 = result) := by sorry

end vector_equation_solution_l163_16343


namespace cos_75_cos_15_minus_sin_75_sin_15_l163_16397

theorem cos_75_cos_15_minus_sin_75_sin_15 :
  Real.cos (75 * π / 180) * Real.cos (15 * π / 180) -
  Real.sin (75 * π / 180) * Real.sin (15 * π / 180) = 0 := by
  sorry

end cos_75_cos_15_minus_sin_75_sin_15_l163_16397


namespace complex_square_magnitude_l163_16356

theorem complex_square_magnitude (z : ℂ) (h : z^2 + Complex.abs z^2 = 8 - 3*I) : 
  Complex.abs z^2 = 73/16 := by sorry

end complex_square_magnitude_l163_16356


namespace plumber_max_shower_charge_l163_16350

def plumber_problem (sink_charge toilet_charge shower_charge : ℕ) : Prop :=
  let job1 := 3 * toilet_charge + 3 * sink_charge
  let job2 := 2 * toilet_charge + 5 * sink_charge
  let job3 := toilet_charge + 2 * shower_charge + 3 * sink_charge
  sink_charge = 30 ∧
  toilet_charge = 50 ∧
  (job1 ≤ 250 ∧ job2 ≤ 250 ∧ job3 ≤ 250) ∧
  (job1 = 250 ∨ job2 = 250 ∨ job3 = 250) →
  shower_charge ≤ 55

theorem plumber_max_shower_charge :
  ∃ (shower_charge : ℕ), plumber_problem 30 50 shower_charge ∧
  ∀ (x : ℕ), x > shower_charge → ¬ plumber_problem 30 50 x :=
sorry

end plumber_max_shower_charge_l163_16350


namespace antonella_toonies_l163_16391

/-- Represents the number of coins of each type -/
structure CoinCount where
  loonies : ℕ
  toonies : ℕ

/-- Calculates the total value of coins in dollars -/
def totalValue (coins : CoinCount) : ℕ :=
  coins.loonies + 2 * coins.toonies

/-- Represents Antonella's coin situation -/
def antonellasCoins (coins : CoinCount) : Prop :=
  coins.loonies + coins.toonies = 10 ∧
  totalValue coins = 14

theorem antonella_toonies :
  ∃ (coins : CoinCount), antonellasCoins coins ∧ coins.toonies = 4 := by
  sorry

end antonella_toonies_l163_16391


namespace pencil_difference_l163_16318

/-- The number of pencils Paige has in her desk -/
def pencils_in_desk : ℕ := 2

/-- The number of pencils Paige has in her backpack -/
def pencils_in_backpack : ℕ := 2

/-- The number of pencils Paige has at home -/
def pencils_at_home : ℕ := 15

/-- The difference between the number of pencils at Paige's home and in Paige's backpack -/
theorem pencil_difference : pencils_at_home - pencils_in_backpack = 13 := by
  sorry

end pencil_difference_l163_16318


namespace complex_number_equality_l163_16348

theorem complex_number_equality (b : ℝ) : 
  (Complex.re ((1 + b * Complex.I) / (1 - Complex.I)) = 
   Complex.im ((1 + b * Complex.I) / (1 - Complex.I))) → b = 0 := by
sorry

end complex_number_equality_l163_16348


namespace gcd_8008_11011_l163_16322

theorem gcd_8008_11011 : Nat.gcd 8008 11011 = 1001 := by sorry

end gcd_8008_11011_l163_16322


namespace product_reciprocal_sum_l163_16340

theorem product_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_product : a * b = 16) (h_reciprocal : 1 / a = 3 * (1 / b)) : 
  a + b = 16 * Real.sqrt 3 / 3 := by
sorry

end product_reciprocal_sum_l163_16340


namespace two_std_dev_below_mean_l163_16383

/-- For a normal distribution with mean 10.5 and standard deviation 1,
    the value that is exactly 2 standard deviations less than the mean is 8.5. -/
theorem two_std_dev_below_mean (μ σ : ℝ) (hμ : μ = 10.5) (hσ : σ = 1) :
  μ - 2 * σ = 8.5 := by
  sorry

end two_std_dev_below_mean_l163_16383


namespace circle_area_l163_16358

theorem circle_area (r : ℝ) (h : 3 / (2 * Real.pi * r) = r) : r ^ 2 * Real.pi = 3 / 2 := by
  sorry

end circle_area_l163_16358


namespace fraction_value_l163_16307

theorem fraction_value : (2024 - 1935)^2 / 225 = 35 := by
  sorry

end fraction_value_l163_16307


namespace units_digit_100_factorial_l163_16371

theorem units_digit_100_factorial (n : ℕ) : n = 100 → n.factorial % 10 = 0 := by
  sorry

end units_digit_100_factorial_l163_16371


namespace train_meetings_l163_16333

-- Define the travel time in minutes
def travel_time : ℕ := 210

-- Define the departure interval in minutes
def departure_interval : ℕ := 60

-- Define the time difference between the 9:00 AM train and the first train in minutes
def time_difference : ℕ := 180

-- Define a function to calculate the number of meetings
def number_of_meetings (travel_time departure_interval time_difference : ℕ) : ℕ :=
  -- The actual calculation would go here, but we're using sorry as per instructions
  sorry

-- Theorem statement
theorem train_meetings :
  number_of_meetings travel_time departure_interval time_difference = 7 := by
  sorry

end train_meetings_l163_16333


namespace medicine_weight_l163_16375

/-- Represents the weight system used for measurement -/
inductive WeightSystem
  | Ancient
  | Modern

/-- Represents a weight measurement -/
structure Weight where
  jin : ℕ
  liang : ℕ
  system : WeightSystem

/-- Converts a Weight to grams -/
def Weight.toGrams (w : Weight) : ℕ :=
  match w.system with
  | WeightSystem.Ancient => w.jin * 600 + w.liang * (600 / 16)
  | WeightSystem.Modern => w.jin * 500 + w.liang * (500 / 10)

/-- The theorem to be proved -/
theorem medicine_weight (w₁ w₂ : Weight) 
  (h₁ : w₁.system = WeightSystem.Ancient)
  (h₂ : w₂.system = WeightSystem.Modern)
  (h₃ : w₁.jin + w₂.jin = 5)
  (h₄ : w₁.liang + w₂.liang = 68)
  (h₅ : w₁.jin * 16 + w₁.liang = w₁.liang)
  (h₆ : w₂.jin * 10 + w₂.liang = w₂.liang) :
  w₁.toGrams + w₂.toGrams = 2800 := by
  sorry


end medicine_weight_l163_16375


namespace least_n_for_fraction_inequality_l163_16323

theorem least_n_for_fraction_inequality : 
  ∃ (n : ℕ), n > 0 ∧ (∀ k : ℕ, k > 0 → k < n → (1 : ℚ) / k - (1 : ℚ) / (k + 1) ≥ 1 / 15) ∧
  ((1 : ℚ) / n - (1 : ℚ) / (n + 1) < 1 / 15) ∧ n = 4 :=
by
  sorry

end least_n_for_fraction_inequality_l163_16323


namespace certain_number_exists_l163_16377

theorem certain_number_exists : ∃ x : ℝ, 
  3500 - (1000 / x) = 3451.2195121951218 ∧ 
  abs (x - 20.5) < 0.0000000000001 := by
  sorry

end certain_number_exists_l163_16377


namespace weight_lowering_feel_l163_16346

theorem weight_lowering_feel (num_plates : ℕ) (weight_per_plate : ℝ) (increase_percentage : ℝ) :
  num_plates = 10 →
  weight_per_plate = 30 →
  increase_percentage = 0.2 →
  (num_plates : ℝ) * weight_per_plate * (1 + increase_percentage) = 360 := by
  sorry

end weight_lowering_feel_l163_16346


namespace count_fours_to_1000_l163_16385

/-- Count of digit 4 in a single number -/
def count_fours (n : ℕ) : ℕ := sorry

/-- Sum of count_fours for all numbers from 1 to n -/
def total_fours (n : ℕ) : ℕ := sorry

/-- The count of the digit 4 appearing in the integers from 1 to 1000 is equal to 300 -/
theorem count_fours_to_1000 : total_fours 1000 = 300 := by sorry

end count_fours_to_1000_l163_16385


namespace beth_candies_theorem_l163_16390

def total_candies : ℕ := 10

def is_valid_distribution (a b c : ℕ) : Prop :=
  a + b + c = total_candies ∧ a ≥ 3 ∧ b ≥ 2 ∧ c ≥ 2 ∧ c ≤ 3

def possible_beth_candies : Set ℕ := {2, 3, 4, 5}

theorem beth_candies_theorem :
  ∀ b : ℕ, (∃ a c : ℕ, is_valid_distribution a b c) ↔ b ∈ possible_beth_candies :=
sorry

end beth_candies_theorem_l163_16390


namespace halloween_cleaning_time_l163_16334

/-- Calculates the total cleaning time for Halloween pranks -/
theorem halloween_cleaning_time 
  (egg_cleaning_time : ℕ) 
  (tp_cleaning_time : ℕ) 
  (num_eggs : ℕ) 
  (num_tp : ℕ) : 
  egg_cleaning_time = 15 ∧ 
  tp_cleaning_time = 30 ∧ 
  num_eggs = 60 ∧ 
  num_tp = 7 → 
  (num_eggs * egg_cleaning_time) / 60 + num_tp * tp_cleaning_time = 225 := by
  sorry

#check halloween_cleaning_time

end halloween_cleaning_time_l163_16334


namespace power_equality_l163_16335

theorem power_equality (J : ℕ) (h : (32^4) * (4^4) = 2^J) : J = 28 := by
  sorry

end power_equality_l163_16335


namespace final_value_of_A_l163_16372

theorem final_value_of_A : ∀ A : ℤ, A = 15 → (A = -15 + 5) → A = -10 := by
  sorry

end final_value_of_A_l163_16372


namespace max_lateral_surface_area_rectangular_prism_l163_16311

theorem max_lateral_surface_area_rectangular_prism :
  ∀ l w h : ℕ,
  l + w + h = 88 →
  2 * (l * w + l * h + w * h) ≤ 224 :=
by sorry

end max_lateral_surface_area_rectangular_prism_l163_16311


namespace inconsistent_inventory_report_max_consistent_statements_no_more_than_three_consistent_l163_16330

theorem inconsistent_inventory_report (n : ℕ) (h_n : n ≥ 1000) : 
  ¬(n % 2 = 1 ∧ n % 3 = 1 ∧ n % 4 = 2 ∧ n % 5 = 2) :=
sorry

theorem max_consistent_statements : 
  ∃ (n : ℕ), n ≥ 1000 ∧ 
  ((n % 2 = 1 ∧ n % 3 = 1 ∧ n % 5 = 2) ∨
   (n % 3 = 1 ∧ n % 4 = 2 ∧ n % 5 = 2)) :=
sorry

theorem no_more_than_three_consistent (n : ℕ) (h_n : n ≥ 1000) :
  ¬∃ (a b c d : Bool), a ∧ b ∧ c ∧ d ∧
  (a → n % 2 = 1) ∧
  (b → n % 3 = 1) ∧
  (c → n % 4 = 2) ∧
  (d → n % 5 = 2) ∧
  (a.toNat + b.toNat + c.toNat + d.toNat > 3) :=
sorry

end inconsistent_inventory_report_max_consistent_statements_no_more_than_three_consistent_l163_16330


namespace tangent_inclination_range_implies_x_coordinate_range_l163_16339

/-- The curve C defined by y = x^2 + 2x + 3 -/
def C (x : ℝ) : ℝ := x^2 + 2*x + 3

/-- The derivative of C -/
def C' (x : ℝ) : ℝ := 2*x + 2

theorem tangent_inclination_range_implies_x_coordinate_range :
  ∀ x : ℝ,
  (∃ y : ℝ, y = C x) →
  (π/4 ≤ Real.arctan (C' x) ∧ Real.arctan (C' x) ≤ π/2) →
  x ≥ -1/2 := by
  sorry

end tangent_inclination_range_implies_x_coordinate_range_l163_16339


namespace watch_cost_price_l163_16302

theorem watch_cost_price (loss_price gain_price cost_price : ℝ) 
  (h1 : loss_price = 0.88 * cost_price)
  (h2 : gain_price = 1.08 * cost_price)
  (h3 : gain_price - loss_price = 350) : 
  cost_price = 1750 := by
sorry

end watch_cost_price_l163_16302


namespace power_of_power_of_two_l163_16345

theorem power_of_power_of_two :
  let a : ℕ := 2
  a^(a^2) = 16 := by
  sorry

end power_of_power_of_two_l163_16345


namespace regular_polygon_150_degrees_has_12_sides_l163_16369

/-- A regular polygon with interior angles measuring 150° has 12 sides. -/
theorem regular_polygon_150_degrees_has_12_sides :
  ∀ n : ℕ, 
  n ≥ 3 → 
  (180 * (n - 2) : ℝ) = 150 * n → 
  n = 12 := by
sorry

end regular_polygon_150_degrees_has_12_sides_l163_16369


namespace circle_equation_l163_16341

theorem circle_equation (h : ℝ) :
  (∃ (x : ℝ), (x - 2)^2 + (-3)^2 = 5^2 ∧ h = x) →
  ((h - 6)^2 + y^2 = 25 ∨ (h + 2)^2 + y^2 = 25) :=
by sorry

end circle_equation_l163_16341


namespace symmetrical_shape_three_equal_parts_l163_16319

/-- A symmetrical 2D shape -/
structure SymmetricalShape where
  area : ℝ
  height : ℝ
  width : ℝ
  is_symmetrical : Bool

/-- A straight cut on the shape -/
inductive Cut
  | Vertical : ℝ → Cut  -- position along width
  | Horizontal : ℝ → Cut  -- position along height

/-- Result of applying cuts to a shape -/
def apply_cuts (shape : SymmetricalShape) (cuts : List Cut) : List ℝ :=
  sorry

theorem symmetrical_shape_three_equal_parts (shape : SymmetricalShape) :
  shape.is_symmetrical →
  ∃ (vertical_cut : Cut) (horizontal_cut : Cut),
    vertical_cut = Cut.Vertical (shape.width / 2) ∧
    horizontal_cut = Cut.Horizontal (shape.height / 3) ∧
    apply_cuts shape [vertical_cut, horizontal_cut] = [shape.area / 3, shape.area / 3, shape.area / 3] :=
  sorry

end symmetrical_shape_three_equal_parts_l163_16319


namespace complex_power_magnitude_l163_16378

theorem complex_power_magnitude : Complex.abs ((1 - Complex.I * 2) ^ 8) = 625 := by
  sorry

end complex_power_magnitude_l163_16378


namespace expand_and_simplify_l163_16304

theorem expand_and_simplify (x : ℝ) : (2*x + 6)*(x + 10) = 2*x^2 + 26*x + 60 := by
  sorry

end expand_and_simplify_l163_16304


namespace rick_cards_count_l163_16374

theorem rick_cards_count : ℕ := by
  -- Define the number of cards Rick kept
  let cards_kept : ℕ := 15

  -- Define the number of cards given to Miguel
  let cards_to_miguel : ℕ := 13

  -- Define the number of friends and cards given to each friend
  let num_friends : ℕ := 8
  let cards_per_friend : ℕ := 12

  -- Define the number of sisters and cards given to each sister
  let num_sisters : ℕ := 2
  let cards_per_sister : ℕ := 3

  -- Calculate the total number of cards
  let total_cards : ℕ := 
    cards_kept + 
    cards_to_miguel + 
    (num_friends * cards_per_friend) + 
    (num_sisters * cards_per_sister)

  -- Prove that the total number of cards is 130
  have h : total_cards = 130 := by sorry

  -- Return the result
  exact 130

end rick_cards_count_l163_16374


namespace total_chimpanzees_l163_16399

/- Define the number of chimps moving to the new cage -/
def chimps_new_cage : ℕ := 18

/- Define the number of chimps staying in the old cage -/
def chimps_old_cage : ℕ := 27

/- Theorem stating that the total number of chimpanzees is 45 -/
theorem total_chimpanzees : chimps_new_cage + chimps_old_cage = 45 := by
  sorry

end total_chimpanzees_l163_16399


namespace snack_eaters_problem_l163_16349

/-- The number of new outsiders who joined for snacks after the first group left -/
def new_outsiders : ℕ := sorry

theorem snack_eaters_problem (initial_people : ℕ) (initial_snackers : ℕ) (first_outsiders : ℕ) 
  (more_left : ℕ) (final_snackers : ℕ) :
  initial_people = 200 →
  initial_snackers = 100 →
  first_outsiders = 20 →
  more_left = 30 →
  final_snackers = 20 →
  new_outsiders = 40 := by sorry

end snack_eaters_problem_l163_16349


namespace complement_implies_a_value_l163_16328

def U (a : ℝ) : Set ℝ := {2, 4, 3 - a^2}
def P (a : ℝ) : Set ℝ := {2, a^2 + 2 - a}

theorem complement_implies_a_value (a : ℝ) : 
  (U a \ P a = {-1}) → a = 2 :=
by sorry

end complement_implies_a_value_l163_16328


namespace sqrt_relationship_l163_16367

theorem sqrt_relationship (h1 : Real.sqrt 23.6 = 4.858) (h2 : Real.sqrt 2.36 = 1.536) :
  Real.sqrt 0.00236 = 0.04858 := by
  sorry

end sqrt_relationship_l163_16367


namespace complex_power_sum_l163_16332

theorem complex_power_sum (z : ℂ) (h : z = (1 - I) / Real.sqrt 2) : 
  z^100 + z^50 + 1 = -I :=
by sorry

end complex_power_sum_l163_16332


namespace total_pencils_l163_16309

/-- Given an initial number of pencils in a drawer and a number of pencils added,
    the total number of pencils is equal to the sum of the initial number and the added number. -/
theorem total_pencils (initial : ℕ) (added : ℕ) : 
  initial + added = initial + added :=
by sorry

end total_pencils_l163_16309


namespace warehouse_inventory_l163_16342

theorem warehouse_inventory (x y : ℝ) : 
  x + y = 92 ∧ 
  (2/5) * x + (1/4) * y = 26 → 
  x = 20 ∧ y = 72 := by
sorry

end warehouse_inventory_l163_16342


namespace tan_sin_intersection_count_l163_16347

open Real

theorem tan_sin_intersection_count :
  let f : ℝ → ℝ := λ x => tan x - sin x
  ∃! (s : Finset ℝ), s.card = 5 ∧ (∀ x ∈ s, -2*π ≤ x ∧ x ≤ 2*π ∧ f x = 0) ∧
    (∀ x, -2*π ≤ x ∧ x ≤ 2*π ∧ f x = 0 → x ∈ s) :=
by
  sorry

end tan_sin_intersection_count_l163_16347


namespace square_areas_l163_16357

theorem square_areas (a b : ℝ) (h1 : 4*a - 4*b = 12) (h2 : a^2 - b^2 = 69) :
  (a^2 = 169 ∧ b^2 = 100) :=
sorry

end square_areas_l163_16357


namespace complex_power_sum_l163_16376

theorem complex_power_sum (z : ℂ) (h : z = -(1 - Complex.I) / Real.sqrt 2) : 
  z^100 + z^50 + 1 = -Complex.I := by sorry

end complex_power_sum_l163_16376


namespace square_area_decrease_l163_16325

theorem square_area_decrease (areaI areaIII areaII : ℝ) (decrease_percent : ℝ) :
  areaI = 18 * Real.sqrt 3 →
  areaIII = 50 * Real.sqrt 3 →
  areaII = 72 →
  decrease_percent = 20 →
  let side_length := Real.sqrt areaII
  let new_side_length := side_length * (1 - decrease_percent / 100)
  let new_area := new_side_length ^ 2
  (areaII - new_area) / areaII * 100 = 36 := by
  sorry

end square_area_decrease_l163_16325


namespace second_invoice_not_23_l163_16355

def systematic_sampling (first : ℕ) : ℕ → ℕ := fun n => first + 10 * (n - 1)

theorem second_invoice_not_23 :
  ∀ first : ℕ, 1 ≤ first ∧ first ≤ 10 →
  systematic_sampling first 2 ≠ 23 := by
sorry

end second_invoice_not_23_l163_16355


namespace school_route_time_difference_l163_16300

theorem school_route_time_difference :
  let first_route_uphill_time : ℕ := 6
  let first_route_path_time : ℕ := 2 * first_route_uphill_time
  let first_route_first_two_stages : ℕ := first_route_uphill_time + first_route_path_time
  let first_route_final_time : ℕ := first_route_first_two_stages / 3
  let first_route_total_time : ℕ := first_route_first_two_stages + first_route_final_time

  let second_route_flat_time : ℕ := 14
  let second_route_final_time : ℕ := 2 * second_route_flat_time
  let second_route_total_time : ℕ := second_route_flat_time + second_route_final_time

  second_route_total_time - first_route_total_time = 18 :=
by
  sorry

end school_route_time_difference_l163_16300


namespace square_root_three_expansion_l163_16392

theorem square_root_three_expansion 
  (a b c d : ℕ+) 
  (h : (a : ℝ) + (b : ℝ) * Real.sqrt 3 = ((c : ℝ) + (d : ℝ) * Real.sqrt 3) ^ 2) : 
  (a : ℝ) = (c : ℝ) ^ 2 + 3 * (d : ℝ) ^ 2 ∧ (b : ℝ) = 2 * (c : ℝ) * (d : ℝ) := by
  sorry

end square_root_three_expansion_l163_16392


namespace employee_reduction_l163_16396

theorem employee_reduction (original : ℕ) : 
  let after_first := (9 : ℚ) / 10 * original
  let after_second := (19 : ℚ) / 20 * after_first
  let after_third := (22 : ℚ) / 25 * after_second
  after_third = 195 → original = 259 := by
sorry

end employee_reduction_l163_16396


namespace min_value_expression_l163_16389

theorem min_value_expression (a b : ℝ) (h1 : a > b) (h2 : a * b = 2) :
  (a^2 + b^2 + 1) / (a - b) ≥ 2 * Real.sqrt 5 := by
  sorry

end min_value_expression_l163_16389


namespace total_production_l163_16381

/-- The daily production of fertilizer in tons -/
def daily_production : ℕ := 105

/-- The number of days of production -/
def days : ℕ := 24

/-- Theorem stating the total production over the given number of days -/
theorem total_production : daily_production * days = 2520 := by
  sorry

end total_production_l163_16381


namespace cycling_equation_correct_l163_16387

/-- Represents the cycling speeds and time difference between two cyclists A and B. -/
structure CyclingProblem where
  distance : ℝ  -- Distance between points A and B
  speed_diff : ℝ  -- Speed difference between A and B
  time_diff : ℝ  -- Time difference of arrival (in hours)

/-- Checks if the given equation correctly represents the cycling problem. -/
def is_correct_equation (prob : CyclingProblem) (x : ℝ) : Prop :=
  prob.distance / x - prob.distance / (x + prob.speed_diff) = prob.time_diff

/-- The main theorem stating that the given equation correctly represents the cycling problem. -/
theorem cycling_equation_correct : 
  let prob : CyclingProblem := { distance := 30, speed_diff := 3, time_diff := 2/3 }
  ∀ x > 0, is_correct_equation prob x := by
  sorry

end cycling_equation_correct_l163_16387


namespace unique_gcd_triplet_l163_16360

-- Define the sets of possible values for x, y, and z
def X : Set ℕ := {6, 8, 12, 18, 24}
def Y : Set ℕ := {14, 20, 28, 44, 56}
def Z : Set ℕ := {5, 15, 18, 27, 42}

-- Define the theorem
theorem unique_gcd_triplet :
  ∃! (a b c x y z : ℕ),
    x ∈ X ∧ y ∈ Y ∧ z ∈ Z ∧
    x = Nat.gcd a b ∧
    y = Nat.gcd b c ∧
    z = Nat.gcd c a ∧
    x = 8 ∧ y = 14 ∧ z = 18 :=
by
  sorry

#check unique_gcd_triplet

end unique_gcd_triplet_l163_16360


namespace derivative_of_complex_function_l163_16321

/-- The derivative of ln(4x - 1 + √(16x^2 - 8x + 2)) - √(16x^2 - 8x + 2) * arctan(4x - 1) -/
theorem derivative_of_complex_function (x : ℝ) 
  (h1 : 16 * x^2 - 8 * x + 2 ≥ 0) 
  (h2 : 4 * x - 1 + Real.sqrt (16 * x^2 - 8 * x + 2) > 0) :
  deriv (fun x => Real.log (4 * x - 1 + Real.sqrt (16 * x^2 - 8 * x + 2)) - 
    Real.sqrt (16 * x^2 - 8 * x + 2) * Real.arctan (4 * x - 1)) x = 
  (4 * (1 - 4 * x) / Real.sqrt (16 * x^2 - 8 * x + 2)) * Real.arctan (4 * x - 1) := by
  sorry

end derivative_of_complex_function_l163_16321


namespace fraction_comparison_l163_16398

theorem fraction_comparison (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a / b > (a + 1) / (b + 1) := by
sorry

end fraction_comparison_l163_16398


namespace hyperbola_through_point_l163_16313

/-- A hyperbola with its axes of symmetry along the coordinate axes -/
structure CoordinateAxisHyperbola where
  a : ℝ
  equation : ℝ × ℝ → Prop := fun (x, y) ↦ x^2 / a^2 - y^2 / a^2 = 1

/-- The hyperbola passes through the point (3, -1) -/
def passes_through (h : CoordinateAxisHyperbola) : Prop :=
  h.equation (3, -1)

theorem hyperbola_through_point :
  ∃ (h : CoordinateAxisHyperbola), passes_through h ∧ h.a^2 = 8 :=
sorry

end hyperbola_through_point_l163_16313


namespace fruit_punch_total_l163_16315

theorem fruit_punch_total (orange_punch : ℝ) (cherry_punch : ℝ) (apple_juice : ℝ) : 
  orange_punch = 4.5 →
  cherry_punch = 2 * orange_punch →
  apple_juice = cherry_punch - 1.5 →
  orange_punch + cherry_punch + apple_juice = 21 := by
  sorry

end fruit_punch_total_l163_16315


namespace complementary_angle_supplement_l163_16331

theorem complementary_angle_supplement (A B : Real) : 
  (A + B = 90) → (180 - A = 90 + B) := by
  sorry

end complementary_angle_supplement_l163_16331


namespace five_students_three_events_outcomes_l163_16366

/-- The number of different possible outcomes for champions in a sports competition. -/
def championOutcomes (numStudents : ℕ) (numEvents : ℕ) : ℕ :=
  numStudents ^ numEvents

/-- Theorem stating that with 5 students and 3 events, there are 125 possible outcomes. -/
theorem five_students_three_events_outcomes :
  championOutcomes 5 3 = 125 := by
  sorry

end five_students_three_events_outcomes_l163_16366


namespace student_average_age_l163_16310

theorem student_average_age
  (n : ℕ) -- number of students
  (teacher_age : ℕ) -- age of the teacher
  (avg_increase : ℝ) -- increase in average when teacher is included
  (h1 : n = 25) -- there are 25 students
  (h2 : teacher_age = 52) -- teacher's age is 52
  (h3 : avg_increase = 1) -- average increases by 1 when teacher is included
  : (n : ℝ) * ((n + 1 : ℝ) * (x + avg_increase) - teacher_age) / n = 26 :=
by sorry

#check student_average_age

end student_average_age_l163_16310


namespace polygon_diagonals_l163_16382

theorem polygon_diagonals (n : ℕ) : 
  (n ≥ 3) → (n - 3 ≤ 5) → (n = 8) :=
by sorry

end polygon_diagonals_l163_16382


namespace point_on_line_implies_m_value_l163_16380

/-- Given a point P(1, -2) on the line 4x - my + 12 = 0, prove that m = -8 -/
theorem point_on_line_implies_m_value (m : ℝ) : 
  (4 * 1 - m * (-2) + 12 = 0) → m = -8 := by
  sorry

end point_on_line_implies_m_value_l163_16380


namespace product_mod_500_l163_16326

theorem product_mod_500 : (2367 * 1023) % 500 = 41 := by
  sorry

end product_mod_500_l163_16326


namespace ellipse_eccentricity_and_fixed_point_l163_16388

noncomputable section

-- Define the ellipse Γ
def Γ (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 = 1

-- Define the circle E
def E (x y : ℝ) : Prop := x^2 + (y - 3/2)^2 = 4

-- Define point D
def D : ℝ × ℝ := (0, -1/2)

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define symmetry about y-axis
def symmetric_about_y_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = -p2.1 ∧ p1.2 = p2.2

theorem ellipse_eccentricity_and_fixed_point 
  (a : ℝ) (A B : ℝ × ℝ) 
  (h1 : a > 1)
  (h2 : Γ a A.1 A.2)
  (h3 : Γ a B.1 B.2)
  (h4 : E A.1 A.2)
  (h5 : E B.1 B.2)
  (h6 : distance A B = 2 * Real.sqrt 3) :
  (∃ (e : ℝ), e = Real.sqrt 3 / 2 ∧ 
    e^2 = 1 - 1/a^2) ∧ 
  (∀ (M N N' : ℝ × ℝ), 
    Γ a M.1 M.2 → Γ a N.1 N.2 → symmetric_about_y_axis N N' →
    (∃ (k : ℝ), M.2 - D.2 = k * (M.1 - D.1) ∧ 
                N.2 - D.2 = k * (N.1 - D.1)) →
    ∃ (t : ℝ), M.2 - N'.2 = (M.1 - N'.1) * (0 - M.1) / (t - M.1) ∧ 
               t = 0 ∧ M.2 - (0 - M.1) * (M.2 - N'.2) / (M.1 - N'.1) = -2) := by
  sorry

end ellipse_eccentricity_and_fixed_point_l163_16388


namespace f_min_at_zero_l163_16393

def f (x : ℝ) : ℝ := (x^2 - 4)^3 + 1

theorem f_min_at_zero :
  (∀ x : ℝ, f 0 ≤ f x) ∧ f 0 = -63 :=
sorry

end f_min_at_zero_l163_16393
