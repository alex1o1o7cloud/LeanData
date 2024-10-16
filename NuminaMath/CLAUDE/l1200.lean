import Mathlib

namespace NUMINAMATH_CALUDE_fraction_difference_l1200_120013

theorem fraction_difference : (7 : ℚ) / 12 - (3 : ℚ) / 8 = (5 : ℚ) / 24 := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_l1200_120013


namespace NUMINAMATH_CALUDE_omicron_ba3_sample_size_l1200_120059

/-- The number of Omicron BA.3 virus strains in a stratified random sample -/
theorem omicron_ba3_sample_size 
  (total_strains : ℕ) 
  (ba3_strains : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_strains = 120) 
  (h2 : ba3_strains = 40) 
  (h3 : sample_size = 30) :
  (ba3_strains : ℚ) / total_strains * sample_size = 10 :=
sorry

end NUMINAMATH_CALUDE_omicron_ba3_sample_size_l1200_120059


namespace NUMINAMATH_CALUDE_transformation_of_curve_l1200_120015

-- Define the transformation φ
def φ (p : ℝ × ℝ) : ℝ × ℝ := (3 * p.1, 4 * p.2)

-- Define the initial curve
def initial_curve (p : ℝ × ℝ) : Prop := p.1^2 + p.2^2 = 1

-- Define the final curve
def final_curve (p : ℝ × ℝ) : Prop := p.1^2 / 9 + p.2^2 / 16 = 1

-- Theorem statement
theorem transformation_of_curve :
  ∀ p : ℝ × ℝ, initial_curve p ↔ final_curve (φ p) := by sorry

end NUMINAMATH_CALUDE_transformation_of_curve_l1200_120015


namespace NUMINAMATH_CALUDE_distribute_tickets_count_l1200_120071

/-- The number of ways to distribute 4 consecutive numbered tickets among 3 people -/
def distribute_tickets : ℕ :=
  -- Number of ways to split 4 tickets into 3 portions
  let split_ways := Nat.choose 3 2
  -- Number of ways to distribute 3 portions to 3 people
  let distribute_ways := Nat.factorial 3
  -- Total number of distribution methods
  split_ways * distribute_ways

/-- Theorem stating that the number of distribution methods is 18 -/
theorem distribute_tickets_count : distribute_tickets = 18 := by
  sorry

end NUMINAMATH_CALUDE_distribute_tickets_count_l1200_120071


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1200_120078

def vector_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (3, 2)
  let b : ℝ × ℝ := (x, 4)
  vector_parallel a b → x = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1200_120078


namespace NUMINAMATH_CALUDE_range_of_m_l1200_120017

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x - 6

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc 0 m, f x ∈ Set.Icc (-10) (-6)) ∧
  (∃ x ∈ Set.Icc 0 m, f x = -10) ∧
  (∃ x ∈ Set.Icc 0 m, f x = -6) →
  m ∈ Set.Icc 2 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1200_120017


namespace NUMINAMATH_CALUDE_hula_hoop_difference_is_three_l1200_120075

/-- The difference in hula hooping times between Nancy and Casey -/
def hula_hoop_time_difference (nancy_time : ℕ) (morgan_time : ℕ) : ℕ :=
  let casey_time := morgan_time / 3
  nancy_time - casey_time

/-- Theorem stating that the difference in hula hooping times between Nancy and Casey is 3 minutes -/
theorem hula_hoop_difference_is_three :
  hula_hoop_time_difference 10 21 = 3 := by
  sorry

end NUMINAMATH_CALUDE_hula_hoop_difference_is_three_l1200_120075


namespace NUMINAMATH_CALUDE_sasha_salt_adjustment_l1200_120028

theorem sasha_salt_adjustment (x y : ℝ) 
  (h1 : y > 0)  -- Yesterday's extra salt was positive
  (h2 : x > 0)  -- Initial salt amount is positive
  (h3 : x + y = 2*x + y/2)  -- Total salt needed is the same for both days
  : (3*x) / (2*x) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sasha_salt_adjustment_l1200_120028


namespace NUMINAMATH_CALUDE_digit_sum_theorem_l1200_120049

/-- Given single-digit integers a and b satisfying the equation 3a * (10b + 4) = 146, 
    prove that a + b = 13 -/
theorem digit_sum_theorem (a b : ℕ) : 
  a < 10 → b < 10 → 3 * a * (10 * b + 4) = 146 → a + b = 13 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_theorem_l1200_120049


namespace NUMINAMATH_CALUDE_racket_purchase_cost_l1200_120073

/-- The cost of two rackets with discounts -/
def total_cost (original_price : ℝ) : ℝ :=
  let first_racket_cost := original_price * (1 - 0.2)
  let second_racket_cost := original_price * 0.5
  first_racket_cost + second_racket_cost

/-- Theorem stating the total cost of two rackets -/
theorem racket_purchase_cost :
  total_cost 60 = 78 := by sorry

end NUMINAMATH_CALUDE_racket_purchase_cost_l1200_120073


namespace NUMINAMATH_CALUDE_four_purchase_options_l1200_120055

/-- Represents the number of different ways to buy masks and alcohol wipes -/
def purchase_options : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 3 * p.1 + 2 * p.2 = 30) (Finset.product (Finset.range 31) (Finset.range 31))).card

/-- Theorem stating that there are exactly 4 ways to purchase masks and alcohol wipes -/
theorem four_purchase_options : purchase_options = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_purchase_options_l1200_120055


namespace NUMINAMATH_CALUDE_sum_of_roots_zero_l1200_120092

theorem sum_of_roots_zero (p q : ℝ) : 
  (∀ x, x^2 + p*x + q = 0 ↔ x = p ∨ x = q) → 
  p = 2*q → 
  p + q = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_zero_l1200_120092


namespace NUMINAMATH_CALUDE_friends_eating_pizza_l1200_120029

/-- The number of friends eating pizza with Ron -/
def num_friends : ℕ := 2

/-- The number of slices in the pizza -/
def total_slices : ℕ := 12

/-- The number of slices each person ate -/
def slices_per_person : ℕ := 4

/-- The total number of people eating, including Ron -/
def total_people : ℕ := total_slices / slices_per_person

theorem friends_eating_pizza : 
  num_friends = total_people - 1 :=
sorry

end NUMINAMATH_CALUDE_friends_eating_pizza_l1200_120029


namespace NUMINAMATH_CALUDE_gardener_flower_expenses_l1200_120035

/-- The total expenses for flowers ordered by a gardener -/
theorem gardener_flower_expenses :
  let tulips : ℕ := 250
  let carnations : ℕ := 375
  let roses : ℕ := 320
  let price_per_flower : ℕ := 2
  let total_flowers : ℕ := tulips + carnations + roses
  let total_expenses : ℕ := total_flowers * price_per_flower
  total_expenses = 1890 := by sorry

end NUMINAMATH_CALUDE_gardener_flower_expenses_l1200_120035


namespace NUMINAMATH_CALUDE_total_flowers_l1200_120087

def flower_types := 4
def flowers_per_type := 40

theorem total_flowers :
  flower_types * flowers_per_type = 160 := by
  sorry

end NUMINAMATH_CALUDE_total_flowers_l1200_120087


namespace NUMINAMATH_CALUDE_diamond_two_three_l1200_120021

def diamond (a b : ℝ) : ℝ := a^3 * b^2 - b + 2

theorem diamond_two_three : diamond 2 3 = 71 := by sorry

end NUMINAMATH_CALUDE_diamond_two_three_l1200_120021


namespace NUMINAMATH_CALUDE_regular_hourly_wage_l1200_120045

theorem regular_hourly_wage (
  working_days_per_week : ℕ)
  (working_hours_per_day : ℕ)
  (overtime_rate : ℚ)
  (total_earnings : ℚ)
  (total_hours_worked : ℕ)
  (weeks : ℕ)
  (h1 : working_days_per_week = 6)
  (h2 : working_hours_per_day = 10)
  (h3 : overtime_rate = 21/5)
  (h4 : total_earnings = 525)
  (h5 : total_hours_worked = 245)
  (h6 : weeks = 4) :
  let regular_hours := working_days_per_week * working_hours_per_day * weeks
  let overtime_hours := total_hours_worked - regular_hours
  let overtime_pay := overtime_rate * overtime_hours
  let regular_pay := total_earnings - overtime_pay
  let regular_hourly_wage := regular_pay / regular_hours
  regular_hourly_wage = 21/10 := by
sorry

end NUMINAMATH_CALUDE_regular_hourly_wage_l1200_120045


namespace NUMINAMATH_CALUDE_root_between_a_and_b_l1200_120064

theorem root_between_a_and_b (p q a b : ℝ) 
  (ha : a^2 + p*a + q = 0)
  (hb : b^2 - p*b - q = 0)
  (hq : q ≠ 0) :
  ∃ c ∈ Set.Ioo a b, c^2 + 2*p*c + 2*q = 0 := by
sorry

end NUMINAMATH_CALUDE_root_between_a_and_b_l1200_120064


namespace NUMINAMATH_CALUDE_remaining_liquid_weight_l1200_120000

/-- Proves that the weight of the remaining liquid after evaporation is 6 kg --/
theorem remaining_liquid_weight (initial_weight : ℝ) (evaporated_water : ℝ) (added_solution : ℝ) 
  (initial_x_percent : ℝ) (final_x_percent : ℝ) :
  initial_weight = 8 →
  evaporated_water = 2 →
  added_solution = 2 →
  initial_x_percent = 0.2 →
  final_x_percent = 0.25 →
  ∃ (remaining_weight : ℝ),
    remaining_weight = initial_weight - evaporated_water ∧
    (remaining_weight + added_solution) * final_x_percent = 
      initial_weight * initial_x_percent + added_solution * initial_x_percent ∧
    remaining_weight = 6 :=
by sorry

end NUMINAMATH_CALUDE_remaining_liquid_weight_l1200_120000


namespace NUMINAMATH_CALUDE_ann_shorts_purchase_l1200_120090

/-- Calculates the maximum number of shorts Ann can buy -/
def max_shorts (total_spent : ℕ) (shoe_cost : ℕ) (shorts_cost : ℕ) (num_tops : ℕ) : ℕ :=
  ((total_spent - shoe_cost) / shorts_cost)

theorem ann_shorts_purchase :
  let total_spent := 75
  let shoe_cost := 20
  let shorts_cost := 7
  let num_tops := 4
  max_shorts total_spent shoe_cost shorts_cost num_tops = 7 := by
  sorry

#eval max_shorts 75 20 7 4

end NUMINAMATH_CALUDE_ann_shorts_purchase_l1200_120090


namespace NUMINAMATH_CALUDE_system_solution_l1200_120098

theorem system_solution (x y : ℝ) (eq1 : x + y = 2) (eq2 : 3 * x - y = 8) : x - y = 3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1200_120098


namespace NUMINAMATH_CALUDE_solve_star_equation_l1200_120018

/-- Custom binary operation -/
def star (a b : ℚ) : ℚ := a * b + 3 * b - 2 * a

/-- Theorem stating the solution to the equation -/
theorem solve_star_equation : ∃ x : ℚ, star 3 x = 23 ∧ x = 29 / 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_star_equation_l1200_120018


namespace NUMINAMATH_CALUDE_total_paper_weight_is_2074_l1200_120012

/-- Calculates the total weight of paper Barbara removed from the chest of drawers. -/
def total_paper_weight : ℕ :=
  let bundle_size : ℕ := 2
  let bunch_size : ℕ := 4
  let heap_size : ℕ := 20
  let pile_size : ℕ := 10
  let stack_size : ℕ := 5

  let colored_bundles : ℕ := 3
  let white_bunches : ℕ := 2
  let scrap_heaps : ℕ := 5
  let glossy_piles : ℕ := 4
  let cardstock_stacks : ℕ := 3

  let colored_weight : ℕ := 8
  let white_weight : ℕ := 12
  let scrap_weight : ℕ := 10
  let glossy_weight : ℕ := 15
  let cardstock_weight : ℕ := 22

  let colored_total := colored_bundles * bundle_size * colored_weight
  let white_total := white_bunches * bunch_size * white_weight
  let scrap_total := scrap_heaps * heap_size * scrap_weight
  let glossy_total := glossy_piles * pile_size * glossy_weight
  let cardstock_total := cardstock_stacks * stack_size * cardstock_weight

  colored_total + white_total + scrap_total + glossy_total + cardstock_total

theorem total_paper_weight_is_2074 : total_paper_weight = 2074 := by
  sorry

end NUMINAMATH_CALUDE_total_paper_weight_is_2074_l1200_120012


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1200_120058

theorem perpendicular_vectors_x_value :
  ∀ (x : ℝ),
  let a : Fin 2 → ℝ := ![(-2), 1]
  let b : Fin 2 → ℝ := ![x, 2]
  (∀ i : Fin 2, a i * b i = 0) →
  x = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1200_120058


namespace NUMINAMATH_CALUDE_zoo_meat_supply_duration_l1200_120067

/-- The number of full days a meat supply lasts for a group of animals -/
def meat_supply_duration (lion_consumption tiger_consumption leopard_consumption hyena_consumption total_meat : ℕ) : ℕ :=
  (total_meat / (lion_consumption + tiger_consumption + leopard_consumption + hyena_consumption))

/-- Theorem: Given the specified daily meat consumption for four animals and a total meat supply of 500 kg, the meat supply will last for 7 full days -/
theorem zoo_meat_supply_duration :
  meat_supply_duration 25 20 15 10 500 = 7 := by
  sorry

end NUMINAMATH_CALUDE_zoo_meat_supply_duration_l1200_120067


namespace NUMINAMATH_CALUDE_two_digit_addition_problem_l1200_120026

theorem two_digit_addition_problem (A B : ℕ) : 
  A ≠ B →
  A * 10 + 7 + 30 + B = 73 →
  A = 3 := by
sorry

end NUMINAMATH_CALUDE_two_digit_addition_problem_l1200_120026


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1200_120016

theorem quadratic_equation_solution (C : ℝ) (h : C = 3) :
  ∃ x : ℝ, 3 * x^2 - 6 * x + C = 0 ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1200_120016


namespace NUMINAMATH_CALUDE_solution_set_when_m_equals_3_range_of_m_for_inequality_l1200_120041

-- Define the function f
def f (x m : ℝ) : ℝ := |x + 6| - |m - x|

-- Theorem for part I
theorem solution_set_when_m_equals_3 :
  {x : ℝ | f x 3 ≥ 5} = {x : ℝ | x ≥ 1} := by sorry

-- Theorem for part II
theorem range_of_m_for_inequality :
  {m : ℝ | ∀ x, f x m ≤ 7} = {m : ℝ | -13 ≤ m ∧ m ≤ 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_when_m_equals_3_range_of_m_for_inequality_l1200_120041


namespace NUMINAMATH_CALUDE_fair_coin_tails_probability_l1200_120089

-- Define a fair coin
def FairCoin : Type := Unit

-- Define the possible outcomes of a coin flip
inductive CoinOutcome : Type
| Heads : CoinOutcome
| Tails : CoinOutcome

-- Define the probability of getting tails for a fair coin
def probTails (coin : FairCoin) : ℚ := 1 / 2

-- Theorem statement
theorem fair_coin_tails_probability (coin : FairCoin) (previous_flips : List CoinOutcome) :
  probTails coin = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fair_coin_tails_probability_l1200_120089


namespace NUMINAMATH_CALUDE_find_A_in_terms_of_B_l1200_120065

/-- Given functions f and g, prove the value of A in terms of B -/
theorem find_A_in_terms_of_B (B : ℝ) (hB : B ≠ 0) :
  let f := fun x => A * x - 3 * B^2 + B * x^2
  let g := fun x => B * x^2
  let A := (3 - 16 * B^2) / 4
  f (g 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_find_A_in_terms_of_B_l1200_120065


namespace NUMINAMATH_CALUDE_jessica_bank_account_l1200_120002

theorem jessica_bank_account (initial_balance : ℝ) 
  (withdrawal : ℝ) (final_balance : ℝ) (deposit_fraction : ℝ) :
  withdrawal = 200 ∧
  initial_balance - withdrawal = (3/5) * initial_balance ∧
  final_balance = 360 ∧
  final_balance = (initial_balance - withdrawal) + deposit_fraction * (initial_balance - withdrawal) →
  deposit_fraction = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_jessica_bank_account_l1200_120002


namespace NUMINAMATH_CALUDE_solve_dales_potatoes_l1200_120048

/-- The number of potatoes Dale bought -/
def dales_potatoes (marcel_corn dale_corn marcel_potatoes total_vegetables : ℕ) : ℕ :=
  total_vegetables - (marcel_corn + dale_corn + marcel_potatoes)

theorem solve_dales_potatoes :
  ∀ (marcel_corn dale_corn marcel_potatoes total_vegetables : ℕ),
    marcel_corn = 10 →
    dale_corn = marcel_corn / 2 →
    marcel_potatoes = 4 →
    total_vegetables = 27 →
    dales_potatoes marcel_corn dale_corn marcel_potatoes total_vegetables = 8 := by
  sorry

end NUMINAMATH_CALUDE_solve_dales_potatoes_l1200_120048


namespace NUMINAMATH_CALUDE_distribution_of_distinct_objects_l1200_120006

theorem distribution_of_distinct_objects (n : ℕ) (m : ℕ) :
  n = 6 → m = 12 → n^m = 2985984 := by
  sorry

end NUMINAMATH_CALUDE_distribution_of_distinct_objects_l1200_120006


namespace NUMINAMATH_CALUDE_sugar_substitute_usage_l1200_120046

/-- Proves that Christopher uses 1 packet of sugar substitute per coffee --/
theorem sugar_substitute_usage
  (coffees_per_day : ℕ)
  (packets_per_box : ℕ)
  (cost_per_box : ℚ)
  (total_cost : ℚ)
  (total_days : ℕ)
  (h1 : coffees_per_day = 2)
  (h2 : packets_per_box = 30)
  (h3 : cost_per_box = 4)
  (h4 : total_cost = 24)
  (h5 : total_days = 90) :
  (total_cost / cost_per_box * packets_per_box) / (total_days * coffees_per_day) = 1 := by
  sorry


end NUMINAMATH_CALUDE_sugar_substitute_usage_l1200_120046


namespace NUMINAMATH_CALUDE_largest_C_inequality_l1200_120050

theorem largest_C_inequality : 
  ∃ (C : ℝ), C = 17/4 ∧ 
  (∀ (x y : ℝ), y ≥ 4*x ∧ x > 0 → x^2 + y^2 ≥ C*x*y) ∧
  (∀ (C' : ℝ), C' > C → 
    ∃ (x y : ℝ), y ≥ 4*x ∧ x > 0 ∧ x^2 + y^2 < C'*x*y) :=
sorry

end NUMINAMATH_CALUDE_largest_C_inequality_l1200_120050


namespace NUMINAMATH_CALUDE_problem_solution_l1200_120009

-- Define x as the solution to the equation x = 1 + √3 / x
noncomputable def x : ℝ := Real.sqrt 3 + 1

-- State the theorem
theorem problem_solution : 
  1 / ((x + 1) * (x - 2)) = -(Real.sqrt 3 + 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1200_120009


namespace NUMINAMATH_CALUDE_belt_length_sufficient_l1200_120011

/-- Given three pulleys with parallel axes and identical radii, prove that 
    a 54 cm cord is sufficient for the belt connecting them. -/
theorem belt_length_sufficient 
  (r : ℝ) 
  (O₁O₂ O₁O₃ O₃_to_plane : ℝ) 
  (h_r : r = 2)
  (h_O₁O₂ : O₁O₂ = 12)
  (h_O₁O₃ : O₁O₃ = 10)
  (h_O₃_to_plane : O₃_to_plane = 8) :
  ∃ (belt_length : ℝ), 
    belt_length < 54 ∧ 
    belt_length = 
      O₁O₂ + O₁O₃ + Real.sqrt (O₁O₂^2 + O₁O₃^2 - 2 * O₁O₂ * O₁O₃ * (O₃_to_plane / O₁O₃)) + 
      2 * π * r :=
by sorry

end NUMINAMATH_CALUDE_belt_length_sufficient_l1200_120011


namespace NUMINAMATH_CALUDE_zhe_same_meaning_and_usage_l1200_120066

/-- Represents a function word in classical Chinese --/
structure FunctionWord where
  word : String
  meaning : String
  usage : String

/-- Represents a sentence in classical Chinese --/
structure Sentence where
  text : String
  functionWords : List FunctionWord

/-- The function word "者" as it appears in the first sentence --/
def zhe1 : FunctionWord := {
  word := "者",
  meaning := "the person",
  usage := "nominalizer"
}

/-- The function word "者" as it appears in the second sentence --/
def zhe2 : FunctionWord := {
  word := "者",
  meaning := "the person",
  usage := "nominalizer"
}

/-- The first sentence containing "者" --/
def sentence1 : Sentence := {
  text := "智者能勿丧",
  functionWords := [zhe1]
}

/-- The second sentence containing "者" --/
def sentence2 : Sentence := {
  text := "所知贫穷者，将从我乎？",
  functionWords := [zhe2]
}

/-- Theorem stating that the function word "者" has the same meaning and usage in both sentences --/
theorem zhe_same_meaning_and_usage : 
  zhe1.meaning = zhe2.meaning ∧ zhe1.usage = zhe2.usage :=
sorry

end NUMINAMATH_CALUDE_zhe_same_meaning_and_usage_l1200_120066


namespace NUMINAMATH_CALUDE_triangle_square_side_ratio_l1200_120097

theorem triangle_square_side_ratio :
  ∀ (t s : ℝ),
  (3 * t = 15) →  -- Perimeter of equilateral triangle
  (4 * s = 12) →  -- Perimeter of square
  (t / s = 5 / 3) :=  -- Ratio of side lengths
by
  sorry

end NUMINAMATH_CALUDE_triangle_square_side_ratio_l1200_120097


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l1200_120023

theorem ratio_x_to_y (x y : ℝ) (h : 0.1 * x = 0.2 * y) : x / y = 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l1200_120023


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1200_120020

theorem complex_equation_solution (z : ℂ) : (z * Complex.I = 4 + 3 * Complex.I) → z = 3 - 4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1200_120020


namespace NUMINAMATH_CALUDE_rationalize_sqrt_five_twelfths_l1200_120095

theorem rationalize_sqrt_five_twelfths : 
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_sqrt_five_twelfths_l1200_120095


namespace NUMINAMATH_CALUDE_exists_term_with_100_nines_l1200_120037

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
def ArithmeticProgression (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A number contains 100 consecutive nines if it can be written in the form
    k * 10^(100 + m) + (10^100 - 1) for some natural numbers k and m. -/
def Contains100ConsecutiveNines (n : ℕ) : Prop :=
  ∃ k m : ℕ, n = k * (10^(100 + m)) + (10^100 - 1)

/-- In any infinite arithmetic progression of natural numbers,
    there exists a term that contains 100 consecutive nines. -/
theorem exists_term_with_100_nines (a : ℕ → ℕ) (h : ArithmeticProgression a) :
  ∃ n : ℕ, Contains100ConsecutiveNines (a n) := by
  sorry


end NUMINAMATH_CALUDE_exists_term_with_100_nines_l1200_120037


namespace NUMINAMATH_CALUDE_new_persons_weight_l1200_120069

theorem new_persons_weight (W : ℝ) (X Y : ℝ) :
  (∀ (T : ℝ), T = 8 * W) →
  (∀ (new_total : ℝ), new_total = 8 * W - 140 + X + Y) →
  (∀ (new_avg : ℝ), new_avg = W + 5) →
  (∀ (new_total : ℝ), new_total = 8 * new_avg) →
  X + Y = 180 := by
sorry

end NUMINAMATH_CALUDE_new_persons_weight_l1200_120069


namespace NUMINAMATH_CALUDE_t_cube_surface_area_l1200_120025

/-- Represents a T-shaped structure made of unit cubes -/
structure TCube where
  vertical_cubes : ℕ
  horizontal_cubes : ℕ
  intersection_position : ℕ

/-- Calculates the surface area of a T-shaped structure -/
def surface_area (t : TCube) : ℕ :=
  sorry

/-- The specific T-shaped structure described in the problem -/
def problem_t_cube : TCube :=
  { vertical_cubes := 5
  , horizontal_cubes := 5
  , intersection_position := 3 }

theorem t_cube_surface_area :
  surface_area problem_t_cube = 33 :=
sorry

end NUMINAMATH_CALUDE_t_cube_surface_area_l1200_120025


namespace NUMINAMATH_CALUDE_smaller_number_proof_l1200_120010

theorem smaller_number_proof (x y : ℤ) 
  (sum_condition : x + y = 84)
  (ratio_condition : y = 3 * x) :
  x = 21 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l1200_120010


namespace NUMINAMATH_CALUDE_johns_paintball_expenditure_l1200_120076

/-- John's monthly paintball expenditure --/
theorem johns_paintball_expenditure :
  ∀ (plays_per_month boxes_per_play box_cost : ℕ),
  plays_per_month = 3 →
  boxes_per_play = 3 →
  box_cost = 25 →
  plays_per_month * boxes_per_play * box_cost = 225 :=
by
  sorry

end NUMINAMATH_CALUDE_johns_paintball_expenditure_l1200_120076


namespace NUMINAMATH_CALUDE_tank_emptying_rate_l1200_120052

/-- Proves that given a tank of 30 cubic feet, with an inlet pipe rate of 5 cubic inches/min,
    one outlet pipe rate of 9 cubic inches/min, and a total emptying time of 4320 minutes
    when all pipes are open, the rate of the second outlet pipe is 8 cubic inches/min. -/
theorem tank_emptying_rate (tank_volume : ℝ) (inlet_rate : ℝ) (outlet_rate1 : ℝ)
    (emptying_time : ℝ) (inches_per_foot : ℝ) :
  tank_volume = 30 →
  inlet_rate = 5 →
  outlet_rate1 = 9 →
  emptying_time = 4320 →
  inches_per_foot = 12 →
  ∃ (outlet_rate2 : ℝ),
    outlet_rate2 = 8 ∧
    tank_volume * inches_per_foot^3 = (outlet_rate1 + outlet_rate2 - inlet_rate) * emptying_time :=
by sorry

end NUMINAMATH_CALUDE_tank_emptying_rate_l1200_120052


namespace NUMINAMATH_CALUDE_rulers_in_drawer_l1200_120005

/-- The total number of rulers in a drawer after an addition. -/
def total_rulers (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem: Given 11 initial rulers and 14 added rulers, the total is 25. -/
theorem rulers_in_drawer : total_rulers 11 14 = 25 := by
  sorry

end NUMINAMATH_CALUDE_rulers_in_drawer_l1200_120005


namespace NUMINAMATH_CALUDE_best_overall_value_l1200_120038

structure Box where
  brand : String
  size : Nat
  price : Rat
  quality : Rat

def pricePerOunce (b : Box) : Rat :=
  b.price / b.size

def overallValue (b : Box) : Rat :=
  b.quality / (pricePerOunce b)

theorem best_overall_value (box1 box2 box3 box4 : Box) 
  (h1 : box1 = { brand := "A", size := 30, price := 480/100, quality := 9/2 })
  (h2 : box2 = { brand := "A", size := 20, price := 340/100, quality := 9/2 })
  (h3 : box3 = { brand := "B", size := 15, price := 200/100, quality := 39/10 })
  (h4 : box4 = { brand := "B", size := 25, price := 325/100, quality := 39/10 }) :
  overallValue box1 ≥ overallValue box2 ∧ 
  overallValue box1 ≥ overallValue box3 ∧ 
  overallValue box1 ≥ overallValue box4 := by
  sorry

#check best_overall_value

end NUMINAMATH_CALUDE_best_overall_value_l1200_120038


namespace NUMINAMATH_CALUDE_function_value_at_2014_l1200_120053

noncomputable def f (a b α β x : ℝ) : ℝ :=
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β) + 4

theorem function_value_at_2014 
  (a b α β : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hα : α ≠ 0) (hβ : β ≠ 0)
  (h : f a b α β 2013 = 5) :
  f a b α β 2014 = 3 :=
sorry

end NUMINAMATH_CALUDE_function_value_at_2014_l1200_120053


namespace NUMINAMATH_CALUDE_store_discount_calculation_l1200_120033

theorem store_discount_calculation (initial_discount : ℝ) (additional_discount : ℝ) (claimed_discount : ℝ) :
  initial_discount = 0.30 →
  additional_discount = 0.15 →
  claimed_discount = 0.45 →
  let remaining_after_initial := 1 - initial_discount
  let remaining_after_both := remaining_after_initial * (1 - additional_discount)
  let actual_discount := 1 - remaining_after_both
  (actual_discount = 0.405 ∧ claimed_discount - actual_discount = 0.045) := by
  sorry

#check store_discount_calculation

end NUMINAMATH_CALUDE_store_discount_calculation_l1200_120033


namespace NUMINAMATH_CALUDE_car_speed_problem_l1200_120084

/-- Proves that car R's average speed is 50 miles per hour given the problem conditions -/
theorem car_speed_problem (distance : ℝ) (time_difference : ℝ) (speed_difference : ℝ) :
  distance = 800 ∧ 
  time_difference = 2 ∧ 
  speed_difference = 10 →
  ∃ (speed_r : ℝ),
    speed_r > 0 ∧
    distance / speed_r - time_difference = distance / (speed_r + speed_difference) ∧
    speed_r = 50 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l1200_120084


namespace NUMINAMATH_CALUDE_number_equality_l1200_120027

theorem number_equality : ∃ x : ℝ, (30 / 100) * x = (15 / 100) * 40 ∧ x = 20 := by
  sorry

end NUMINAMATH_CALUDE_number_equality_l1200_120027


namespace NUMINAMATH_CALUDE_committee_selection_count_l1200_120051

/-- The number of committee members -/
def total_members : ℕ := 5

/-- The number of roles to be filled -/
def roles_to_fill : ℕ := 3

/-- The number of members ineligible for the entertainment officer role -/
def ineligible_members : ℕ := 2

/-- The number of ways to select members for the given roles under the specified conditions -/
def selection_count : ℕ := 36

theorem committee_selection_count : 
  (total_members - ineligible_members) * 
  (total_members - 1) * 
  (total_members - 2) = selection_count :=
by sorry

end NUMINAMATH_CALUDE_committee_selection_count_l1200_120051


namespace NUMINAMATH_CALUDE_prob_monochromatic_triangle_l1200_120070

/-- A complete graph K6 where each edge is colored red or blue -/
def ColoredK6 := Fin 15 → Bool

/-- The probability of an edge being red (or blue) -/
def p : ℚ := 1/2

/-- The set of all possible colorings of K6 -/
def allColorings : Set ColoredK6 := Set.univ

/-- A triangle in K6 -/
structure Triangle :=
  (a b c : Fin 6)
  (ha : a < b)
  (hb : b < c)

/-- The set of all triangles in K6 -/
def allTriangles : Set Triangle := sorry

/-- A coloring has a monochromatic triangle -/
def hasMonochromaticTriangle (coloring : ColoredK6) : Prop := sorry

/-- The probability of having at least one monochromatic triangle -/
noncomputable def probMonochromaticTriangle : ℚ := sorry

theorem prob_monochromatic_triangle :
  probMonochromaticTriangle = 1048575/1048576 := by sorry

end NUMINAMATH_CALUDE_prob_monochromatic_triangle_l1200_120070


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l1200_120091

/-- Given a circle with equation x^2 + y^2 - 2x + 6y + 6 = 0, prove that its center is at (1, -3) and its radius is 2 -/
theorem circle_center_and_radius :
  let circle_eq := fun (x y : ℝ) => x^2 + y^2 - 2*x + 6*y + 6 = 0
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (1, -3) ∧ 
    radius = 2 ∧
    ∀ (x y : ℝ), circle_eq x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l1200_120091


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l1200_120024

/-- A line in the form ax + by + c = 0 --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if their slopes are equal --/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b

/-- A point lies on a line if it satisfies the line's equation --/
def point_on_line (x y : ℝ) (l : Line) : Prop :=
  l.a * x + l.b * y + l.c = 0

theorem parallel_line_through_point :
  let l1 : Line := { a := 3, b := 4, c := 1 }
  let l2 : Line := { a := 3, b := 4, c := -11 }
  parallel l1 l2 ∧ point_on_line 1 2 l2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l1200_120024


namespace NUMINAMATH_CALUDE_eggs_last_24_days_l1200_120007

/-- Calculates the number of days eggs will last given initial eggs, daily egg laying, and daily consumption. -/
def days_eggs_last (initial_eggs : ℕ) (daily_laid : ℕ) (daily_consumed : ℕ) : ℕ :=
  initial_eggs / (daily_consumed - daily_laid)

/-- Theorem: Given 72 initial eggs, a hen laying 1 egg per day, and a family consuming 4 eggs per day, the eggs will last for 24 days. -/
theorem eggs_last_24_days :
  days_eggs_last 72 1 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_eggs_last_24_days_l1200_120007


namespace NUMINAMATH_CALUDE_log_expression_equals_two_l1200_120072

-- Define the logarithm function (base 10)
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_two :
  2 * (log10 2) + log10 25 = 2 := by sorry

end NUMINAMATH_CALUDE_log_expression_equals_two_l1200_120072


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_values_l1200_120086

theorem perpendicular_lines_a_values (a : ℝ) : 
  let l1 := {(x, y) : ℝ × ℝ | a * x + 2 * y + 1 = 0}
  let l2 := {(x, y) : ℝ × ℝ | (3 - a) * x - y + a = 0}
  let slope1 := -a / 2
  let slope2 := 3 - a
  (slope1 * slope2 = -1) → (a = 1 ∨ a = 2) := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_values_l1200_120086


namespace NUMINAMATH_CALUDE_mobius_speed_theorem_l1200_120039

theorem mobius_speed_theorem (total_distance : ℝ) (loaded_speed : ℝ) (total_time : ℝ) (rest_time : ℝ) :
  total_distance = 286 →
  loaded_speed = 11 →
  total_time = 26 →
  rest_time = 2 →
  ∃ v : ℝ, v > 0 ∧ (total_distance / 2) / loaded_speed + (total_distance / 2) / v = total_time - rest_time ∧ v = 13 := by
  sorry

end NUMINAMATH_CALUDE_mobius_speed_theorem_l1200_120039


namespace NUMINAMATH_CALUDE_solution_set_implies_range_l1200_120063

/-- The solution set of the inequality ax^2 + ax - 4 < 0 is ℝ -/
def solution_set_is_reals (a : ℝ) : Prop :=
  ∀ x, a * x^2 + a * x - 4 < 0

/-- The range of a is (-16, 0] -/
def range_of_a : Set ℝ := Set.Ioc (-16) 0

theorem solution_set_implies_range :
  (∃ a, solution_set_is_reals a) → (∀ a, solution_set_is_reals a ↔ a ∈ range_of_a) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_implies_range_l1200_120063


namespace NUMINAMATH_CALUDE_rosette_area_l1200_120036

/-- The area of a rosette formed by four semicircles on the sides of a square -/
theorem rosette_area (a : ℝ) (h : a > 0) :
  let square_side := a
  let semicircle_radius := a / 2
  let rosette_area := (a^2 * (Real.pi - 2)) / 2
  rosette_area = (square_side^2 * (Real.pi - 2)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_rosette_area_l1200_120036


namespace NUMINAMATH_CALUDE_four_digit_integers_with_specific_remainders_l1200_120047

theorem four_digit_integers_with_specific_remainders :
  ∃! (s : Finset ℕ), 
    (∀ n ∈ s, 1000 ≤ n ∧ n < 10000 ∧ 
      n % 7 = 1 ∧ n % 10 = 3 ∧ n % 13 = 5) ∧
    (∀ n, 1000 ≤ n ∧ n < 10000 ∧ 
      n % 7 = 1 ∧ n % 10 = 3 ∧ n % 13 = 5 → n ∈ s) ∧
    Finset.card s = 6 :=
by sorry

end NUMINAMATH_CALUDE_four_digit_integers_with_specific_remainders_l1200_120047


namespace NUMINAMATH_CALUDE_soccer_stars_draw_points_l1200_120060

/-- Represents a soccer team's season statistics -/
structure SoccerTeamStats where
  total_games : ℕ
  games_won : ℕ
  games_lost : ℕ
  points_per_win : ℕ
  total_points : ℕ

/-- Calculates the points earned for a draw given a team's season statistics -/
def points_per_draw (stats : SoccerTeamStats) : ℕ :=
  let games_drawn := stats.total_games - stats.games_won - stats.games_lost
  let points_from_wins := stats.games_won * stats.points_per_win
  let points_from_draws := stats.total_points - points_from_wins
  points_from_draws / games_drawn

/-- Theorem stating that Team Soccer Stars earns 1 point for each draw -/
theorem soccer_stars_draw_points :
  let stats : SoccerTeamStats := {
    total_games := 20,
    games_won := 14,
    games_lost := 2,
    points_per_win := 3,
    total_points := 46
  }
  points_per_draw stats = 1 := by sorry

end NUMINAMATH_CALUDE_soccer_stars_draw_points_l1200_120060


namespace NUMINAMATH_CALUDE_largest_root_of_g_l1200_120080

def g (x : ℝ) : ℝ := 10 * x^4 - 17 * x^2 + 7

theorem largest_root_of_g :
  ∃ (r : ℝ), r = Real.sqrt (7/5) ∧
  g r = 0 ∧
  ∀ (x : ℝ), g x = 0 → x ≤ r :=
sorry

end NUMINAMATH_CALUDE_largest_root_of_g_l1200_120080


namespace NUMINAMATH_CALUDE_four_tuple_count_l1200_120079

theorem four_tuple_count (p : ℕ) (hp : Prime p) : 
  (Finset.filter 
    (fun (t : ℕ × ℕ × ℕ × ℕ) => 
      0 < t.1 ∧ t.1 < p - 1 ∧
      0 < t.2.1 ∧ t.2.1 < p - 1 ∧
      0 < t.2.2.1 ∧ t.2.2.1 < p - 1 ∧
      0 < t.2.2.2 ∧ t.2.2.2 < p - 1 ∧
      (t.1 * t.2.2.2) % p = (t.2.1 * t.2.2.1) % p)
    (Finset.product 
      (Finset.range (p - 1)) 
      (Finset.product 
        (Finset.range (p - 1)) 
        (Finset.product 
          (Finset.range (p - 1)) 
          (Finset.range (p - 1)))))).card = (p - 1)^3 :=
by sorry


end NUMINAMATH_CALUDE_four_tuple_count_l1200_120079


namespace NUMINAMATH_CALUDE_defective_firecracker_fraction_l1200_120054

theorem defective_firecracker_fraction 
  (initial : ℕ) 
  (confiscated : ℕ) 
  (set_off : ℕ) 
  (h1 : initial = 48)
  (h2 : confiscated = 12)
  (h3 : set_off = 15)
  (h4 : set_off * 2 = initial - confiscated - (initial - confiscated - (set_off * 2))) :
  (initial - confiscated - (set_off * 2)) / (initial - confiscated) = 1 / 6 :=
by sorry

end NUMINAMATH_CALUDE_defective_firecracker_fraction_l1200_120054


namespace NUMINAMATH_CALUDE_persons_count_l1200_120001

/-- The total number of persons in the group --/
def n : ℕ := sorry

/-- The total amount spent by the group in rupees --/
def total_spent : ℚ := 292.5

/-- The amount spent by each of the first 8 persons in rupees --/
def regular_spend : ℚ := 30

/-- The number of persons who spent the regular amount --/
def regular_count : ℕ := 8

/-- The extra amount spent by the last person compared to the average --/
def extra_spend : ℚ := 20

theorem persons_count :
  n = 9 ∧
  total_spent = regular_count * regular_spend + (total_spent / n + extra_spend) :=
sorry

end NUMINAMATH_CALUDE_persons_count_l1200_120001


namespace NUMINAMATH_CALUDE_line_perpendicular_to_parallel_line_l1200_120083

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicularLines : Line → Line → Prop)

-- State the theorem
theorem line_perpendicular_to_parallel_line
  (a b : Line) (α : Plane)
  (h1 : perpendicular a α)
  (h2 : parallel b α) :
  perpendicularLines a b :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_parallel_line_l1200_120083


namespace NUMINAMATH_CALUDE_cube_inequality_equivalence_l1200_120043

theorem cube_inequality_equivalence (a b : ℝ) : a > b ↔ a^3 > b^3 := by sorry

end NUMINAMATH_CALUDE_cube_inequality_equivalence_l1200_120043


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_8_12_l1200_120081

theorem gcd_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_8_12_l1200_120081


namespace NUMINAMATH_CALUDE_small_cuboid_height_l1200_120096

/-- Represents the dimensions of a cuboid -/
structure CuboidDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a cuboid given its dimensions -/
def cuboidVolume (d : CuboidDimensions) : ℝ :=
  d.length * d.width * d.height

theorem small_cuboid_height
  (large : CuboidDimensions)
  (small_length : ℝ)
  (small_width : ℝ)
  (num_small_cuboids : ℕ)
  (h_large : large = { length := 16, width := 10, height := 12 })
  (h_small_length : small_length = 5)
  (h_small_width : small_width = 4)
  (h_num_small : num_small_cuboids = 32) :
  ∃ (small_height : ℝ),
    cuboidVolume large = num_small_cuboids * (small_length * small_width * small_height) ∧
    small_height = 3 := by
  sorry


end NUMINAMATH_CALUDE_small_cuboid_height_l1200_120096


namespace NUMINAMATH_CALUDE_line_hyperbola_intersection_l1200_120022

theorem line_hyperbola_intersection :
  ∃ (k : ℝ), k > 0 ∧
  ∃ (x y : ℝ), y = Real.sqrt 3 * x ∧ y = k / x :=
by sorry

end NUMINAMATH_CALUDE_line_hyperbola_intersection_l1200_120022


namespace NUMINAMATH_CALUDE_triangle_sides_expression_l1200_120094

theorem triangle_sides_expression (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_triangle_ineq_1 : a + b > c)
  (h_triangle_ineq_2 : a + c > b)
  (h_triangle_ineq_3 : b + c > a) :
  |a + b + c| - |a - b - c| - |a + b - c| = a - b + c := by
  sorry

end NUMINAMATH_CALUDE_triangle_sides_expression_l1200_120094


namespace NUMINAMATH_CALUDE_vector_computation_l1200_120008

theorem vector_computation : 
  4 • !![3, -5] - 3 • !![2, -6] + 2 • !![0, 3] = !![6, 4] := by sorry

end NUMINAMATH_CALUDE_vector_computation_l1200_120008


namespace NUMINAMATH_CALUDE_complement_intersection_equals_l1200_120082

def U : Set ℕ := {1, 2, 3, 4}
def P : Set ℕ := {2, 3}
def Q : Set ℕ := {3, 4}

theorem complement_intersection_equals :
  (U \ (P ∩ Q)) = {1, 2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_l1200_120082


namespace NUMINAMATH_CALUDE_ellipse_right_angle_triangle_area_l1200_120019

/-- The ellipse with equation x²/2 + y² = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 2 + p.2^2 = 1}

/-- The foci of the ellipse -/
def Foci : Set (ℝ × ℝ) :=
  {f : ℝ × ℝ | ∃ (x y : ℝ), f = (x, y) ∧ x^2 + y^2 = 1}

/-- Angle between two vectors -/
def angle (v w : ℝ × ℝ) : ℝ := sorry

/-- Area of a triangle given by three points -/
def triangleArea (a b c : ℝ × ℝ) : ℝ := sorry

theorem ellipse_right_angle_triangle_area 
  (p : ℝ × ℝ) (f₁ f₂ : ℝ × ℝ) 
  (h_p : p ∈ Ellipse) 
  (h_f : f₁ ∈ Foci ∧ f₂ ∈ Foci ∧ f₁ ≠ f₂) 
  (h_angle : angle (f₁.1 - p.1, f₁.2 - p.2) (f₂.1 - p.1, f₂.2 - p.2) = π / 2) :
  triangleArea f₁ p f₂ = 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_right_angle_triangle_area_l1200_120019


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l1200_120004

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem perpendicular_lines_a_value (a : ℝ) :
  let l1 : ℝ → ℝ → Prop := λ x y => a * x + (3 - a) * y + 1 = 0
  let l2 : ℝ → ℝ → Prop := λ x y => x - 2 * y = 0
  let m1 : ℝ := a / (a - 3)  -- slope of l1
  let m2 : ℝ := 1 / 2        -- slope of l2
  perpendicular m1 m2 → a = 2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l1200_120004


namespace NUMINAMATH_CALUDE_odd_function_implies_m_equals_one_inequality_implies_a_range_l1200_120061

open Real

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := exp x - m / exp x

theorem odd_function_implies_m_equals_one (m : ℝ) :
  (∀ x, f m x = -f m (-x)) → m = 1 := by sorry

theorem inequality_implies_a_range (m : ℝ) :
  m = 1 →
  (∀ a : ℝ, f m (a - 1) + f m (2 * a^2) ≤ 0 → -1 ≤ a ∧ a ≤ 1/2) := by sorry

end NUMINAMATH_CALUDE_odd_function_implies_m_equals_one_inequality_implies_a_range_l1200_120061


namespace NUMINAMATH_CALUDE_machine_work_time_l1200_120068

/-- Proves that a machine making 6 shirts per minute worked for 23 minutes yesterday,
    given it made 14 shirts today and 156 shirts in total over two days. -/
theorem machine_work_time (shirts_per_minute : ℕ) (shirts_today : ℕ) (total_shirts : ℕ) :
  shirts_per_minute = 6 →
  shirts_today = 14 →
  total_shirts = 156 →
  (total_shirts - shirts_today) / shirts_per_minute = 23 :=
by sorry

end NUMINAMATH_CALUDE_machine_work_time_l1200_120068


namespace NUMINAMATH_CALUDE_least_positive_integer_to_multiple_of_three_l1200_120003

theorem least_positive_integer_to_multiple_of_three :
  ∃ (n : ℕ), n > 0 ∧ (527 + n) % 3 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ (527 + m) % 3 = 0 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_to_multiple_of_three_l1200_120003


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l1200_120042

theorem quadratic_equation_properties (m : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 - m*x + m - 2
  (∃ x, f x = 0 ∧ x = -1) → m = 1/2 ∧
  ∃ x y, x ≠ y ∧ f x = 0 ∧ f y = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l1200_120042


namespace NUMINAMATH_CALUDE_square_minus_one_roots_l1200_120031

theorem square_minus_one_roots (x : ℝ) : x^2 - 1 = 0 → x = -1 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_one_roots_l1200_120031


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l1200_120040

theorem profit_percentage_calculation 
  (cost original_profit selling_price : ℝ)
  (h1 : selling_price = cost + original_profit)
  (h2 : selling_price = 1.12 * cost + 0.53333333333333333 * selling_price) :
  original_profit / cost = 1.4 := by
sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l1200_120040


namespace NUMINAMATH_CALUDE_complex_fraction_theorem_l1200_120044

theorem complex_fraction_theorem (z₁ z₂ z₃ : ℂ) 
  (h1 : Complex.abs z₁ = Real.sqrt 5)
  (h2 : Complex.abs z₂ = Real.sqrt 5)
  (h3 : z₁ + z₃ = z₂) : 
  z₁ * z₂ / z₃^2 = -5 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_theorem_l1200_120044


namespace NUMINAMATH_CALUDE_jim_journey_l1200_120034

theorem jim_journey (total_journey : ℕ) (remaining_miles : ℕ) 
  (h1 : total_journey = 1200)
  (h2 : remaining_miles = 816) :
  total_journey - remaining_miles = 384 := by
  sorry

end NUMINAMATH_CALUDE_jim_journey_l1200_120034


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_50_l1200_120056

theorem largest_four_digit_divisible_by_50 : 
  ∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ n % 50 = 0 → n ≤ 9950 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_50_l1200_120056


namespace NUMINAMATH_CALUDE_min_value_exponential_sum_l1200_120099

theorem min_value_exponential_sum (a b : ℝ) (h : 2 * a + b = 6) :
  ∃ (min : ℝ), min = 4 * Real.sqrt 2 ∧ ∀ (x y : ℝ), 2 * x + y = 6 → 2^x + Real.sqrt 2^y ≥ min := by
  sorry

end NUMINAMATH_CALUDE_min_value_exponential_sum_l1200_120099


namespace NUMINAMATH_CALUDE_certain_number_proof_l1200_120074

theorem certain_number_proof (x : ℤ) : x - 82 = 17 → x = 99 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1200_120074


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l1200_120085

theorem triangle_angle_measure (a b : ℝ) (area : ℝ) (h1 : a = 5) (h2 : b = 8) (h3 : area = 10) :
  ∃ (C : ℝ), (C = π / 6 ∨ C = 5 * π / 6) ∧ 
  (1 / 2 * a * b * Real.sin C = area) ∧ 
  (0 < C) ∧ (C < π) := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l1200_120085


namespace NUMINAMATH_CALUDE_voronovich_inequality_l1200_120014

theorem voronovich_inequality (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a + b + c = 1) :
  (a^2 + b^2 + c^2)^2 + 6*a*b*c ≥ a*b + b*c + c*a := by
  sorry

end NUMINAMATH_CALUDE_voronovich_inequality_l1200_120014


namespace NUMINAMATH_CALUDE_shaded_area_between_circles_l1200_120062

theorem shaded_area_between_circles (r₁ r₂ : ℝ) : 
  r₁ = Real.sqrt 2 → r₂ = 2 * r₁ → π * r₂^2 - π * r₁^2 = 6 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_between_circles_l1200_120062


namespace NUMINAMATH_CALUDE_arithmetic_simplification_l1200_120088

theorem arithmetic_simplification : 2 - (-3) * 2 - 4 - (-5) - 6 - (-7) * 2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_simplification_l1200_120088


namespace NUMINAMATH_CALUDE_max_value_m_inequality_solution_l1200_120093

theorem max_value_m (a b : ℝ) (h : a ≠ b) :
  (∃ m : ℝ, ∀ M : ℝ, (∀ a b : ℝ, a ≠ b → M * |a - b| ≤ |2*a + b| + |a + 2*b|) → M ≤ m) ∧
  (∀ a b : ℝ, a ≠ b → 1 * |a - b| ≤ |2*a + b| + |a + 2*b|) :=
by sorry

theorem inequality_solution (x : ℝ) :
  |x - 1| < 1 * (2*x + 1) ↔ x > 0 :=
by sorry

end NUMINAMATH_CALUDE_max_value_m_inequality_solution_l1200_120093


namespace NUMINAMATH_CALUDE_quadratic_properties_l1200_120030

def f (x : ℝ) := 2 * x^2 - 4 * x + 3

theorem quadratic_properties :
  (∀ x y : ℝ, x < y → f x < f y) ∧ 
  (f 0 > 0) ∧ 
  (∀ x : ℝ, f x ≠ 0) ∧ 
  (∀ x y : ℝ, x < y → x < 1 → f x > f y) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_properties_l1200_120030


namespace NUMINAMATH_CALUDE_unique_solution_equation_l1200_120032

theorem unique_solution_equation (x p q : ℕ) : 
  x ≥ 2 ∧ p ≥ 2 ∧ q ≥ 2 →
  ((x + 1) ^ p - x ^ q = 1) ↔ (x = 2 ∧ p = 2 ∧ q = 3) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_equation_l1200_120032


namespace NUMINAMATH_CALUDE_discount_calculation_l1200_120077

/-- Given a cost price, prove that if the marked price is 150% of the cost price
    and the selling price results in a 1% loss on the cost price, then the discount
    (difference between marked price and selling price) is 51% of the cost price. -/
theorem discount_calculation (CP : ℝ) (CP_pos : CP > 0) : 
  let MP := 1.5 * CP
  let SP := 0.99 * CP
  MP - SP = 0.51 * CP := by sorry

end NUMINAMATH_CALUDE_discount_calculation_l1200_120077


namespace NUMINAMATH_CALUDE_tip_calculation_correct_l1200_120057

/-- Calculates the tip for a restaurant check with given conditions. -/
def calculate_tip (check_amount : ℚ) (tax_rate : ℚ) (senior_discount : ℚ) (dine_in_surcharge : ℚ) (payment : ℚ) : ℚ :=
  let total_with_tax := check_amount * (1 + tax_rate)
  let discount_amount := check_amount * senior_discount
  let surcharge_amount := check_amount * dine_in_surcharge
  let final_total := total_with_tax - discount_amount + surcharge_amount
  payment - final_total

/-- Theorem stating that the tip calculation for the given conditions results in $2.75. -/
theorem tip_calculation_correct :
  calculate_tip 15 (20/100) (10/100) (5/100) 20 = 275/100 := by
  sorry

end NUMINAMATH_CALUDE_tip_calculation_correct_l1200_120057
