import Mathlib

namespace country_club_monthly_cost_l3675_367509

/-- Calculates the monthly cost per person for a country club membership --/
def monthly_cost_per_person (
  num_people : ℕ
  ) (initial_fee_per_person : ℚ
  ) (john_payment : ℚ
  ) : ℚ :=
  let total_cost := 2 * john_payment
  let total_initial_fee := num_people * initial_fee_per_person
  let total_monthly_cost := total_cost - total_initial_fee
  let yearly_cost_per_person := total_monthly_cost / num_people
  yearly_cost_per_person / 12

theorem country_club_monthly_cost :
  monthly_cost_per_person 4 4000 32000 = 1000 := by
  sorry

end country_club_monthly_cost_l3675_367509


namespace complement_M_intersect_N_l3675_367568

-- Define the set M
def M : Set ℝ := {x | 2/x < 1}

-- Define the set N
def N : Set ℝ := {y | ∃ x, y = Real.sqrt (x - 1)}

-- The theorem to prove
theorem complement_M_intersect_N : (Set.univ \ M) ∩ N = Set.Icc 0 2 := by sorry

end complement_M_intersect_N_l3675_367568


namespace min_surface_pips_is_58_l3675_367522

/-- Represents a standard die -/
structure StandardDie :=
  (faces : Fin 6 → ℕ)
  (opposite_sum : ∀ i : Fin 6, faces i + faces (5 - i) = 7)

/-- Represents four dice glued in a 2x2 square configuration -/
structure GluedDice :=
  (dice : Fin 4 → StandardDie)

/-- Calculates the number of pips on the surface of glued dice -/
def surface_pips (gd : GluedDice) : ℕ :=
  sorry

/-- The minimum number of pips on the surface of glued dice -/
def min_surface_pips : ℕ :=
  sorry

theorem min_surface_pips_is_58 : min_surface_pips = 58 :=
  sorry

end min_surface_pips_is_58_l3675_367522


namespace quadratic_function_property_l3675_367521

theorem quadratic_function_property (a m : ℝ) (h1 : a > 0) : 
  let f : ℝ → ℝ := λ x ↦ x^2 - x + a
  f m < 0 → f (m - 1) > 0 := by
sorry

end quadratic_function_property_l3675_367521


namespace shopping_problem_l3675_367520

/-- Shopping problem -/
theorem shopping_problem (initial_amount : ℝ) (baguette_cost : ℝ) (water_cost : ℝ)
  (chocolate_cost : ℝ) (milk_cost : ℝ) (discount_rate : ℝ) (tax_rate : ℝ) :
  initial_amount = 50 →
  baguette_cost = 2 →
  water_cost = 1 →
  chocolate_cost = 1.5 →
  milk_cost = 3.5 →
  discount_rate = 0.1 →
  tax_rate = 0.07 →
  let baguette_total := 2 * baguette_cost
  let water_total := 2 * water_cost
  let chocolate_total := 2 * chocolate_cost
  let milk_total := milk_cost * (1 - discount_rate)
  let subtotal := baguette_total + water_total + chocolate_total + milk_total
  let tax := chocolate_total * tax_rate
  let total_cost := subtotal + tax
  initial_amount - total_cost = 37.64 := by
  sorry

end shopping_problem_l3675_367520


namespace ryegrass_percentage_in_y_l3675_367573

/-- Represents the composition of a seed mixture -/
structure SeedMixture where
  ryegrass : ℝ
  bluegrass : ℝ
  fescue : ℝ

/-- The final mixture of X and Y -/
structure FinalMixture where
  x_percentage : ℝ
  y_percentage : ℝ
  ryegrass_percentage : ℝ

/-- Theorem stating the percentage of ryegrass in seed mixture Y -/
theorem ryegrass_percentage_in_y
  (x : SeedMixture)
  (y : SeedMixture)
  (final : FinalMixture)
  (hx_ryegrass : x.ryegrass = 0.4)
  (hx_bluegrass : x.bluegrass = 0.6)
  (hy_fescue : y.fescue = 0.75)
  (hfinal_x : final.x_percentage = 0.13333333333333332)
  (hfinal_y : final.y_percentage = 1 - final.x_percentage)
  (hfinal_ryegrass : final.ryegrass_percentage = 0.27)
  : y.ryegrass = 0.25 := by
  sorry

end ryegrass_percentage_in_y_l3675_367573


namespace arithmetic_sequence_sum_l3675_367593

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℚ) :
  arithmetic_sequence a →
  a 1 + a 3 = 2 →
  a 3 + a 5 = 4 →
  a 7 + a 9 = 8 :=
by
  sorry

end arithmetic_sequence_sum_l3675_367593


namespace hyperbola_theorem_l3675_367514

/-- Represents a hyperbola -/
structure Hyperbola where
  equation : ℝ → ℝ → Prop

/-- The hyperbola passes through a given point -/
def passes_through (h : Hyperbola) (x y : ℝ) : Prop :=
  h.equation x y

/-- The asymptotes of the hyperbola -/
def has_asymptotes (h : Hyperbola) (f g : ℝ → ℝ) : Prop :=
  ∀ x, (h.equation x (f x) ∨ h.equation x (g x))

/-- The foci of the hyperbola -/
def foci (h : Hyperbola) : ℝ × ℝ := sorry

/-- Distance from a point to a line -/
def distance_to_line (x y : ℝ) (m b : ℝ) : ℝ := sorry

theorem hyperbola_theorem (h : Hyperbola) 
  (center_origin : h.equation 0 0)
  (asymptotes : has_asymptotes h (λ x => Real.sqrt 3 * x) (λ x => -Real.sqrt 3 * x))
  (point : passes_through h (Real.sqrt 2) (Real.sqrt 3)) :
  (∀ x y, h.equation x y ↔ x^2 - y^2/3 = 1) ∧ 
  (let (fx, fy) := foci h
   distance_to_line fx fy (Real.sqrt 3) 0 = Real.sqrt 3) := by
sorry

end hyperbola_theorem_l3675_367514


namespace candied_fruit_earnings_l3675_367565

/-- The number of candied apples made -/
def num_apples : ℕ := 15

/-- The price of each candied apple in dollars -/
def price_apple : ℚ := 2

/-- The number of candied grapes made -/
def num_grapes : ℕ := 12

/-- The price of each candied grape in dollars -/
def price_grape : ℚ := (3 : ℚ) / 2

/-- The total earnings from selling all candied apples and grapes -/
def total_earnings : ℚ := num_apples * price_apple + num_grapes * price_grape

theorem candied_fruit_earnings : total_earnings = 48 := by
  sorry

end candied_fruit_earnings_l3675_367565


namespace min_distance_between_curves_l3675_367519

/-- The minimum distance between points on y = x^2 + 1 and y = √(x - 1) -/
theorem min_distance_between_curves : 
  let P : ℝ × ℝ → Prop := λ p => ∃ x : ℝ, x ≥ 0 ∧ p = (x, x^2 + 1)
  let Q : ℝ × ℝ → Prop := λ q => ∃ y : ℝ, y ≥ 1 ∧ q = (y, Real.sqrt (y - 1))
  ∀ p q : ℝ × ℝ, P p → Q q → 
    ∃ p' q' : ℝ × ℝ, P p' ∧ Q q' ∧ 
      Real.sqrt ((p'.1 - q'.1)^2 + (p'.2 - q'.2)^2) = 3 * Real.sqrt 2 / 4 ∧
      ∀ p'' q'' : ℝ × ℝ, P p'' → Q q'' → 
        Real.sqrt ((p''.1 - q''.1)^2 + (p''.2 - q''.2)^2) ≥ 3 * Real.sqrt 2 / 4 :=
by
  sorry


end min_distance_between_curves_l3675_367519


namespace negative_two_times_inequality_l3675_367597

theorem negative_two_times_inequality {a b : ℝ} (h : a > b) : -2 * a < -2 * b := by
  sorry

end negative_two_times_inequality_l3675_367597


namespace sufficient_not_necessary_l3675_367594

theorem sufficient_not_necessary (x : ℝ) : 
  (x = 0 → x^2 - 2*x = 0) ∧ (∃ y : ℝ, y ≠ 0 ∧ y^2 - 2*y = 0) := by
  sorry

end sufficient_not_necessary_l3675_367594


namespace tg_ctg_roots_relation_l3675_367578

-- Define the tangent and cotangent functions
noncomputable def tg (α : Real) : Real := Real.tan α
noncomputable def ctg (α : Real) : Real := 1 / Real.tan α

-- State the theorem
theorem tg_ctg_roots_relation (p q r s α β : Real) :
  (tg α)^2 - p * (tg α) + q = 0 ∧
  (tg β)^2 - p * (tg β) + q = 0 ∧
  (ctg α)^2 - r * (ctg α) + s = 0 ∧
  (ctg β)^2 - r * (ctg β) + s = 0 →
  r * s = p / q^2 := by
sorry

end tg_ctg_roots_relation_l3675_367578


namespace quadratic_roots_sum_l3675_367554

theorem quadratic_roots_sum (m n p : ℤ) : 
  (∃ x : ℝ, 4 * x * (2 * x - 5) = -4) →
  (∃ x : ℝ, x = (m + Real.sqrt n : ℝ) / p ∧ 4 * x * (2 * x - 5) = -4) →
  (∃ x : ℝ, x = (m - Real.sqrt n : ℝ) / p ∧ 4 * x * (2 * x - 5) = -4) →
  m + n + p = 26 := by
sorry

end quadratic_roots_sum_l3675_367554


namespace fraction_equality_l3675_367524

theorem fraction_equality : (2 * 0.24) / (20 * 2.4) = 0.01 := by
  sorry

end fraction_equality_l3675_367524


namespace delores_initial_money_l3675_367559

/-- Calculates the final price of an item after applying discount and sales tax -/
def finalPrice (originalPrice discount salesTax : ℚ) : ℚ :=
  (originalPrice * (1 - discount)) * (1 + salesTax)

/-- Represents the problem of calculating Delores' initial amount of money -/
theorem delores_initial_money (computerPrice printerPrice headphonesPrice : ℚ)
  (computerDiscount computerTax printerTax headphonesTax leftoverMoney : ℚ) :
  computerPrice = 400 →
  printerPrice = 40 →
  headphonesPrice = 60 →
  computerDiscount = 0.1 →
  computerTax = 0.08 →
  printerTax = 0.05 →
  headphonesTax = 0.06 →
  leftoverMoney = 10 →
  ∃ initialMoney : ℚ,
    initialMoney = 
      finalPrice computerPrice computerDiscount computerTax +
      finalPrice printerPrice 0 printerTax +
      finalPrice headphonesPrice 0 headphonesTax +
      leftoverMoney ∧
    initialMoney = 504.4 := by
  sorry

end delores_initial_money_l3675_367559


namespace function_properties_l3675_367574

noncomputable def f (x : ℝ) : ℝ := 2^(Real.sin x)

theorem function_properties :
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < π ∧ 0 < x₂ ∧ x₂ < π ∧ f x₁ + f x₂ = 2) ∨
  (∀ x₁ x₂ : ℝ, -π/2 < x₁ ∧ x₁ < x₂ ∧ x₂ < π/2 → f x₁ < f x₂) := by
  sorry

end function_properties_l3675_367574


namespace smallest_class_size_l3675_367534

/-- Represents the number of students in a physical education class with the given arrangement. -/
def class_size (n : ℕ) : ℕ := 5 * n + 2

/-- Proves that the smallest possible class size satisfying the given conditions is 42 students. -/
theorem smallest_class_size :
  ∃ (n : ℕ), 
    (class_size n > 40) ∧ 
    (∀ m : ℕ, class_size m > 40 → m ≥ class_size n) ∧
    (class_size n = 42) :=
by
  -- Proof goes here
  sorry

end smallest_class_size_l3675_367534


namespace tangent_sum_simplification_l3675_367582

theorem tangent_sum_simplification :
  (Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + Real.tan (60 * π / 180) + Real.tan (80 * π / 180)) / Real.cos (10 * π / 180) =
  2 / (Real.sqrt 3 * Real.sin (70 * π / 180) * Real.sin (10 * π / 180) ^ 2) := by
  sorry

end tangent_sum_simplification_l3675_367582


namespace pulley_system_force_l3675_367599

/-- The force required to move a load using a pulley system -/
def required_force (m : ℝ) (g : ℝ) : ℝ := 2 * m * g

/-- Theorem: The required force to move a 2 kg load with a pulley system is 20 N -/
theorem pulley_system_force :
  let m : ℝ := 2 -- mass of the load in kg
  let g : ℝ := 10 -- acceleration due to gravity in m/s²
  required_force m g = 20 := by
  sorry

#check pulley_system_force

end pulley_system_force_l3675_367599


namespace heater_purchase_comparison_l3675_367516

/-- Represents the total cost of purchasing heaters from a store -/
structure HeaterPurchase where
  aPrice : ℝ  -- Price of A type heater
  bPrice : ℝ  -- Price of B type heater
  aShipping : ℝ  -- Shipping cost for A type heater
  bShipping : ℝ  -- Shipping cost for B type heater

/-- Calculate the total cost for a given number of A type heaters -/
def totalCost (p : HeaterPurchase) (x : ℝ) : ℝ :=
  (p.aPrice + p.aShipping) * x + (p.bPrice + p.bShipping) * (100 - x)

/-- Store A's pricing -/
def storeA : HeaterPurchase :=
  { aPrice := 100, bPrice := 200, aShipping := 10, bShipping := 10 }

/-- Store B's pricing -/
def storeB : HeaterPurchase :=
  { aPrice := 120, bPrice := 190, aShipping := 0, bShipping := 12 }

theorem heater_purchase_comparison :
  (∀ x, totalCost storeA x = -100 * x + 21000) ∧
  (∀ x, totalCost storeB x = -82 * x + 20200) ∧
  (totalCost storeA 60 < totalCost storeB 60) := by
  sorry

end heater_purchase_comparison_l3675_367516


namespace intersection_line_circle_l3675_367523

/-- Given a line ax + y - 2 = 0 intersecting a circle (x-1)² + (y-a)² = 4 at points A and B,
    where AB is the diameter of the circle, prove that a = 1. -/
theorem intersection_line_circle (a : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (a * A.1 + A.2 - 2 = 0) ∧ 
    ((A.1 - 1)^2 + (A.2 - a)^2 = 4) ∧
    (a * B.1 + B.2 - 2 = 0) ∧ 
    ((B.1 - 1)^2 + (B.2 - a)^2 = 4) ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 16) → 
  a = 1 := by
sorry

end intersection_line_circle_l3675_367523


namespace nephews_count_l3675_367572

/-- The number of nephews Alden had 10 years ago -/
def alden_nephews_10_years_ago : ℕ := 50

/-- The number of nephews Alden has now -/
def alden_nephews_now : ℕ := 2 * alden_nephews_10_years_ago

/-- The number of nephews Vihaan has now -/
def vihaan_nephews : ℕ := alden_nephews_now + 60

/-- The total number of nephews Alden and Vihaan have -/
def total_nephews : ℕ := alden_nephews_now + vihaan_nephews

theorem nephews_count : total_nephews = 260 := by
  sorry

end nephews_count_l3675_367572


namespace square_greater_than_negative_double_l3675_367589

theorem square_greater_than_negative_double {a : ℝ} (h : a < -2) : a^2 > -2*a := by
  sorry

end square_greater_than_negative_double_l3675_367589


namespace remainder_sum_l3675_367585

theorem remainder_sum (n : ℤ) (h : n % 24 = 11) : (n % 4 + n % 6 = 8) := by
  sorry

end remainder_sum_l3675_367585


namespace constant_d_value_l3675_367541

theorem constant_d_value (a d : ℝ) (h : ∀ x : ℝ, (x - 3) * (x + a) = x^2 + d*x - 18) : d = 3 := by
  sorry

end constant_d_value_l3675_367541


namespace trivia_team_score_l3675_367596

/-- Represents a trivia team member's performance -/
structure MemberPerformance where
  two_point_questions : ℕ
  four_point_questions : ℕ
  six_point_questions : ℕ

/-- Calculates the total points for a member's performance -/
def calculate_member_points (performance : MemberPerformance) : ℕ :=
  2 * performance.two_point_questions +
  4 * performance.four_point_questions +
  6 * performance.six_point_questions

/-- The trivia team's performance -/
def team_performance : List MemberPerformance := [
  ⟨3, 0, 0⟩, -- Member A
  ⟨0, 5, 1⟩, -- Member B
  ⟨0, 0, 2⟩, -- Member C
  ⟨4, 2, 0⟩, -- Member D
  ⟨1, 3, 0⟩, -- Member E
  ⟨0, 0, 5⟩, -- Member F
  ⟨1, 2, 0⟩, -- Member G
  ⟨2, 0, 3⟩, -- Member H
  ⟨0, 1, 4⟩, -- Member I
  ⟨7, 1, 0⟩  -- Member J
]

theorem trivia_team_score :
  (team_performance.map calculate_member_points).sum = 182 := by
  sorry

end trivia_team_score_l3675_367596


namespace product_eleven_cubed_sum_l3675_367598

theorem product_eleven_cubed_sum (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a * b * c = 11^3 →
  (a : ℕ) + (b : ℕ) + (c : ℕ) = 133 := by
sorry

end product_eleven_cubed_sum_l3675_367598


namespace solution_set_implies_a_value_l3675_367511

theorem solution_set_implies_a_value (a : ℝ) : 
  (∀ x : ℝ, ((a * x - 1) * (x + 2) > 0) ↔ (-3 < x ∧ x < -2)) →
  a = -1/3 := by
sorry

end solution_set_implies_a_value_l3675_367511


namespace complex_power_six_l3675_367504

/-- The imaginary unit i -/
def i : ℂ := Complex.I

/-- Statement: (1 + i)^6 = -8i -/
theorem complex_power_six : (1 + i)^6 = -8 * i := by sorry

end complex_power_six_l3675_367504


namespace average_speed_calculation_l3675_367569

theorem average_speed_calculation (distance1 distance2 time1 time2 : ℝ) 
  (h1 : distance1 = 90)
  (h2 : distance2 = 80)
  (h3 : time1 = 1)
  (h4 : time2 = 1) :
  (distance1 + distance2) / (time1 + time2) = 85 := by sorry

end average_speed_calculation_l3675_367569


namespace compound_prop_evaluation_l3675_367538

-- Define the propositions
variable (p q : Prop)

-- Define the truth values of p and q
axiom p_true : p
axiom q_false : ¬q

-- Define the compound propositions
def prop1 := p ∨ q
def prop2 := p ∧ q
def prop3 := ¬p ∧ q
def prop4 := ¬p ∨ ¬q

-- State the theorem
theorem compound_prop_evaluation :
  prop1 p q ∧ prop4 p q ∧ ¬(prop2 p q) ∧ ¬(prop3 p q) :=
sorry

end compound_prop_evaluation_l3675_367538


namespace geometric_sequence_second_term_l3675_367579

theorem geometric_sequence_second_term
  (a : ℕ+) -- first term
  (r : ℕ+) -- common ratio
  (h1 : a = 6)
  (h2 : a * r^3 = 768) :
  a * r = 24 :=
sorry

end geometric_sequence_second_term_l3675_367579


namespace arithmetic_sequence_fifth_term_l3675_367540

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 1 + a 2 = 7)
  (h_diff : a 1 - a 3 = -6) :
  a 5 = 14 := by
sorry

end arithmetic_sequence_fifth_term_l3675_367540


namespace bus_stop_timing_l3675_367557

theorem bus_stop_timing (distance : ℝ) (speed1 speed2 : ℝ) (T : ℝ) : 
  distance = 9.999999999999993 →
  speed1 = 5 →
  speed2 = 6 →
  distance / speed1 * 60 - distance / speed2 * 60 = 2 * T →
  T = 10 := by
  sorry

end bus_stop_timing_l3675_367557


namespace events_mutually_exclusive_but_not_opposed_l3675_367592

/-- Represents a card color -/
inductive CardColor
| Red
| White
| Black

/-- Represents a person -/
inductive Person
| A
| B
| C

/-- Represents the distribution of cards to people -/
def Distribution := Person → CardColor

/-- The event "A receives the red card" -/
def event_A_red (d : Distribution) : Prop := d Person.A = CardColor.Red

/-- The event "B receives the red card" -/
def event_B_red (d : Distribution) : Prop := d Person.B = CardColor.Red

/-- The set of all possible distributions -/
def all_distributions : Set Distribution :=
  {d | ∀ c : CardColor, ∃! p : Person, d p = c}

theorem events_mutually_exclusive_but_not_opposed :
  (∀ d ∈ all_distributions, ¬(event_A_red d ∧ event_B_red d)) ∧
  (∃ d ∈ all_distributions, ¬event_A_red d ∧ ¬event_B_red d) :=
sorry

end events_mutually_exclusive_but_not_opposed_l3675_367592


namespace expression_nonnegative_iff_x_in_interval_l3675_367567

/-- The expression (x-12x^2+36x^3)/(9-x^3) is nonnegative if and only if x is in the interval [0, 3). -/
theorem expression_nonnegative_iff_x_in_interval :
  ∀ x : ℝ, (x - 12 * x^2 + 36 * x^3) / (9 - x^3) ≥ 0 ↔ x ∈ Set.Icc 0 3 ∧ x ≠ 3 := by
  sorry

end expression_nonnegative_iff_x_in_interval_l3675_367567


namespace waiter_customers_l3675_367563

/-- Calculates the number of remaining customers given the initial number and the number who left. -/
def remaining_customers (initial : ℕ) (left : ℕ) : ℕ :=
  initial - left

/-- Theorem stating that for a waiter with 14 initial customers, after 5 leave, 9 remain. -/
theorem waiter_customers : remaining_customers 14 5 = 9 := by
  sorry

end waiter_customers_l3675_367563


namespace hyperbola_eccentricity_l3675_367510

/-- A hyperbola with center at the origin, focus on the x-axis, and an asymptote tangent to a specific circle has eccentricity 2. -/
theorem hyperbola_eccentricity (a b c : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 4*y + 3 = 0 → (b*x - a*y)^2 ≤ (a^2 + b^2) * ((x - 0)^2 + (y - 2)^2)) → 
  (∃ x : ℝ, x ≠ 0 ∧ (0, x) ∈ {(x, y) | (x/a)^2 - (y/b)^2 = 1}) →
  c^2 = a^2 + b^2 →
  c / a = 2 := by
  sorry

end hyperbola_eccentricity_l3675_367510


namespace sin_shift_minimum_value_l3675_367564

open Real

theorem sin_shift_minimum_value (a : ℝ) :
  (a > 0) →
  (∀ x, sin (2 * x - π / 3) = sin (2 * (x - a))) →
  a = π / 6 :=
by sorry

end sin_shift_minimum_value_l3675_367564


namespace positive_difference_is_zero_l3675_367584

/-- The quadratic equation from the problem -/
def quadratic_equation (x : ℂ) : Prop :=
  x^2 + 5*x + 20 = 2*x + 16

/-- The solutions of the quadratic equation -/
def solutions : Set ℂ :=
  {x : ℂ | quadratic_equation x}

/-- The positive difference between the solutions is 0 -/
theorem positive_difference_is_zero :
  ∃ (x y : ℂ), x ∈ solutions ∧ y ∈ solutions ∧ x ≠ y ∧ |x.re - y.re| = 0 :=
sorry

end positive_difference_is_zero_l3675_367584


namespace better_fit_example_l3675_367550

/-- Represents a regression model with its RSS (Residual Sum of Squares) -/
structure RegressionModel where
  rss : ℝ

/-- Determines if one model has a better fit than another based on RSS -/
def better_fit (model1 model2 : RegressionModel) : Prop :=
  model1.rss < model2.rss

theorem better_fit_example :
  let model1 : RegressionModel := ⟨168⟩
  let model2 : RegressionModel := ⟨197⟩
  better_fit model1 model2 := by
  sorry

end better_fit_example_l3675_367550


namespace ring_toss_daily_income_l3675_367501

theorem ring_toss_daily_income (total_income : ℕ) (num_days : ℕ) (daily_income : ℕ) : 
  total_income = 7560 → 
  num_days = 12 → 
  total_income = daily_income * num_days →
  daily_income = 630 := by
sorry

end ring_toss_daily_income_l3675_367501


namespace data_center_connections_l3675_367588

theorem data_center_connections (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 4) :
  (n * k) / 2 = 60 := by
  sorry

end data_center_connections_l3675_367588


namespace cos_180_degrees_l3675_367544

theorem cos_180_degrees : Real.cos (π) = -1 := by sorry

end cos_180_degrees_l3675_367544


namespace solution_set_when_a_is_one_range_of_a_given_inequality_l3675_367506

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x - 2*a|

-- Part I
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x > 2} = {x : ℝ | x < 1/2 ∨ x > 5/2} := by sorry

-- Part II
theorem range_of_a_given_inequality (a : ℝ) :
  (∀ x : ℝ, f a x ≥ a^2 - 3*a - 3) → a ∈ Set.Icc (-1) (2 + Real.sqrt 7) := by sorry

end solution_set_when_a_is_one_range_of_a_given_inequality_l3675_367506


namespace complex_number_problem_l3675_367590

theorem complex_number_problem (z : ℂ) 
  (h1 : ∃ (r : ℝ), z + 2*I = r)
  (h2 : ∃ (s : ℝ), z / (2 - I) = s) :
  z = 4 - 2*I ∧ 
  ∀ (a : ℝ), (z - a*I)^2 ∈ {w : ℂ | 0 < w.re ∧ 0 < w.im} ↔ -6 < a ∧ a < -2 :=
sorry

end complex_number_problem_l3675_367590


namespace rectangle_length_equal_square_side_l3675_367532

/-- The length of a rectangle with width 4 cm and area equal to a square with side length 4 cm -/
theorem rectangle_length_equal_square_side : ∀ (length : ℝ), 
  (4 : ℝ) * length = (4 : ℝ) * (4 : ℝ) → length = (4 : ℝ) := by
  sorry

end rectangle_length_equal_square_side_l3675_367532


namespace four_spheres_block_light_l3675_367595

-- Define a point in 3D space
def Point := ℝ × ℝ × ℝ

-- Define a sphere in 3D space
structure Sphere where
  center : Point
  radius : ℝ
  radius_pos : radius > 0

-- Define the property of a sphere being opaque
def isOpaque (s : Sphere) : Prop := sorry

-- Define the property of two spheres being non-intersecting
def nonIntersecting (s1 s2 : Sphere) : Prop := sorry

-- Define the property of a set of spheres blocking light from a point source
def blocksLight (source : Point) (spheres : List Sphere) : Prop := sorry

-- The main theorem
theorem four_spheres_block_light :
  ∃ (s1 s2 s3 s4 : Sphere) (source : Point),
    isOpaque s1 ∧ isOpaque s2 ∧ isOpaque s3 ∧ isOpaque s4 ∧
    nonIntersecting s1 s2 ∧ nonIntersecting s1 s3 ∧ nonIntersecting s1 s4 ∧
    nonIntersecting s2 s3 ∧ nonIntersecting s2 s4 ∧ nonIntersecting s3 s4 ∧
    blocksLight source [s1, s2, s3, s4] := by
  sorry

end four_spheres_block_light_l3675_367595


namespace rhombus_diagonal_l3675_367587

/-- Given a rhombus with area 80 and one diagonal of length 16, 
    prove that the other diagonal has length 10. -/
theorem rhombus_diagonal (area : ℝ) (d1 : ℝ) (d2 : ℝ) 
  (h_area : area = 80) 
  (h_d1 : d1 = 16) 
  (h_rhombus : area = d1 * d2 / 2) : 
  d2 = 10 := by
  sorry

end rhombus_diagonal_l3675_367587


namespace bill_amount_calculation_l3675_367560

/-- Given a true discount and a banker's discount, calculate the amount of the bill. -/
def billAmount (trueDiscount : ℚ) (bankersDiscount : ℚ) : ℚ :=
  trueDiscount + trueDiscount

/-- Theorem: Given a true discount of 360 and a banker's discount of 428.21, the amount of the bill is 720. -/
theorem bill_amount_calculation :
  let trueDiscount : ℚ := 360
  let bankersDiscount : ℚ := 428.21
  billAmount trueDiscount bankersDiscount = 720 := by
  sorry

#eval billAmount 360 428.21

end bill_amount_calculation_l3675_367560


namespace no_integer_solutions_l3675_367525

theorem no_integer_solutions : 
  ¬ ∃ (m n : ℤ), m^3 + n^4 + 130*m*n = 35^3 ∧ m*n ≥ 0 :=
by sorry

end no_integer_solutions_l3675_367525


namespace mass_BaSO4_produced_l3675_367512

-- Define the molar masses of elements (in g/mol)
def molar_mass_Ba : ℝ := 137.327
def molar_mass_S : ℝ := 32.065
def molar_mass_O : ℝ := 15.999

-- Define the molar mass of Barium sulfate
def molar_mass_BaSO4 : ℝ := molar_mass_Ba + molar_mass_S + 4 * molar_mass_O

-- Define the number of moles of Barium bromide
def moles_BaBr2 : ℝ := 4

-- Theorem statement
theorem mass_BaSO4_produced (excess_Na2SO4 : Prop) (double_displacement : Prop) :
  moles_BaBr2 * molar_mass_BaSO4 = 933.552 := by
  sorry


end mass_BaSO4_produced_l3675_367512


namespace customer_difference_l3675_367536

theorem customer_difference (initial : Nat) (remaining : Nat) : 
  initial = 11 → remaining = 3 → (initial - remaining) - remaining = 5 := by
  sorry

end customer_difference_l3675_367536


namespace flag_arrangement_congruence_l3675_367571

/-- Number of blue flags -/
def blue_flags : ℕ := 10

/-- Number of green flags -/
def green_flags : ℕ := 10

/-- Total number of flags -/
def total_flags : ℕ := blue_flags + green_flags

/-- Number of flagpoles -/
def flagpoles : ℕ := 2

/-- 
  N is the number of distinguishable arrangements of flags on two distinguishable flagpoles,
  where each flagpole has at least one green flag and no two green flags on either pole are adjacent
-/
def N : ℕ := sorry

theorem flag_arrangement_congruence : N ≡ 77 [MOD 1000] := by sorry

end flag_arrangement_congruence_l3675_367571


namespace complex_number_in_fourth_quadrant_l3675_367581

theorem complex_number_in_fourth_quadrant (z : ℂ) (h : (1 + Complex.I) * z = 2 - Complex.I) :
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 :=
sorry

end complex_number_in_fourth_quadrant_l3675_367581


namespace sarah_earnings_l3675_367503

/-- Sarah's earnings for the week given her work hours and pay rates -/
theorem sarah_earnings : 
  let weekday_hours := 1.75 + 65/60 + 2.75 + 45/60
  let weekend_hours := 2
  let weekday_rate := 4
  let weekend_rate := 6
  (weekday_hours * weekday_rate + weekend_hours * weekend_rate : ℝ) = 37.33 := by
  sorry

end sarah_earnings_l3675_367503


namespace min_button_presses_correct_l3675_367548

/-- Represents the time difference in minutes between the correct time and the displayed time -/
def time_difference : ℤ := 13

/-- Represents the increase in minutes when the first button is pressed -/
def button1_adjustment : ℤ := 9

/-- Represents the decrease in minutes when the second button is pressed -/
def button2_adjustment : ℤ := 20

/-- Represents the equation for adjusting the clock -/
def clock_adjustment (a b : ℤ) : Prop :=
  button1_adjustment * a - button2_adjustment * b = time_difference

/-- The minimum number of button presses required -/
def min_button_presses : ℕ := 24

/-- Theorem stating that the minimum number of button presses to correctly set the clock is 24 -/
theorem min_button_presses_correct :
  ∃ (a b : ℤ), clock_adjustment a b ∧ a ≥ 0 ∧ b ≥ 0 ∧ a + b = min_button_presses ∧
  (∀ (c d : ℤ), clock_adjustment c d → c ≥ 0 → d ≥ 0 → c + d ≥ min_button_presses) :=
by sorry

end min_button_presses_correct_l3675_367548


namespace sheet_width_calculation_l3675_367576

/-- The width of a rectangular sheet of paper with specific properties -/
def sheet_width : ℝ := sorry

theorem sheet_width_calculation : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  abs (sheet_width - 6.6) < ε ∧
  sheet_width * 13 * 2 = 6.5 * 11 + 100 := by sorry

end sheet_width_calculation_l3675_367576


namespace pencil_price_l3675_367570

theorem pencil_price 
  (total_items : ℕ) 
  (pen_count : ℕ) 
  (pencil_count : ℕ) 
  (total_cost : ℚ) 
  (avg_pen_price : ℚ) 
  (h1 : total_items = pen_count + pencil_count)
  (h2 : total_items = 105)
  (h3 : pen_count = 30)
  (h4 : pencil_count = 75)
  (h5 : total_cost = 750)
  (h6 : avg_pen_price = 20) :
  (total_cost - pen_count * avg_pen_price) / pencil_count = 2 := by
sorry


end pencil_price_l3675_367570


namespace ellipse_equation_l3675_367529

theorem ellipse_equation (major_axis : ℝ) (eccentricity : ℝ) :
  major_axis = 8 →
  eccentricity = 3/4 →
  (∃ x y : ℝ, x^2/16 + y^2/7 = 1) ∨ (∃ x y : ℝ, x^2/7 + y^2/16 = 1) :=
by sorry

end ellipse_equation_l3675_367529


namespace bank_savings_exceed_two_dollars_l3675_367556

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem bank_savings_exceed_two_dollars :
  let a : ℚ := 1/100  -- 1 cent in dollars
  let r : ℚ := 2      -- doubling each day
  (geometric_sum a r 8 > 2) ∧ (geometric_sum a r 7 ≤ 2) :=
by sorry

end bank_savings_exceed_two_dollars_l3675_367556


namespace chemistry_class_size_l3675_367562

/-- Represents the number of students in a school with chemistry and biology classes -/
structure School where
  total : ℕ
  chemistry : ℕ
  biology : ℕ
  both : ℕ

/-- The conditions of the problem -/
def school_conditions (s : School) : Prop :=
  s.total = 43 ∧
  s.both = 5 ∧
  s.chemistry = 3 * s.biology ∧
  s.total = (s.chemistry - s.both) + (s.biology - s.both) + s.both

/-- The theorem to be proved -/
theorem chemistry_class_size (s : School) :
  school_conditions s → s.chemistry = 36 := by
  sorry

end chemistry_class_size_l3675_367562


namespace classroom_ratio_l3675_367517

theorem classroom_ratio (total_students : ℕ) (num_boys : ℕ) (h1 : total_students > 0) (h2 : num_boys ≤ total_students) :
  let prob_boy := num_boys / total_students
  let prob_girl := (total_students - num_boys) / total_students
  (prob_boy / prob_girl = 3 / 4) → (num_boys / total_students = 3 / 7) := by
  sorry

end classroom_ratio_l3675_367517


namespace inequality_proof_l3675_367507

theorem inequality_proof (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  a / Real.sqrt (b^2 + 1) + b / Real.sqrt (a^2 + 1) ≥ (a + b) / Real.sqrt (a * b + 1) ∧
  (a / Real.sqrt (b^2 + 1) + b / Real.sqrt (a^2 + 1) = (a + b) / Real.sqrt (a * b + 1) ↔ a = b) :=
by sorry

end inequality_proof_l3675_367507


namespace greatest_integer_inequality_l3675_367526

theorem greatest_integer_inequality :
  ∀ x : ℤ, (3 * x + 2 < 7 - 2 * x) → x ≤ 0 :=
by sorry

end greatest_integer_inequality_l3675_367526


namespace quadratic_roots_properties_l3675_367508

theorem quadratic_roots_properties (a b c x₁ x₂ : ℝ) (ha : a ≠ 0)
  (hx₁ : a * x₁^2 + b * x₁ + c = 0) (hx₂ : a * x₂^2 + b * x₂ + c = 0) :
  x₁^2 + x₂^2 = (b^2 - 2*a*c) / a^2 ∧ x₁^3 + x₂^3 = (3*a*b*c - b^3) / a^3 := by
  sorry

end quadratic_roots_properties_l3675_367508


namespace triangle_inequality_check_l3675_367535

/-- A function that checks if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The theorem stating that among the given sets, only {17, 17, 25} can form a triangle -/
theorem triangle_inequality_check : 
  ¬(can_form_triangle 3 4 8) ∧ 
  ¬(can_form_triangle 5 6 11) ∧ 
  ¬(can_form_triangle 6 8 16) ∧ 
  can_form_triangle 17 17 25 :=
sorry

end triangle_inequality_check_l3675_367535


namespace ryan_tokens_l3675_367591

def arcade_tokens (initial_tokens : ℕ) : ℕ :=
  let pacman_tokens := (2 * initial_tokens) / 3
  let remaining_after_pacman := initial_tokens - pacman_tokens
  let candy_crush_tokens := remaining_after_pacman / 2
  let remaining_after_candy_crush := remaining_after_pacman - candy_crush_tokens
  let skeball_tokens := min remaining_after_candy_crush 7
  let parents_bought := 10 * skeball_tokens
  remaining_after_candy_crush - skeball_tokens + parents_bought

theorem ryan_tokens : arcade_tokens 36 = 66 := by
  sorry

end ryan_tokens_l3675_367591


namespace horse_race_equation_l3675_367542

/-- Represents the scenario of two horses racing --/
structure HorseRace where
  fast_speed : ℕ  -- Speed of the fast horse in miles per day
  slow_speed : ℕ  -- Speed of the slow horse in miles per day
  head_start : ℕ  -- Number of days the slow horse starts earlier

/-- The equation for when the fast horse catches up to the slow horse --/
def catch_up_equation (race : HorseRace) (x : ℕ) : Prop :=
  race.slow_speed * (x + race.head_start) = race.fast_speed * x

/-- The theorem stating the correct equation for the given scenario --/
theorem horse_race_equation :
  let race := HorseRace.mk 240 150 12
  ∀ x, catch_up_equation race x ↔ 150 * (x + 12) = 240 * x :=
by sorry

end horse_race_equation_l3675_367542


namespace fraction_equality_l3675_367552

theorem fraction_equality (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) :
  (1/x - 1/y) / (1/x + 1/y) = 2023 → (x + y) / (x - y) = -1 := by
  sorry

end fraction_equality_l3675_367552


namespace largest_whole_number_nine_times_less_than_150_l3675_367575

theorem largest_whole_number_nine_times_less_than_150 :
  ∃ (x : ℕ), x = 16 ∧ (∀ y : ℕ, 9 * y < 150 → y ≤ x) := by
  sorry

end largest_whole_number_nine_times_less_than_150_l3675_367575


namespace combined_average_age_l3675_367558

theorem combined_average_age (x_count y_count : ℕ) (x_avg y_avg : ℝ) 
  (hx : x_count = 8) (hy : y_count = 5) 
  (hxa : x_avg = 30) (hya : y_avg = 45) : 
  (x_count * x_avg + y_count * y_avg) / (x_count + y_count) = 36 := by
  sorry

end combined_average_age_l3675_367558


namespace equation_solutions_l3675_367513

theorem equation_solutions :
  (∀ x : ℝ, (x - 1)^2 - 25 = 0 ↔ x = 6 ∨ x = -4) ∧
  (∀ x : ℝ, 3*x*(x - 2) = x - 2 ↔ x = 2 ∨ x = 1/3) := by
  sorry

end equation_solutions_l3675_367513


namespace arithmetic_sequence_15th_term_l3675_367566

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_15th_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_7 : a 7 = 8)
  (h_23 : a 23 = 22) :
  a 15 = 15 :=
sorry

end arithmetic_sequence_15th_term_l3675_367566


namespace product_and_sum_of_factors_l3675_367539

theorem product_and_sum_of_factors : ∃ a b : ℕ, 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 8670 ∧ 
  a + b = 136 := by
sorry

end product_and_sum_of_factors_l3675_367539


namespace keith_bought_cards_l3675_367502

/-- The number of baseball cards Keith bought -/
def cards_bought : ℕ := sorry

/-- Fred's initial number of baseball cards -/
def initial_cards : ℕ := 40

/-- Fred's current number of baseball cards -/
def current_cards : ℕ := 18

/-- Theorem: The number of cards Keith bought is equal to the difference
    between Fred's initial and current number of cards -/
theorem keith_bought_cards : 
  cards_bought = initial_cards - current_cards := by sorry

end keith_bought_cards_l3675_367502


namespace winnie_balloons_l3675_367551

theorem winnie_balloons (red white green chartreuse : ℕ) 
  (h1 : red = 17) 
  (h2 : white = 33) 
  (h3 : green = 65) 
  (h4 : chartreuse = 83) 
  (friends : ℕ) 
  (h5 : friends = 8) : 
  (red + white + green + chartreuse) % friends = 6 := by
sorry

end winnie_balloons_l3675_367551


namespace a_less_than_one_l3675_367580

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- State the conditions
axiom deriv_f : ∀ x, HasDerivAt f (f' x) x
axiom symm_cond : ∀ x, f x + f (-x) = x^2
axiom deriv_gt : ∀ x ≥ 0, f' x > x
axiom ineq_cond : ∀ a, f (2 - a) + 2*a > f a + 2

-- State the theorem
theorem a_less_than_one (a : ℝ) : a < 1 := by
  sorry

end a_less_than_one_l3675_367580


namespace annie_cookies_l3675_367549

theorem annie_cookies (x : ℝ) 
  (h1 : x + 2*x + 2.8*x = 29) : x = 5 := by
  sorry

end annie_cookies_l3675_367549


namespace average_marks_chem_math_l3675_367561

/-- Given that the total marks in physics, chemistry, and mathematics is 150 more than
    the marks in physics, prove that the average mark in chemistry and mathematics is 75. -/
theorem average_marks_chem_math (P C M : ℝ) 
  (h : P + C + M = P + 150) : (C + M) / 2 = 75 := by
  sorry

end average_marks_chem_math_l3675_367561


namespace product_of_three_numbers_l3675_367533

theorem product_of_three_numbers (p q r m : ℝ) 
  (sum_eq : p + q + r = 180)
  (p_eq : 8 * p = m)
  (q_eq : q - 10 = m)
  (r_eq : r + 10 = m)
  (p_smallest : p < q ∧ p < r) :
  p * q * r = 90000 := by
  sorry

end product_of_three_numbers_l3675_367533


namespace existence_of_divisible_difference_l3675_367537

theorem existence_of_divisible_difference (x : Fin 2022 → ℤ) :
  ∃ i j : Fin 2022, i ≠ j ∧ (2021 : ℤ) ∣ (x j - x i) := by
  sorry

end existence_of_divisible_difference_l3675_367537


namespace floor_sqrt_116_l3675_367545

theorem floor_sqrt_116 : ⌊Real.sqrt 116⌋ = 10 := by
  sorry

end floor_sqrt_116_l3675_367545


namespace integer_solutions_x4_minus_2y2_eq_1_l3675_367555

theorem integer_solutions_x4_minus_2y2_eq_1 :
  ∀ x y : ℤ, x^4 - 2*y^2 = 1 ↔ (x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = 0) := by
  sorry

end integer_solutions_x4_minus_2y2_eq_1_l3675_367555


namespace square_of_real_not_always_positive_l3675_367547

theorem square_of_real_not_always_positive : 
  ¬ (∀ a : ℝ, a^2 > 0) :=
by sorry

end square_of_real_not_always_positive_l3675_367547


namespace total_battery_after_exam_l3675_367583

def calculator_remaining_battery (full_capacity : ℝ) (used_proportion : ℝ) (exam_duration : ℝ) : ℝ :=
  full_capacity * (1 - used_proportion) - exam_duration

def total_remaining_battery (calc1_capacity : ℝ) (calc1_used : ℝ) 
                            (calc2_capacity : ℝ) (calc2_used : ℝ)
                            (calc3_capacity : ℝ) (calc3_used : ℝ)
                            (exam_duration : ℝ) : ℝ :=
  calculator_remaining_battery calc1_capacity calc1_used exam_duration +
  calculator_remaining_battery calc2_capacity calc2_used exam_duration +
  calculator_remaining_battery calc3_capacity calc3_used exam_duration

theorem total_battery_after_exam :
  total_remaining_battery 60 (3/4) 80 (1/2) 120 (2/3) 2 = 89 := by
  sorry

end total_battery_after_exam_l3675_367583


namespace overlap_area_theorem_l3675_367543

/-- Represents a square with a given side length -/
structure Square where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

/-- The area of a square -/
def Square.area (s : Square) : ℝ := s.sideLength * s.sideLength

/-- The overlap configuration of two squares -/
structure SquareOverlap where
  largeSquare : Square
  smallSquare : Square
  smallSquareTouchesCenter : smallSquare.sideLength = largeSquare.sideLength / 2

/-- The area covered only by the larger square in the overlap configuration -/
def SquareOverlap.areaOnlyLarger (so : SquareOverlap) : ℝ :=
  so.largeSquare.area - so.smallSquare.area

/-- The main theorem -/
theorem overlap_area_theorem (so : SquareOverlap) 
    (h1 : so.largeSquare.sideLength = 8) 
    (h2 : so.smallSquare.sideLength = 4) : 
    so.areaOnlyLarger = 48 := by
  sorry


end overlap_area_theorem_l3675_367543


namespace larger_cuboid_height_l3675_367530

/-- The height of a larger cuboid given its dimensions and the number and dimensions of smaller cuboids it contains. -/
theorem larger_cuboid_height (length width : ℝ) (num_small_cuboids : ℕ) 
  (small_length small_width small_height : ℝ) : 
  length = 12 →
  width = 14 →
  num_small_cuboids = 56 →
  small_length = 5 →
  small_width = 3 →
  small_height = 2 →
  (length * width * (num_small_cuboids * small_length * small_width * small_height) / (length * width)) = 10 := by
  sorry

end larger_cuboid_height_l3675_367530


namespace boris_candy_distribution_l3675_367527

/-- Given the initial conditions of Boris's candy distribution, 
    prove that the final number of pieces in each bowl is 83. -/
theorem boris_candy_distribution (initial_candy : ℕ) 
  (daughter_eats : ℕ) (set_aside : ℕ) (num_bowls : ℕ) (take_away : ℕ) :
  initial_candy = 300 →
  daughter_eats = 25 →
  set_aside = 10 →
  num_bowls = 6 →
  take_away = 5 →
  let remaining := initial_candy - daughter_eats - set_aside
  let per_bowl := remaining / num_bowls
  let doubled := per_bowl * 2
  doubled - take_away = 83 := by
  sorry

end boris_candy_distribution_l3675_367527


namespace multiply_by_213_equals_3408_l3675_367500

theorem multiply_by_213_equals_3408 (x : ℝ) : 213 * x = 3408 → x = 16 := by
  sorry

end multiply_by_213_equals_3408_l3675_367500


namespace max_value_of_fraction_difference_l3675_367505

theorem max_value_of_fraction_difference (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h : 4 * a - b ≥ 2) : 
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 4 * x - y ≥ 2 ∧ 1 / x - 1 / y ≤ 1 / 2) ∧ 
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 4 * x - y ≥ 2 ∧ 1 / x - 1 / y = 1 / 2) :=
sorry

end max_value_of_fraction_difference_l3675_367505


namespace speed_is_pi_over_three_l3675_367553

/-- Represents a rectangular track with looped ends -/
structure Track where
  width : ℝ
  straightLength : ℝ

/-- Calculates the speed of a person walking around the track -/
def calculateSpeed (track : Track) (timeDifference : ℝ) : ℝ :=
  sorry

/-- Theorem stating that given the specific track conditions, the calculated speed is π/3 -/
theorem speed_is_pi_over_three (track : Track) (h1 : track.width = 8)
    (h2 : track.straightLength = 100) (timeDifference : ℝ) (h3 : timeDifference = 48) :
    calculateSpeed track timeDifference = π / 3 := by
  sorry

end speed_is_pi_over_three_l3675_367553


namespace largest_prime_factor_of_expression_l3675_367577

theorem largest_prime_factor_of_expression :
  ∃ (p : ℕ), 
    Nat.Prime p ∧ 
    p ∣ (20^4 + 15^4 - 10^5) ∧
    ∀ (q : ℕ), Nat.Prime q → q ∣ (20^4 + 15^4 - 10^5) → q ≤ p ∧
    p = 59 := by
  sorry

end largest_prime_factor_of_expression_l3675_367577


namespace fold_line_length_squared_l3675_367531

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop :=
  let d_AB := ((t.A.x - t.B.x)^2 + (t.A.y - t.B.y)^2)^(1/2)
  let d_BC := ((t.B.x - t.C.x)^2 + (t.B.y - t.C.y)^2)^(1/2)
  let d_CA := ((t.C.x - t.A.x)^2 + (t.C.y - t.A.y)^2)^(1/2)
  d_AB = d_BC ∧ d_BC = d_CA

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ :=
  ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)^(1/2)

/-- Theorem: The square of the length of the fold line in the given triangle problem -/
theorem fold_line_length_squared (t : Triangle) (P : Point) :
  isEquilateral t →
  distance t.A t.B = 15 →
  distance t.B P = 11 →
  P.x = t.B.x + 11 * (t.C.x - t.B.x) / 15 →
  P.y = t.B.y + 11 * (t.C.y - t.B.y) / 15 →
  ∃ Q : Point,
    Q.x = t.A.x + (P.x - t.A.x) / 2 ∧
    Q.y = t.A.y + (P.y - t.A.y) / 2 ∧
    (distance Q P)^2 = 1043281 / 31109 :=
sorry

end fold_line_length_squared_l3675_367531


namespace inequality_proof_l3675_367546

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a - d > b - c := by
  sorry

end inequality_proof_l3675_367546


namespace expand_expression_l3675_367586

theorem expand_expression (x : ℝ) : 24 * (3 * x + 4 - 2) = 72 * x + 48 := by
  sorry

end expand_expression_l3675_367586


namespace largest_value_l3675_367515

theorem largest_value : 
  ∀ (a b c d : ℤ), a = 2^3 ∧ b = -3^2 ∧ c = (-3)^2 ∧ d = (-2)^3 →
  (c ≥ a ∧ c ≥ b ∧ c ≥ d) := by
  sorry

end largest_value_l3675_367515


namespace line_up_five_people_two_youngest_not_first_l3675_367528

/-- The number of ways to arrange 5 people in a line with restrictions -/
def lineUpWays (n : ℕ) (y : ℕ) (f : ℕ) : ℕ :=
  (n - y) * (n - 1) * (n - 2) * (n - 3) * (n - 4)

/-- Theorem: There are 72 ways for 5 people to line up when 2 youngest can't be first -/
theorem line_up_five_people_two_youngest_not_first :
  lineUpWays 5 2 1 = 72 := by sorry

end line_up_five_people_two_youngest_not_first_l3675_367528


namespace power_multiplication_l3675_367518

theorem power_multiplication (a b : ℕ) : (10 : ℕ) ^ 655 * (10 : ℕ) ^ 652 = (10 : ℕ) ^ (655 + 652) := by
  sorry

end power_multiplication_l3675_367518
