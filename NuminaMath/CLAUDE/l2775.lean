import Mathlib

namespace NUMINAMATH_CALUDE_jellybean_count_l2775_277568

theorem jellybean_count (remaining_ratio : ℝ) (days : ℕ) (final_count : ℕ) 
  (h1 : remaining_ratio = 0.75)
  (h2 : days = 3)
  (h3 : final_count = 27) :
  ∃ (original_count : ℕ), 
    (remaining_ratio ^ days) * (original_count : ℝ) = final_count ∧ 
    original_count = 64 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_count_l2775_277568


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l2775_277512

theorem smallest_solution_of_equation (x : ℝ) :
  x = (5 - Real.sqrt 241) / 6 →
  (3 * x / (x - 3) + (3 * x^2 - 36) / x = 14) ∧
  ∀ y : ℝ, (3 * y / (y - 3) + (3 * y^2 - 36) / y = 14) → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l2775_277512


namespace NUMINAMATH_CALUDE_a_income_l2775_277580

def income_ratio : ℚ := 5 / 4
def expenditure_ratio : ℚ := 3 / 2
def savings : ℕ := 1600

theorem a_income (a_income b_income a_expenditure b_expenditure : ℚ) 
  (h1 : a_income / b_income = income_ratio)
  (h2 : a_expenditure / b_expenditure = expenditure_ratio)
  (h3 : a_income - a_expenditure = savings)
  (h4 : b_income - b_expenditure = savings) :
  a_income = 4000 := by
  sorry

end NUMINAMATH_CALUDE_a_income_l2775_277580


namespace NUMINAMATH_CALUDE_solve_bank_problem_l2775_277513

def bank_problem (initial_balance : ℚ) : Prop :=
  let tripled_balance := initial_balance * 3
  let balance_after_withdrawal := tripled_balance - 250
  balance_after_withdrawal = 950

theorem solve_bank_problem :
  ∃ (initial_balance : ℚ), bank_problem initial_balance ∧ initial_balance = 400 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_bank_problem_l2775_277513


namespace NUMINAMATH_CALUDE_target_hit_probability_l2775_277588

def probability_hit : ℚ := 1 / 2

def total_shots : ℕ := 6

def successful_hits : ℕ := 3

def consecutive_hits : ℕ := 2

theorem target_hit_probability :
  (probability_hit ^ successful_hits) *
  ((1 - probability_hit) ^ (total_shots - successful_hits)) *
  (3 * (Nat.factorial 2 * Nat.factorial 4 / (Nat.factorial 2 * Nat.factorial 2))) =
  3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_target_hit_probability_l2775_277588


namespace NUMINAMATH_CALUDE_inverse_proportionality_l2775_277523

/-- Proves that given α is inversely proportional to β, and α = 4 when β = 12, then α = -16 when β = -3 -/
theorem inverse_proportionality (α β : ℝ → ℝ) (k : ℝ) : 
  (∀ x, α x * β x = k) →  -- α is inversely proportional to β
  (α 12 = 4) →            -- α = 4 when β = 12
  (β 12 = 12) →           -- ensuring β 12 is indeed 12
  (β (-3) = -3) →         -- ensuring β (-3) is indeed -3
  (α (-3) = -16) :=       -- α = -16 when β = -3
by
  sorry


end NUMINAMATH_CALUDE_inverse_proportionality_l2775_277523


namespace NUMINAMATH_CALUDE_larger_number_value_l2775_277587

theorem larger_number_value (x y : ℝ) (hx : x = 48) (hdiff : y - x = (1/3) * y) : y = 72 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_value_l2775_277587


namespace NUMINAMATH_CALUDE_encryption_of_2568_l2775_277556

def encrypt_digit (d : Nat) : Nat :=
  (d^3 + 1) % 10

def encrypt_number (n : List Nat) : List Nat :=
  n.map encrypt_digit

theorem encryption_of_2568 :
  encrypt_number [2, 5, 6, 8] = [9, 6, 7, 3] := by
  sorry

end NUMINAMATH_CALUDE_encryption_of_2568_l2775_277556


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_thirteen_sixths_l2775_277565

theorem sqrt_sum_equals_thirteen_sixths : 
  Real.sqrt (9 / 4) + Real.sqrt (4 / 9) = 13 / 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_thirteen_sixths_l2775_277565


namespace NUMINAMATH_CALUDE_max_distance_ellipse_to_line_l2775_277530

/-- The maximum distance from a point on the ellipse x^2/4 + y^2 = 1 to the line x + 2y = 0 -/
theorem max_distance_ellipse_to_line :
  let ellipse := {P : ℝ × ℝ | P.1^2/4 + P.2^2 = 1}
  let line := {P : ℝ × ℝ | P.1 + 2*P.2 = 0}
  ∃ (d : ℝ), d = 2*Real.sqrt 10/5 ∧
    ∀ P ∈ ellipse, ∀ Q ∈ line,
      Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≤ d ∧
      ∃ P' ∈ ellipse, ∃ Q' ∈ line,
        Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2) = d :=
by sorry

end NUMINAMATH_CALUDE_max_distance_ellipse_to_line_l2775_277530


namespace NUMINAMATH_CALUDE_cos_pi_third_minus_alpha_l2775_277561

theorem cos_pi_third_minus_alpha (α : ℝ) (h : Real.sin (π / 6 + α) = 3 / 5) :
  Real.cos (π / 3 - α) = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_cos_pi_third_minus_alpha_l2775_277561


namespace NUMINAMATH_CALUDE_rebecca_groups_l2775_277562

/-- The number of eggs Rebecca has -/
def total_eggs : ℕ := 20

/-- The number of marbles Rebecca has -/
def total_marbles : ℕ := 6

/-- The number of eggs in each group -/
def eggs_per_group : ℕ := 5

/-- The number of marbles in each group -/
def marbles_per_group : ℕ := 2

/-- The maximum number of groups that can be created -/
def max_groups : ℕ := min (total_eggs / eggs_per_group) (total_marbles / marbles_per_group)

theorem rebecca_groups : max_groups = 3 := by
  sorry

end NUMINAMATH_CALUDE_rebecca_groups_l2775_277562


namespace NUMINAMATH_CALUDE_mass_of_man_is_60kg_l2775_277547

/-- The mass of a man who causes a boat to sink by a certain depth --/
def mass_of_man (boat_length boat_breadth sink_depth water_density : Real) : Real :=
  boat_length * boat_breadth * sink_depth * water_density

/-- Theorem stating that the mass of the man is 60 kg given the specific conditions --/
theorem mass_of_man_is_60kg : 
  mass_of_man 3 2 0.01 1000 = 60 := by
  sorry

#eval mass_of_man 3 2 0.01 1000

end NUMINAMATH_CALUDE_mass_of_man_is_60kg_l2775_277547


namespace NUMINAMATH_CALUDE_system_solution_range_l2775_277566

theorem system_solution_range (x y m : ℝ) : 
  (3 * x + y = 1 + 3 * m) →
  (x + 3 * y = 1 - m) →
  (x + y > 0) →
  (m > -1) := by
sorry

end NUMINAMATH_CALUDE_system_solution_range_l2775_277566


namespace NUMINAMATH_CALUDE_tens_digit_13_2023_l2775_277552

theorem tens_digit_13_2023 : ∃ n : ℕ, 13^2023 ≡ 90 + n [ZMOD 100] := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_13_2023_l2775_277552


namespace NUMINAMATH_CALUDE_juniper_bones_l2775_277574

/-- Calculates the final number of bones Juniper has after transactions --/
def final_bones (initial : ℕ) : ℕ :=
  let additional := (initial * 50) / 100
  let total := initial + additional
  let stolen := (total * 25) / 100
  total - stolen

/-- Theorem stating that Juniper ends up with 5 bones --/
theorem juniper_bones : final_bones 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_juniper_bones_l2775_277574


namespace NUMINAMATH_CALUDE_fruit_purchase_total_l2775_277506

/-- The total amount paid for a fruit purchase given the quantity and rate per kg for two types of fruits -/
def total_amount_paid (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem stating that the total amount paid for 8 kg of grapes at 70 per kg and 9 kg of mangoes at 45 per kg is 965 -/
theorem fruit_purchase_total :
  total_amount_paid 8 70 9 45 = 965 := by
  sorry

end NUMINAMATH_CALUDE_fruit_purchase_total_l2775_277506


namespace NUMINAMATH_CALUDE_root_in_interval_l2775_277594

def f (x : ℝ) := 2*x + x - 2

theorem root_in_interval :
  Continuous f ∧ f 0 < 0 ∧ f 1 > 0 → ∃ x ∈ Set.Ioo 0 1, f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l2775_277594


namespace NUMINAMATH_CALUDE_emily_necklaces_l2775_277576

/-- Given that Emily used a total of 18 beads and each necklace requires 3 beads,
    prove that the number of necklaces she made is 6. -/
theorem emily_necklaces :
  let total_beads : ℕ := 18
  let beads_per_necklace : ℕ := 3
  let necklaces_made : ℕ := total_beads / beads_per_necklace
  necklaces_made = 6 := by
  sorry

end NUMINAMATH_CALUDE_emily_necklaces_l2775_277576


namespace NUMINAMATH_CALUDE_even_function_implies_a_eq_neg_one_l2775_277532

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function y = (x+1)(x-a) -/
def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * (x - a)

/-- If f(a) is an even function, then a = -1 -/
theorem even_function_implies_a_eq_neg_one (a : ℝ) :
  IsEven (f a) → a = -1 := by sorry

end NUMINAMATH_CALUDE_even_function_implies_a_eq_neg_one_l2775_277532


namespace NUMINAMATH_CALUDE_cost_prices_calculation_l2775_277501

/-- Represents the cost price of an item -/
structure CostPrice where
  value : ℝ
  positive : value > 0

/-- Represents the selling price of an item -/
structure SellingPrice where
  value : ℝ
  positive : value > 0

/-- Calculates the selling price given a cost price and a percentage change -/
def calculateSellingPrice (cp : CostPrice) (percentageChange : ℝ) : SellingPrice :=
  { value := cp.value * (1 + percentageChange),
    positive := sorry }

/-- Determines if two real numbers are approximately equal within a small tolerance -/
def approximatelyEqual (x y : ℝ) : Prop :=
  |x - y| < 0.01

theorem cost_prices_calculation
  (diningSet : CostPrice)
  (chandelier : CostPrice)
  (sofaSet : CostPrice)
  (diningSetSelling : SellingPrice)
  (chandelierSelling : SellingPrice)
  (sofaSetSelling : SellingPrice) :
  (diningSetSelling = calculateSellingPrice diningSet (-0.18)) →
  (calculateSellingPrice diningSet 0.15).value = diningSetSelling.value + 2500 →
  (chandelierSelling = calculateSellingPrice chandelier 0.20) →
  (calculateSellingPrice chandelier (-0.20)).value = chandelierSelling.value - 3000 →
  (sofaSetSelling = calculateSellingPrice sofaSet (-0.10)) →
  (calculateSellingPrice sofaSet 0.25).value = sofaSetSelling.value + 4000 →
  approximatelyEqual diningSet.value 7576 ∧
  chandelier.value = 7500 ∧
  approximatelyEqual sofaSet.value 11429 := by
  sorry

#check cost_prices_calculation

end NUMINAMATH_CALUDE_cost_prices_calculation_l2775_277501


namespace NUMINAMATH_CALUDE_two_round_trips_time_l2775_277529

/-- Represents the time for a round trip given the time for one-way trip at normal speed -/
def round_trip_time (one_way_time : ℝ) : ℝ := one_way_time + 2 * one_way_time

/-- Proves that two round trips take 6 hours when one-way trip takes 1 hour -/
theorem two_round_trips_time : round_trip_time 1 * 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_two_round_trips_time_l2775_277529


namespace NUMINAMATH_CALUDE_remaining_length_is_24_l2775_277533

/-- A figure with perpendicular adjacent sides -/
structure PerpendicularFigure where
  sides : List ℝ
  perpendicular : Bool

/-- Function to calculate the total length of remaining segments after removal -/
def remainingLength (figure : PerpendicularFigure) (removedSides : ℕ) : ℝ :=
  sorry

/-- Theorem stating the total length of remaining segments is 24 units -/
theorem remaining_length_is_24 (figure : PerpendicularFigure) 
  (h1 : figure.sides = [10, 3, 8, 1, 1, 5]) 
  (h2 : figure.perpendicular = true) 
  (h3 : removedSides = 6) : 
  remainingLength figure removedSides = 24 :=
sorry

end NUMINAMATH_CALUDE_remaining_length_is_24_l2775_277533


namespace NUMINAMATH_CALUDE_valid_new_usage_exists_l2775_277572

/-- Represents the time spent on an app --/
structure AppTime where
  time : ℝ
  time_positive : time > 0

/-- Represents the usage data for four apps --/
structure AppUsage where
  app1 : AppTime
  app2 : AppTime
  app3 : AppTime
  app4 : AppTime

/-- Checks if the new usage data is consistent with halving two app times --/
def is_valid_new_usage (old_usage new_usage : AppUsage) : Prop :=
  (new_usage.app1.time = old_usage.app1.time / 2 ∧ new_usage.app3.time = old_usage.app3.time / 2 ∧
   new_usage.app2.time = old_usage.app2.time ∧ new_usage.app4.time = old_usage.app4.time) ∨
  (new_usage.app1.time = old_usage.app1.time / 2 ∧ new_usage.app2.time = old_usage.app2.time / 2 ∧
   new_usage.app3.time = old_usage.app3.time ∧ new_usage.app4.time = old_usage.app4.time) ∨
  (new_usage.app1.time = old_usage.app1.time / 2 ∧ new_usage.app4.time = old_usage.app4.time / 2 ∧
   new_usage.app2.time = old_usage.app2.time ∧ new_usage.app3.time = old_usage.app3.time) ∨
  (new_usage.app2.time = old_usage.app2.time / 2 ∧ new_usage.app3.time = old_usage.app3.time / 2 ∧
   new_usage.app1.time = old_usage.app1.time ∧ new_usage.app4.time = old_usage.app4.time) ∨
  (new_usage.app2.time = old_usage.app2.time / 2 ∧ new_usage.app4.time = old_usage.app4.time / 2 ∧
   new_usage.app1.time = old_usage.app1.time ∧ new_usage.app3.time = old_usage.app3.time) ∨
  (new_usage.app3.time = old_usage.app3.time / 2 ∧ new_usage.app4.time = old_usage.app4.time / 2 ∧
   new_usage.app1.time = old_usage.app1.time ∧ new_usage.app2.time = old_usage.app2.time)

theorem valid_new_usage_exists (old_usage : AppUsage) :
  ∃ new_usage : AppUsage, is_valid_new_usage old_usage new_usage :=
sorry

end NUMINAMATH_CALUDE_valid_new_usage_exists_l2775_277572


namespace NUMINAMATH_CALUDE_discount_percentage_proof_l2775_277551

theorem discount_percentage_proof (pants_price : ℝ) (socks_price : ℝ) (total_after_discount : ℝ) :
  pants_price = 110 →
  socks_price = 60 →
  total_after_discount = 392 →
  let original_total := 4 * pants_price + 2 * socks_price
  let discount_amount := original_total - total_after_discount
  let discount_percentage := (discount_amount / original_total) * 100
  discount_percentage = 30 := by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_proof_l2775_277551


namespace NUMINAMATH_CALUDE_circle_area_diameter_13_l2775_277549

/-- The area of a circle with diameter 13 meters is π * (13/2)^2 square meters. -/
theorem circle_area_diameter_13 :
  let diameter : ℝ := 13
  let radius : ℝ := diameter / 2
  let area : ℝ := π * radius ^ 2
  area = π * (13 / 2) ^ 2 := by sorry

end NUMINAMATH_CALUDE_circle_area_diameter_13_l2775_277549


namespace NUMINAMATH_CALUDE_set_A_properties_l2775_277507

/-- Property P: For any i, j (1 ≤ i ≤ j ≤ n), at least one of aᵢaⱼ and aⱼ/aᵢ belongs to A -/
def property_P (A : Set ℝ) : Prop :=
  ∀ (x y : ℝ), x ∈ A → y ∈ A → x ≤ y → (x * y ∈ A ∨ y / x ∈ A)

theorem set_A_properties {n : ℕ} (A : Set ℝ) (a : ℕ → ℝ) 
  (h_n : n ≥ 2)
  (h_A : A = {x | ∃ i, i ∈ Finset.range n ∧ x = a i})
  (h_sorted : ∀ i j, i < j → j < n → a i < a j)
  (h_P : property_P A) :
  (a 0 = 1) ∧ 
  ((Finset.range n).sum a / (Finset.range n).sum (λ i => (a i)⁻¹) = a (n - 1)) ∧
  (n = 5 → ∃ r : ℝ, ∀ i, i < 4 → a (i + 1) = r * a i) :=
by sorry

end NUMINAMATH_CALUDE_set_A_properties_l2775_277507


namespace NUMINAMATH_CALUDE_operation_problem_l2775_277545

-- Define the set of operations
inductive Operation
  | Add
  | Sub
  | Mul
  | Div

-- Define a function to apply an operation
def apply_op (op : Operation) (a b : ℚ) : ℚ :=
  match op with
  | Operation.Add => a + b
  | Operation.Sub => a - b
  | Operation.Mul => a * b
  | Operation.Div => a / b

-- State the theorem
theorem operation_problem (diamond circ : Operation) :
  (apply_op diamond 10 4) / (apply_op circ 6 2) = 5 →
  (apply_op diamond 8 3) / (apply_op circ 10 5) = 8/5 := by
  sorry

end NUMINAMATH_CALUDE_operation_problem_l2775_277545


namespace NUMINAMATH_CALUDE_revenue_difference_is_400_l2775_277521

/-- Represents the revenue difference between making elephant and giraffe statues -/
def revenue_difference (total_jade : ℕ) (giraffe_jade : ℕ) (giraffe_price : ℕ) (elephant_price : ℕ) : ℕ :=
  let elephant_jade := 2 * giraffe_jade
  let num_giraffes := total_jade / giraffe_jade
  let num_elephants := total_jade / elephant_jade
  let giraffe_revenue := num_giraffes * giraffe_price
  let elephant_revenue := num_elephants * elephant_price
  elephant_revenue - giraffe_revenue

/-- Proves that the revenue difference is $400 for the given conditions -/
theorem revenue_difference_is_400 :
  revenue_difference 1920 120 150 350 = 400 := by
  sorry

end NUMINAMATH_CALUDE_revenue_difference_is_400_l2775_277521


namespace NUMINAMATH_CALUDE_necklace_diamonds_l2775_277569

theorem necklace_diamonds (total_necklaces : ℕ) (diamonds_type1 diamonds_type2 : ℕ) (total_diamonds : ℕ) :
  total_necklaces = 20 →
  diamonds_type1 = 2 →
  diamonds_type2 = 5 →
  total_diamonds = 79 →
  ∃ (x y : ℕ), x + y = total_necklaces ∧ 
                diamonds_type1 * x + diamonds_type2 * y = total_diamonds ∧
                y = 13 :=
by sorry

end NUMINAMATH_CALUDE_necklace_diamonds_l2775_277569


namespace NUMINAMATH_CALUDE_fraction_simplification_l2775_277573

theorem fraction_simplification : (5 * (8 + 2)) / 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2775_277573


namespace NUMINAMATH_CALUDE_range_of_k_value_of_k_l2775_277541

-- Define the quadratic equation
def quadratic_equation (k x : ℝ) : Prop :=
  x^2 + (2 - 2*k)*x + k^2 = 0

-- Define the condition for real roots
def has_real_roots (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, quadratic_equation k x₁ ∧ quadratic_equation k x₂ ∧ x₁ ≠ x₂

-- Define the additional condition
def root_condition (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, quadratic_equation k x₁ ∧ quadratic_equation k x₂ ∧ 
    |x₁ + x₂| + 1 = x₁ * x₂

-- Theorem statements
theorem range_of_k (k : ℝ) : has_real_roots k → k ≤ 1/2 :=
sorry

theorem value_of_k : ∀ k : ℝ, has_real_roots k ∧ root_condition k → k = -3 :=
sorry

end NUMINAMATH_CALUDE_range_of_k_value_of_k_l2775_277541


namespace NUMINAMATH_CALUDE_tom_hiking_probability_l2775_277585

theorem tom_hiking_probability (p_fog : ℝ) (p_hike_foggy : ℝ) (p_hike_clear : ℝ)
  (h_fog : p_fog = 0.5)
  (h_hike_foggy : p_hike_foggy = 0.3)
  (h_hike_clear : p_hike_clear = 0.9) :
  p_fog * p_hike_foggy + (1 - p_fog) * p_hike_clear = 0.6 := by
  sorry

#check tom_hiking_probability

end NUMINAMATH_CALUDE_tom_hiking_probability_l2775_277585


namespace NUMINAMATH_CALUDE_smallest_square_area_l2775_277553

-- Define the radius of the circle
def radius : ℝ := 5

-- Define the diameter of the circle
def diameter : ℝ := 2 * radius

-- Define the side length of the square
def side_length : ℝ := diameter

-- Theorem: The area of the smallest square that can completely enclose a circle with a radius of 5 is 100
theorem smallest_square_area : side_length ^ 2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_area_l2775_277553


namespace NUMINAMATH_CALUDE_angle_expression_equality_l2775_277519

theorem angle_expression_equality (θ : Real) 
  (h1 : 0 < θ ∧ θ < π) 
  (h2 : Real.sin θ * Real.cos θ = -1/8) : 
  Real.sin (2*π + θ) - Real.sin (π/2 - θ) = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_expression_equality_l2775_277519


namespace NUMINAMATH_CALUDE_log_product_equality_l2775_277502

theorem log_product_equality : Real.log 3 / Real.log 8 * (Real.log 32 / Real.log 9) = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equality_l2775_277502


namespace NUMINAMATH_CALUDE_radio_range_is_125_l2775_277509

/-- The range of radios for two teams traveling in opposite directions --/
def radio_range (speed1 speed2 time : ℝ) : ℝ :=
  speed1 * time + speed2 * time

/-- Theorem: The radio range for the given scenario is 125 miles --/
theorem radio_range_is_125 :
  radio_range 20 30 2.5 = 125 := by
  sorry

end NUMINAMATH_CALUDE_radio_range_is_125_l2775_277509


namespace NUMINAMATH_CALUDE_f_2023_equals_2_l2775_277508

theorem f_2023_equals_2 (f : ℝ → ℝ) 
  (h1 : ∀ x > 0, f x > 0)
  (h2 : ∀ x y, x > y ∧ y > 0 → f (x - y) = Real.sqrt (f (x * y) + 2)) :
  f 2023 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_2023_equals_2_l2775_277508


namespace NUMINAMATH_CALUDE_simplification_fraction_l2775_277589

theorem simplification_fraction (k : ℝ) :
  ∃ (c d : ℤ), (6 * k + 12 + 3) / 3 = c * k + d ∧ (c : ℚ) / d = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplification_fraction_l2775_277589


namespace NUMINAMATH_CALUDE_completing_square_transform_l2775_277548

theorem completing_square_transform (x : ℝ) : 
  (x^2 - 6*x + 7 = 0) ↔ ((x - 3)^2 - 2 = 0) := by
sorry

end NUMINAMATH_CALUDE_completing_square_transform_l2775_277548


namespace NUMINAMATH_CALUDE_work_completion_l2775_277570

theorem work_completion (initial_days : ℕ) (absent_men : ℕ) (final_days : ℕ) :
  initial_days = 17 →
  absent_men = 8 →
  final_days = 21 →
  ∃ (original_men : ℕ),
    original_men * initial_days = (original_men - absent_men) * final_days ∧
    original_men = 42 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_l2775_277570


namespace NUMINAMATH_CALUDE_valid_medium_triangle_counts_l2775_277505

/-- Represents the side length of the original equilateral triangle -/
def originalSideLength : ℕ := 10

/-- Represents the side length of the smallest equilateral triangles -/
def smallestSideLength : ℕ := 1

/-- Represents the side length of the medium equilateral triangles -/
def mediumSideLength : ℕ := 2

/-- Represents the total number of shapes (triangles and parallelograms) -/
def totalShapes : ℕ := 25

/-- Predicate to check if a number is a valid count of medium triangles -/
def isValidMediumTriangleCount (m : ℕ) : Prop :=
  m % 2 = 1 ∧ 5 ≤ m ∧ m ≤ 25

/-- The set of all valid counts of medium triangles -/
def validMediumTriangleCounts : Set ℕ :=
  {m | isValidMediumTriangleCount m}

/-- Theorem stating the properties of valid medium triangle counts -/
theorem valid_medium_triangle_counts :
  validMediumTriangleCounts = {5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25} :=
sorry

end NUMINAMATH_CALUDE_valid_medium_triangle_counts_l2775_277505


namespace NUMINAMATH_CALUDE_maggie_goldfish_fraction_l2775_277522

theorem maggie_goldfish_fraction (total : ℕ) (caught_fraction : ℚ) (remaining : ℕ) :
  total = 100 →
  caught_fraction = 3 / 5 →
  remaining = 20 →
  (total : ℚ) / 2 = (caught_fraction * ((caught_fraction * (total : ℚ) + remaining) / caught_fraction) + remaining) / caught_fraction :=
by sorry

end NUMINAMATH_CALUDE_maggie_goldfish_fraction_l2775_277522


namespace NUMINAMATH_CALUDE_greatest_four_digit_divisible_l2775_277575

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def reverse_digits (n : ℕ) : ℕ := 
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  d4 * 1000 + d3 * 100 + d2 * 10 + d1

theorem greatest_four_digit_divisible (p : ℕ) 
  (h1 : is_four_digit p)
  (h2 : is_four_digit (reverse_digits p))
  (h3 : p % 63 = 0)
  (h4 : (reverse_digits p) % 63 = 0)
  (h5 : p % 19 = 0) :
  p ≤ 5985 ∧ (∀ q : ℕ, 
    is_four_digit q → 
    is_four_digit (reverse_digits q) → 
    q % 63 = 0 → 
    (reverse_digits q) % 63 = 0 → 
    q % 19 = 0 → 
    q ≤ p) :=
by sorry

end NUMINAMATH_CALUDE_greatest_four_digit_divisible_l2775_277575


namespace NUMINAMATH_CALUDE_minimum_N_l2775_277599

theorem minimum_N (k : ℕ) (h : k > 0) :
  let N := k * (2 * k^2 + 3 * k + 3)
  ∃ (S : Finset ℕ),
    (S.card = 2 * k + 1) ∧
    (∀ x ∈ S, x > 0) ∧
    (S.sum id > N) ∧
    (∀ T ⊆ S, T.card = k → T.sum id ≤ N / 2) ∧
    (∀ M < N, ¬∃ (S' : Finset ℕ),
      (S'.card = 2 * k + 1) ∧
      (∀ x ∈ S', x > 0) ∧
      (S'.sum id > M) ∧
      (∀ T ⊆ S', T.card = k → T.sum id ≤ M / 2)) :=
by sorry

end NUMINAMATH_CALUDE_minimum_N_l2775_277599


namespace NUMINAMATH_CALUDE_meaningful_expression_l2775_277593

/-- The expression sqrt(2-m) / (m+2) is meaningful if and only if m ≤ 2 and m ≠ -2 -/
theorem meaningful_expression (m : ℝ) : 
  (∃ x : ℝ, x^2 = 2 - m ∧ m + 2 ≠ 0) ↔ (m ≤ 2 ∧ m ≠ -2) :=
by sorry

end NUMINAMATH_CALUDE_meaningful_expression_l2775_277593


namespace NUMINAMATH_CALUDE_sprite_volume_calculation_l2775_277586

def maaza_volume : ℕ := 20
def pepsi_volume : ℕ := 144
def total_cans : ℕ := 133

theorem sprite_volume_calculation :
  ∃ (can_volume sprite_volume : ℕ),
    can_volume > 0 ∧
    maaza_volume % can_volume = 0 ∧
    pepsi_volume % can_volume = 0 ∧
    sprite_volume % can_volume = 0 ∧
    maaza_volume / can_volume + pepsi_volume / can_volume + sprite_volume / can_volume = total_cans ∧
    sprite_volume = 368 := by
  sorry

end NUMINAMATH_CALUDE_sprite_volume_calculation_l2775_277586


namespace NUMINAMATH_CALUDE_pole_length_problem_l2775_277595

theorem pole_length_problem (original_length : ℝ) (cut_length : ℝ) : 
  cut_length = 0.7 * original_length →
  cut_length = 14 →
  original_length = 20 := by
sorry

end NUMINAMATH_CALUDE_pole_length_problem_l2775_277595


namespace NUMINAMATH_CALUDE_stones_kept_as_favorite_l2775_277584

theorem stones_kept_as_favorite (original_stones sent_away_stones : ℕ) 
  (h1 : original_stones = 78) 
  (h2 : sent_away_stones = 63) : 
  original_stones - sent_away_stones = 15 := by
  sorry

end NUMINAMATH_CALUDE_stones_kept_as_favorite_l2775_277584


namespace NUMINAMATH_CALUDE_particle_hit_probability_l2775_277514

/-- Probability of hitting (0,0) from position (x,y) -/
def P (x y : ℕ) : ℚ :=
  if x = 0 ∧ y = 0 then 1
  else if x = 0 ∨ y = 0 then 0
  else (1/3) * P (x-1) y + (1/3) * P x (y-1) + (1/3) * P (x-1) (y-1)

/-- The particle starts at (5,5) -/
def start_pos : ℕ × ℕ := (5, 5)

/-- The probability of hitting (0,0) is m/3^n -/
def hit_prob : ℚ := 1 / 3^5

theorem particle_hit_probability :
  P start_pos.1 start_pos.2 = hit_prob :=
sorry

end NUMINAMATH_CALUDE_particle_hit_probability_l2775_277514


namespace NUMINAMATH_CALUDE_polynomial_product_expansion_l2775_277536

theorem polynomial_product_expansion (x : ℝ) : 
  (5 * x + 3) * (6 * x^2 + 2) = 30 * x^3 + 18 * x^2 + 10 * x + 6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_expansion_l2775_277536


namespace NUMINAMATH_CALUDE_steve_union_dues_l2775_277511

/-- Calculate the amount lost to local union dues given gross salary, tax rate, healthcare rate, and take-home pay -/
def union_dues (gross_salary : ℝ) (tax_rate : ℝ) (healthcare_rate : ℝ) (take_home_pay : ℝ) : ℝ :=
  gross_salary - (tax_rate * gross_salary) - (healthcare_rate * gross_salary) - take_home_pay

/-- Theorem: Given Steve's financial information, prove that he loses $800 to local union dues -/
theorem steve_union_dues :
  union_dues 40000 0.20 0.10 27200 = 800 := by
  sorry

end NUMINAMATH_CALUDE_steve_union_dues_l2775_277511


namespace NUMINAMATH_CALUDE_simplify_fraction_l2775_277557

theorem simplify_fraction : (90 : ℚ) / 126 = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2775_277557


namespace NUMINAMATH_CALUDE_average_of_five_numbers_l2775_277543

theorem average_of_five_numbers (total : ℕ) (avg_all : ℚ) (avg_three : ℚ) :
  total = 8 →
  avg_all = 20 →
  avg_three = 33333333333333336 / 1000000000000000 →
  let sum_all := avg_all * total
  let sum_three := avg_three * 3
  let sum_five := sum_all - sum_three
  sum_five / 5 = 12 := by
sorry

#eval 33333333333333336 / 1000000000000000  -- To verify the fraction equals 33.333333333333336

end NUMINAMATH_CALUDE_average_of_five_numbers_l2775_277543


namespace NUMINAMATH_CALUDE_students_guinea_pigs_difference_l2775_277527

theorem students_guinea_pigs_difference (students_per_class : ℕ) (guinea_pigs_per_class : ℕ) (num_classes : ℕ) :
  students_per_class = 22 →
  guinea_pigs_per_class = 3 →
  num_classes = 5 →
  students_per_class * num_classes - guinea_pigs_per_class * num_classes = 95 :=
by sorry

end NUMINAMATH_CALUDE_students_guinea_pigs_difference_l2775_277527


namespace NUMINAMATH_CALUDE_line_perpendicular_to_line_l2775_277535

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between two planes
variable (perpendicular_plane_plane : Plane → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perpendicular_line_line : Line → Line → Prop)

-- Define the subset relation between a line and a plane
variable (subset_line_plane : Line → Plane → Prop)

-- State the theorem
theorem line_perpendicular_to_line
  (l m : Line) (α β : Plane)
  (h1 : perpendicular_line_plane l α)
  (h2 : subset_line_plane m β)
  (h3 : perpendicular_plane_plane α β) :
  perpendicular_line_line l m :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_line_l2775_277535


namespace NUMINAMATH_CALUDE_total_games_won_l2775_277520

-- Define the number of games won by Betsy
def betsy_games : ℕ := 5

-- Define Helen's games in terms of Betsy's
def helen_games : ℕ := 2 * betsy_games

-- Define Susan's games in terms of Betsy's
def susan_games : ℕ := 3 * betsy_games

-- Theorem to prove the total number of games won
theorem total_games_won : betsy_games + helen_games + susan_games = 30 := by
  sorry

end NUMINAMATH_CALUDE_total_games_won_l2775_277520


namespace NUMINAMATH_CALUDE_samuel_apple_ratio_l2775_277560

/-- Prove that the ratio of apples Samuel ate to the total number of apples he bought is 1:2 -/
theorem samuel_apple_ratio :
  let bonnie_apples : ℕ := 8
  let samuel_extra_apples : ℕ := 20
  let samuel_total_apples : ℕ := bonnie_apples + samuel_extra_apples
  let samuel_pie_apples : ℕ := samuel_total_apples / 7
  let samuel_left_apples : ℕ := 10
  let samuel_eaten_apples : ℕ := samuel_total_apples - samuel_pie_apples - samuel_left_apples
  (samuel_eaten_apples : ℚ) / (samuel_total_apples : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_samuel_apple_ratio_l2775_277560


namespace NUMINAMATH_CALUDE_circle_radius_tangent_to_lines_l2775_277539

/-- Theorem: For a circle with center (0, k) where k > 8, if the circle is tangent to the lines y = x, y = -x, and y = 8, then its radius is 8√2 + 8. -/
theorem circle_radius_tangent_to_lines (k : ℝ) (h : k > 8) : 
  let circle := {(x, y) : ℝ × ℝ | x^2 + (y - k)^2 = (k - 8)^2}
  (∀ (x y : ℝ), (x = y ∧ (x, y) ∈ circle) → (x, y) ∈ {(x, y) | x = y}) →
  (∀ (x y : ℝ), (x = -y ∧ (x, y) ∈ circle) → (x, y) ∈ {(x, y) | x = -y}) →
  (∀ (x : ℝ), (x, 8) ∈ circle → x = 0) →
  k - 8 = 8 * (Real.sqrt 2 + 1) := by
sorry

end NUMINAMATH_CALUDE_circle_radius_tangent_to_lines_l2775_277539


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2775_277550

/-- An arithmetic sequence with general term a_n = 4n - 3 -/
def arithmetic_sequence (n : ℕ) : ℤ := 4 * n - 3

/-- The first term of the sequence -/
def first_term : ℤ := arithmetic_sequence 1

/-- The second term of the sequence -/
def second_term : ℤ := arithmetic_sequence 2

/-- The common difference of the sequence -/
def common_difference : ℤ := second_term - first_term

theorem arithmetic_sequence_properties :
  first_term = 1 ∧ common_difference = 4 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2775_277550


namespace NUMINAMATH_CALUDE_product_97_103_l2775_277542

theorem product_97_103 : 97 * 103 = 9991 := by
  sorry

end NUMINAMATH_CALUDE_product_97_103_l2775_277542


namespace NUMINAMATH_CALUDE_pollywogs_disappearance_l2775_277500

/-- The number of pollywogs that mature into toads and leave the pond per day -/
def maturation_rate : ℕ := 50

/-- The number of pollywogs Melvin catches per day for the first 20 days -/
def melvin_catch_rate : ℕ := 10

/-- The number of days Melvin catches pollywogs -/
def melvin_catch_days : ℕ := 20

/-- The total number of days it took for all pollywogs to disappear -/
def total_days : ℕ := 44

/-- The initial number of pollywogs in the pond -/
def initial_pollywogs : ℕ := 2400

theorem pollywogs_disappearance :
  initial_pollywogs = 
    (maturation_rate + melvin_catch_rate) * melvin_catch_days + 
    maturation_rate * (total_days - melvin_catch_days) := by
  sorry

end NUMINAMATH_CALUDE_pollywogs_disappearance_l2775_277500


namespace NUMINAMATH_CALUDE_binomial_variance_problem_l2775_277504

/-- A function representing the expectation of a binomial distribution -/
def expectation_binomial (n : ℕ) (p : ℝ) : ℝ := n * p

/-- A function representing the variance of a binomial distribution -/
def variance_binomial (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

theorem binomial_variance_problem (p : ℝ) 
  (hX : expectation_binomial 3 p = 1) 
  (hY : expectation_binomial 4 p = 4 * p) :
  variance_binomial 4 p = 8/9 := by
sorry

end NUMINAMATH_CALUDE_binomial_variance_problem_l2775_277504


namespace NUMINAMATH_CALUDE_g_of_seven_l2775_277518

/-- Given a function g(x) = (2x + 3) / (4x - 5), prove that g(7) = 17/23 -/
theorem g_of_seven (g : ℝ → ℝ) (h : ∀ x, g x = (2 * x + 3) / (4 * x - 5)) : 
  g 7 = 17 / 23 := by
  sorry

end NUMINAMATH_CALUDE_g_of_seven_l2775_277518


namespace NUMINAMATH_CALUDE_captain_age_proof_l2775_277510

def cricket_team_problem (team_size : ℕ) (team_avg_age : ℕ) (age_diff : ℕ) (remaining_avg_diff : ℕ) : Prop :=
  let captain_age : ℕ := 26
  let keeper_age : ℕ := captain_age + age_diff
  let total_age : ℕ := team_size * team_avg_age
  let remaining_players : ℕ := team_size - 2
  let remaining_avg : ℕ := team_avg_age - remaining_avg_diff
  total_age = captain_age + keeper_age + remaining_players * remaining_avg

theorem captain_age_proof :
  cricket_team_problem 11 23 3 1 := by
  sorry

end NUMINAMATH_CALUDE_captain_age_proof_l2775_277510


namespace NUMINAMATH_CALUDE_show_attendance_l2775_277528

theorem show_attendance (adult_price child_price total_cost : ℕ) 
  (num_children : ℕ) (h1 : adult_price = 12) (h2 : child_price = 10) 
  (h3 : num_children = 3) (h4 : total_cost = 66) : 
  ∃ (num_adults : ℕ), num_adults = 3 ∧ 
    adult_price * num_adults + child_price * num_children = total_cost :=
by
  sorry

end NUMINAMATH_CALUDE_show_attendance_l2775_277528


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2775_277592

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 4 + a 6 + a 8 + a 10 + a 12 = 120 →
  2 * a 10 - a 12 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2775_277592


namespace NUMINAMATH_CALUDE_hcf_36_84_l2775_277546

theorem hcf_36_84 : Nat.gcd 36 84 = 12 := by
  sorry

end NUMINAMATH_CALUDE_hcf_36_84_l2775_277546


namespace NUMINAMATH_CALUDE_businessmen_drinking_none_l2775_277538

theorem businessmen_drinking_none (total : ℕ) (coffee tea soda coffee_tea tea_soda coffee_soda all_three : ℕ) : 
  total = 30 ∧ 
  coffee = 15 ∧ 
  tea = 12 ∧ 
  soda = 8 ∧ 
  coffee_tea = 7 ∧ 
  tea_soda = 3 ∧ 
  coffee_soda = 2 ∧ 
  all_three = 1 → 
  total - (coffee + tea + soda - coffee_tea - tea_soda - coffee_soda + all_three) = 6 := by
sorry

end NUMINAMATH_CALUDE_businessmen_drinking_none_l2775_277538


namespace NUMINAMATH_CALUDE_height_relation_l2775_277558

/-- Two right circular cylinders with equal volume and related radii -/
structure TwoCylinders where
  r1 : ℝ  -- radius of the first cylinder
  h1 : ℝ  -- height of the first cylinder
  r2 : ℝ  -- radius of the second cylinder
  h2 : ℝ  -- height of the second cylinder
  r1_pos : 0 < r1  -- r1 is positive
  h1_pos : 0 < h1  -- h1 is positive
  r2_pos : 0 < r2  -- r2 is positive
  h2_pos : 0 < h2  -- h2 is positive
  volume_eq : r1^2 * h1 = r2^2 * h2  -- volumes are equal
  radius_relation : r2 = 1.2 * r1  -- second radius is 20% more than the first

/-- The height of the first cylinder is 44% more than the height of the second cylinder -/
theorem height_relation (c : TwoCylinders) : c.h1 = 1.44 * c.h2 := by
  sorry

end NUMINAMATH_CALUDE_height_relation_l2775_277558


namespace NUMINAMATH_CALUDE_charlies_share_l2775_277577

theorem charlies_share (total : ℚ) (a b c : ℚ) : 
  total = 10000 →
  a = (1/3) * b →
  b = (1/2) * c →
  a + b + c = total →
  c = 6000 := by
sorry

end NUMINAMATH_CALUDE_charlies_share_l2775_277577


namespace NUMINAMATH_CALUDE_sum_of_digits_6_11_l2775_277597

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

def ones_digit (n : ℕ) : ℕ := n % 10

theorem sum_of_digits_6_11 : 
  tens_digit (6^11) + ones_digit (6^11) = 13 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_6_11_l2775_277597


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l2775_277531

theorem simplify_and_rationalize :
  (Real.sqrt 3 / Real.sqrt 7) * (Real.sqrt 5 / Real.sqrt 8) * (Real.sqrt 6 / Real.sqrt 9) = Real.sqrt 35 / 14 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l2775_277531


namespace NUMINAMATH_CALUDE_seashells_given_to_sam_l2775_277596

/-- Given that Joan found a certain number of seashells and has some left after giving some to Sam,
    prove that the number of seashells given to Sam is the difference between the initial and remaining amounts. -/
theorem seashells_given_to_sam 
  (initial : ℕ) 
  (remaining : ℕ) 
  (h1 : initial = 70) 
  (h2 : remaining = 27) 
  (h3 : remaining < initial) : 
  initial - remaining = 43 := by
sorry

end NUMINAMATH_CALUDE_seashells_given_to_sam_l2775_277596


namespace NUMINAMATH_CALUDE_danny_watermelons_l2775_277583

theorem danny_watermelons (danny_slices_per_melon : ℕ) (sister_slices : ℕ) (total_slices : ℕ)
  (h1 : danny_slices_per_melon = 10)
  (h2 : sister_slices = 15)
  (h3 : total_slices = 45) :
  ∃ danny_melons : ℕ, danny_melons * danny_slices_per_melon + sister_slices = total_slices ∧ danny_melons = 3 := by
  sorry

end NUMINAMATH_CALUDE_danny_watermelons_l2775_277583


namespace NUMINAMATH_CALUDE_abc_ratio_theorem_l2775_277526

theorem abc_ratio_theorem (a b c : ℚ) 
  (h : (|a|/a) + (|b|/b) + (|c|/c) = 1) : 
  a * b * c / |a * b * c| = -1 := by
sorry

end NUMINAMATH_CALUDE_abc_ratio_theorem_l2775_277526


namespace NUMINAMATH_CALUDE_max_inscribed_equilateral_triangle_area_l2775_277590

/-- The maximum area of an equilateral triangle inscribed in a 12 by 15 rectangle -/
theorem max_inscribed_equilateral_triangle_area :
  ∃ (A : ℝ), A = 48 * Real.sqrt 3 ∧
  ∀ (s : ℝ), s > 0 →
    s * Real.sqrt 3 / 2 ≤ 12 →
    s ≤ 15 →
    s * s * Real.sqrt 3 / 4 ≤ A :=
by sorry

end NUMINAMATH_CALUDE_max_inscribed_equilateral_triangle_area_l2775_277590


namespace NUMINAMATH_CALUDE_power_multiplication_l2775_277571

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l2775_277571


namespace NUMINAMATH_CALUDE_equation_solution_l2775_277563

theorem equation_solution :
  ∀ x : ℝ, (2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 2) → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2775_277563


namespace NUMINAMATH_CALUDE_father_steps_problem_l2775_277554

/-- Calculates the number of steps taken by Father given the step ratios and total steps of children -/
def father_steps (father_masha_ratio : ℚ) (masha_yasha_ratio : ℚ) (total_children_steps : ℕ) : ℕ :=
  sorry

/-- Theorem stating that Father takes 90 steps given the problem conditions -/
theorem father_steps_problem :
  let father_masha_ratio : ℚ := 3 / 5
  let masha_yasha_ratio : ℚ := 3 / 5
  let total_children_steps : ℕ := 400
  father_steps father_masha_ratio masha_yasha_ratio total_children_steps = 90 := by
  sorry

end NUMINAMATH_CALUDE_father_steps_problem_l2775_277554


namespace NUMINAMATH_CALUDE_simple_interest_rate_problem_l2775_277525

/-- Given the principal, amount, time, and formulas for simple interest and amount,
    prove that the rate percent is 5%. -/
theorem simple_interest_rate_problem (P A : ℕ) (T : ℕ) (h1 : P = 750) (h2 : A = 900) (h3 : T = 4) :
  ∃ R : ℚ,
    R = 5 ∧
    A = P + P * R * (T : ℚ) / 100 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_problem_l2775_277525


namespace NUMINAMATH_CALUDE_dvd_average_price_l2775_277559

theorem dvd_average_price (price1 price2 : ℚ) (count1 count2 : ℕ) 
  (h1 : price1 = 2)
  (h2 : price2 = 5)
  (h3 : count1 = 10)
  (h4 : count2 = 5) :
  (price1 * count1 + price2 * count2) / (count1 + count2) = 3 := by
sorry

end NUMINAMATH_CALUDE_dvd_average_price_l2775_277559


namespace NUMINAMATH_CALUDE_compare_x_powers_l2775_277503

theorem compare_x_powers (x : ℝ) (h : 0 < x ∧ x < 1) : x^2 < Real.sqrt x ∧ Real.sqrt x < x ∧ x < 1/x := by
  sorry

end NUMINAMATH_CALUDE_compare_x_powers_l2775_277503


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2775_277524

def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmeticSequence a → a 3 + a 7 = 37 → a 2 + a 4 + a 6 + a 8 = 74 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2775_277524


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2775_277537

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x < 4}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2775_277537


namespace NUMINAMATH_CALUDE_power_of_1_01_gt_1000_l2775_277540

theorem power_of_1_01_gt_1000 : (1.01 : ℝ) ^ 1000 > 1000 := by
  sorry

end NUMINAMATH_CALUDE_power_of_1_01_gt_1000_l2775_277540


namespace NUMINAMATH_CALUDE_perimeter_semicircular_pentagon_l2775_277578

/-- The perimeter of a region bounded by semicircular arcs constructed on each side of a regular pentagon --/
theorem perimeter_semicircular_pentagon (side_length : ℝ) : 
  side_length = 5 / π → 
  (5 : ℝ) * (π * side_length / 2) = 25 / 2 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_semicircular_pentagon_l2775_277578


namespace NUMINAMATH_CALUDE_smallest_with_four_odd_eight_even_divisors_l2775_277516

/-- Count of positive odd integer divisors of n -/
def oddDivisorCount (n : ℕ+) : ℕ := sorry

/-- Count of positive even integer divisors of n -/
def evenDivisorCount (n : ℕ+) : ℕ := sorry

/-- Predicate for a number having exactly four positive odd integer divisors and eight positive even integer divisors -/
def hasFourOddEightEvenDivisors (n : ℕ+) : Prop :=
  oddDivisorCount n = 4 ∧ evenDivisorCount n = 8

theorem smallest_with_four_odd_eight_even_divisors :
  ∃ (n : ℕ+), hasFourOddEightEvenDivisors n ∧
  ∀ (m : ℕ+), hasFourOddEightEvenDivisors m → n ≤ m :=
by
  use 60
  sorry

end NUMINAMATH_CALUDE_smallest_with_four_odd_eight_even_divisors_l2775_277516


namespace NUMINAMATH_CALUDE_f_positive_iff_l2775_277581

def f (x : ℝ) := (x + 1) * (x - 1) * (x - 3)

theorem f_positive_iff (x : ℝ) : f x > 0 ↔ x ∈ Set.Ioo (-1) 1 ∪ Set.Ioi 3 := by
  sorry

end NUMINAMATH_CALUDE_f_positive_iff_l2775_277581


namespace NUMINAMATH_CALUDE_coin_division_sum_equals_pairs_l2775_277544

/-- Represents the process of dividing coins into piles --/
def CoinDivisionProcess : Type := List (Nat × Nat)

/-- The number of coins --/
def n : Nat := 25

/-- Calculates the sum of products for a given division process --/
def sum_of_products (process : CoinDivisionProcess) : Nat :=
  process.foldl (fun sum pair => sum + pair.1 * pair.2) 0

/-- Represents all possible division processes for n coins --/
def all_division_processes (n : Nat) : Set CoinDivisionProcess :=
  sorry

/-- Theorem stating that the sum of products equals the number of pairs of coins --/
theorem coin_division_sum_equals_pairs :
  ∀ (process : CoinDivisionProcess),
    process ∈ all_division_processes n →
    sum_of_products process = n.choose 2 := by
  sorry

end NUMINAMATH_CALUDE_coin_division_sum_equals_pairs_l2775_277544


namespace NUMINAMATH_CALUDE_A_intersect_B_l2775_277517

def A : Set ℕ := {1, 2, 4, 6, 8}

def B : Set ℕ := {x | ∃ k ∈ A, x = 2 * k}

theorem A_intersect_B : A ∩ B = {2, 4, 8} := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_l2775_277517


namespace NUMINAMATH_CALUDE_olivia_won_five_games_l2775_277579

/-- Represents a contestant in the math quiz competition -/
inductive Contestant
| Liam
| Noah
| Olivia

/-- The number of games won by a contestant -/
def games_won (c : Contestant) : ℕ :=
  match c with
  | Contestant.Liam => 6
  | Contestant.Noah => 4
  | Contestant.Olivia => 5  -- This is what we want to prove

/-- The number of games lost by a contestant -/
def games_lost (c : Contestant) : ℕ :=
  match c with
  | Contestant.Liam => 3
  | Contestant.Noah => 4
  | Contestant.Olivia => 4

/-- The total number of games played by each contestant -/
def total_games (c : Contestant) : ℕ := games_won c + games_lost c

/-- Each win gives 1 point -/
def points (c : Contestant) : ℕ := games_won c

/-- Theorem stating that Olivia won 5 games -/
theorem olivia_won_five_games :
  (∀ c1 c2 : Contestant, c1 ≠ c2 → total_games c1 = total_games c2) →
  games_won Contestant.Olivia = 5 := by sorry

end NUMINAMATH_CALUDE_olivia_won_five_games_l2775_277579


namespace NUMINAMATH_CALUDE_toy_cost_calculation_l2775_277534

/-- Represents the initial weekly cost price of a toy in Rupees -/
def initial_cost : ℝ := 1300

/-- Number of toys sold -/
def num_toys : ℕ := 18

/-- Discount rate applied to the toys -/
def discount_rate : ℝ := 0.1

/-- Total revenue from the sale in Rupees -/
def total_revenue : ℝ := 27300

theorem toy_cost_calculation :
  initial_cost * num_toys * (1 - discount_rate) = total_revenue - 3 * initial_cost := by
  sorry

#check toy_cost_calculation

end NUMINAMATH_CALUDE_toy_cost_calculation_l2775_277534


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2775_277564

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 13) = 11 → x = 108 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2775_277564


namespace NUMINAMATH_CALUDE_concentric_circles_ratio_l2775_277555

theorem concentric_circles_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  π * b^2 - π * a^2 = 4 * (π * a^2) → a / b = 1 / Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_ratio_l2775_277555


namespace NUMINAMATH_CALUDE_parallel_lines_j_value_l2775_277591

/-- Given two points on a line and another line equation, find the j-coordinate of the second point -/
theorem parallel_lines_j_value :
  let line1_point1 : ℝ × ℝ := (5, -6)
  let line1_point2 : ℝ × ℝ := (j, 29)
  let line2_slope : ℝ := 3 / 2
  let line2_equation (x y : ℝ) := 3 * x - 2 * y = 15
  ∀ j : ℝ,
    (line1_point2.2 - line1_point1.2) / (line1_point2.1 - line1_point1.1) = line2_slope →
    j = 85 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_j_value_l2775_277591


namespace NUMINAMATH_CALUDE_inverse_existence_l2775_277582

-- Define the set of graph labels
inductive GraphLabel
  | A | B | C | D | E

-- Define a predicate for passing the horizontal line test
def passes_horizontal_line_test (g : GraphLabel) : Prop :=
  match g with
  | GraphLabel.A => False
  | GraphLabel.B => True
  | GraphLabel.C => True
  | GraphLabel.D => True
  | GraphLabel.E => False

-- Define a predicate for having an inverse
def has_inverse (g : GraphLabel) : Prop :=
  passes_horizontal_line_test g

-- Theorem statement
theorem inverse_existence (g : GraphLabel) :
  has_inverse g ↔ (g = GraphLabel.B ∨ g = GraphLabel.C ∨ g = GraphLabel.D) :=
by sorry

end NUMINAMATH_CALUDE_inverse_existence_l2775_277582


namespace NUMINAMATH_CALUDE_gabriel_has_35_boxes_l2775_277598

-- Define the number of boxes for each person
def stan_boxes : ℕ := 120

-- Define relationships between box counts
def joseph_boxes : ℕ := (stan_boxes * 20) / 100
def jules_boxes : ℕ := joseph_boxes + 5
def john_boxes : ℕ := (jules_boxes * 120) / 100
def martin_boxes : ℕ := (jules_boxes * 150) / 100
def alice_boxes : ℕ := (john_boxes * 75) / 100
def gabriel_boxes : ℕ := (martin_boxes + alice_boxes) / 2

-- Theorem to prove
theorem gabriel_has_35_boxes : gabriel_boxes = 35 := by
  sorry

end NUMINAMATH_CALUDE_gabriel_has_35_boxes_l2775_277598


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_squares_minus_product_l2775_277567

theorem quadratic_roots_sum_squares_minus_product (m n : ℝ) : 
  m^2 - 5*m - 2 = 0 → n^2 - 5*n - 2 = 0 → m^2 + n^2 - m*n = 31 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_squares_minus_product_l2775_277567


namespace NUMINAMATH_CALUDE_p_range_l2775_277515

/-- The function p(x) defined for x ≥ 0 -/
def p (x : ℝ) : ℝ := x^4 + 8*x^2 + 16

/-- The range of p(x) is [16, ∞) -/
theorem p_range :
  ∀ y : ℝ, (∃ x : ℝ, x ≥ 0 ∧ p x = y) ↔ y ≥ 16 := by sorry

end NUMINAMATH_CALUDE_p_range_l2775_277515
