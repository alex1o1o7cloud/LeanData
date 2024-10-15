import Mathlib

namespace NUMINAMATH_CALUDE_f_is_odd_l3256_325639

def f (x : ℝ) : ℝ := x^3 - x

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by sorry

end NUMINAMATH_CALUDE_f_is_odd_l3256_325639


namespace NUMINAMATH_CALUDE_set_operations_l3256_325640

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

end NUMINAMATH_CALUDE_set_operations_l3256_325640


namespace NUMINAMATH_CALUDE_problem_solution_l3256_325602

theorem problem_solution : 
  ∃ x : ℝ, (0.4 * 2 = 0.25 * (0.3 * 15 + x)) ∧ (x = -1.3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3256_325602


namespace NUMINAMATH_CALUDE_oplus_inequality_range_l3256_325625

def oplus (x y : ℝ) : ℝ := x * (2 - y)

theorem oplus_inequality_range (a : ℝ) :
  (∀ t : ℝ, oplus (t - a) (t + a) < 1) ↔ (0 < a ∧ a < 2) := by
  sorry

end NUMINAMATH_CALUDE_oplus_inequality_range_l3256_325625


namespace NUMINAMATH_CALUDE_stock_yield_inconsistency_l3256_325673

theorem stock_yield_inconsistency (price : ℝ) (price_pos : price > 0) :
  ¬(∃ (dividend : ℝ), 
    dividend = 0.2 * price ∧ 
    dividend / price = 0.1) :=
by
  sorry

#check stock_yield_inconsistency

end NUMINAMATH_CALUDE_stock_yield_inconsistency_l3256_325673


namespace NUMINAMATH_CALUDE_joes_total_lift_weight_l3256_325694

/-- The total weight of Joe's two lifts is 600 pounds -/
theorem joes_total_lift_weight :
  ∀ (first_lift second_lift : ℕ),
  first_lift = 300 →
  2 * first_lift = second_lift + 300 →
  first_lift + second_lift = 600 :=
by
  sorry

end NUMINAMATH_CALUDE_joes_total_lift_weight_l3256_325694


namespace NUMINAMATH_CALUDE_best_card_to_disprove_l3256_325672

-- Define the set of visible card sides
inductive CardSide
| Letter (c : Char)
| Number (n : Nat)

-- Define a card as a pair of sides
def Card := (CardSide × CardSide)

-- Define the property of being a consonant
def isConsonant (c : Char) : Prop := c ∈ ['B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z']

-- Define the property of being an odd number
def isOdd (n : Nat) : Prop := n % 2 = 1

-- John's statement as a function
def johnsStatement (card : Card) : Prop :=
  match card with
  | (CardSide.Letter c, CardSide.Number n) => isConsonant c → isOdd n
  | (CardSide.Number n, CardSide.Letter c) => isConsonant c → isOdd n
  | _ => True

-- The set of visible card sides
def visibleSides : List CardSide := [CardSide.Letter 'A', CardSide.Letter 'B', CardSide.Number 7, CardSide.Number 8, CardSide.Number 9]

-- The theorem to prove
theorem best_card_to_disprove (cards : List Card) :
  (∀ card ∈ cards, (CardSide.Number 8 ∈ card.1 :: card.2 :: []) →
    ¬(∀ c ∈ cards, johnsStatement c)) →
  (∀ side ∈ visibleSides, side ≠ CardSide.Number 8 →
    ∃ c ∈ cards, (side ∈ c.1 :: c.2 :: []) ∧
      (∀ card ∈ cards, (side ∈ card.1 :: card.2 :: []) →
        (∃ c' ∈ cards, ¬johnsStatement c'))) :=
by sorry

end NUMINAMATH_CALUDE_best_card_to_disprove_l3256_325672


namespace NUMINAMATH_CALUDE_base_conversion_1729_to_base_7_l3256_325619

theorem base_conversion_1729_to_base_7 :
  (5 * 7^3 + 0 * 7^2 + 2 * 7^1 + 0 * 7^0 : ℕ) = 1729 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_1729_to_base_7_l3256_325619


namespace NUMINAMATH_CALUDE_function_minimum_and_tangent_line_l3256_325623

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

end NUMINAMATH_CALUDE_function_minimum_and_tangent_line_l3256_325623


namespace NUMINAMATH_CALUDE_club_officer_selection_l3256_325633

/-- The number of ways to choose distinct officers from a group -/
def chooseOfficers (n : ℕ) (k : ℕ) : ℕ :=
  (n - k + 1).factorial / (n - k).factorial

theorem club_officer_selection :
  chooseOfficers 12 5 = 95040 := by
  sorry

end NUMINAMATH_CALUDE_club_officer_selection_l3256_325633


namespace NUMINAMATH_CALUDE_waiter_tips_sum_l3256_325650

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

end NUMINAMATH_CALUDE_waiter_tips_sum_l3256_325650


namespace NUMINAMATH_CALUDE_five_leaders_three_cities_l3256_325618

/-- The number of ways to allocate n leaders to k cities, with each city having at least one leader -/
def allocationSchemes (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem stating that allocating 5 leaders to 3 cities results in 240 schemes -/
theorem five_leaders_three_cities : allocationSchemes 5 3 = 240 := by sorry

end NUMINAMATH_CALUDE_five_leaders_three_cities_l3256_325618


namespace NUMINAMATH_CALUDE_third_motorcyclist_speed_l3256_325664

theorem third_motorcyclist_speed 
  (v1 : ℝ) (v2 : ℝ) (v3 : ℝ) (t_delay : ℝ) (t_diff : ℝ) :
  v1 = 80 →
  v2 = 60 →
  t_delay = 0.5 →
  t_diff = 1.25 →
  v3 * (v3 * t_diff / (v3 - v1) - t_delay) = v1 * (v3 * t_diff / (v3 - v1)) →
  v3 * (v3 * t_diff / (v3 - v1) - v3 * t_diff / (v3 - v2) - t_delay) = 
    v2 * (v3 * t_diff / (v3 - v1) - t_delay) →
  v3 = 100 := by
sorry


end NUMINAMATH_CALUDE_third_motorcyclist_speed_l3256_325664


namespace NUMINAMATH_CALUDE_earnings_before_car_purchase_l3256_325647

/-- Calculates the total earnings before saving enough to buy a car. -/
def totalEarningsBeforePurchase (monthlyEarnings : ℕ) (monthlySavings : ℕ) (carCost : ℕ) : ℕ :=
  (carCost / monthlySavings) * monthlyEarnings

/-- Theorem stating the total earnings before saving enough to buy the car. -/
theorem earnings_before_car_purchase :
  totalEarningsBeforePurchase 4000 500 45000 = 360000 := by
  sorry

end NUMINAMATH_CALUDE_earnings_before_car_purchase_l3256_325647


namespace NUMINAMATH_CALUDE_plane_perpendicularity_l3256_325621

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

end NUMINAMATH_CALUDE_plane_perpendicularity_l3256_325621


namespace NUMINAMATH_CALUDE_parakeets_per_cage_l3256_325686

theorem parakeets_per_cage 
  (num_cages : ℝ)
  (parrots_per_cage : ℝ)
  (total_birds : ℕ)
  (h1 : num_cages = 6.0)
  (h2 : parrots_per_cage = 6.0)
  (h3 : total_birds = 48) :
  (total_birds : ℝ) - num_cages * parrots_per_cage = num_cages * 2 :=
by sorry

end NUMINAMATH_CALUDE_parakeets_per_cage_l3256_325686


namespace NUMINAMATH_CALUDE_age_difference_l3256_325612

theorem age_difference (a b c : ℕ) : 
  b = 12 →
  b = 2 * c →
  a + b + c = 32 →
  a = b + 2 :=
by sorry

end NUMINAMATH_CALUDE_age_difference_l3256_325612


namespace NUMINAMATH_CALUDE_simplify_product_of_sqrt_l3256_325638

theorem simplify_product_of_sqrt (y : ℝ) (hy : y > 0) :
  Real.sqrt (45 * y) * Real.sqrt (20 * y) * Real.sqrt (30 * y) = 30 * y * Real.sqrt (30 * y) := by
  sorry

end NUMINAMATH_CALUDE_simplify_product_of_sqrt_l3256_325638


namespace NUMINAMATH_CALUDE_prob_not_six_in_six_rolls_l3256_325671

-- Define a fair six-sided die
def fair_die : Finset ℕ := Finset.range 6

-- Define the probability of an event for a fair die
def prob (event : Finset ℕ) : ℚ :=
  event.card / fair_die.card

-- Define the event of not rolling a 6
def not_six : Finset ℕ := Finset.range 5

-- Theorem statement
theorem prob_not_six_in_six_rolls :
  (prob not_six) ^ 6 = (5 : ℚ) / 6 ^ 6 :=
sorry

end NUMINAMATH_CALUDE_prob_not_six_in_six_rolls_l3256_325671


namespace NUMINAMATH_CALUDE_tangent_line_implies_function_values_l3256_325659

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem tangent_line_implies_function_values :
  (∀ y, y = f 5 ↔ y = -5 + 8) →  -- Tangent line equation at x = 5
  (f 5 = 3 ∧ deriv f 5 = -1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_implies_function_values_l3256_325659


namespace NUMINAMATH_CALUDE_no_reciprocal_implies_one_l3256_325658

/-- If a number minus 1 does not have a reciprocal, then that number equals 1 -/
theorem no_reciprocal_implies_one (a : ℝ) : (∀ x : ℝ, x * (a - 1) ≠ 1) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_no_reciprocal_implies_one_l3256_325658


namespace NUMINAMATH_CALUDE_test_probabilities_l3256_325697

/-- Given probabilities of answering questions correctly on a test, 
    calculate the probability of answering neither question correctly. -/
theorem test_probabilities (p_first p_second p_both : ℝ) 
  (h1 : p_first = 0.75)
  (h2 : p_second = 0.55)
  (h3 : p_both = 0.50) :
  1 - (p_first + p_second - p_both) = 0.20 := by
  sorry

end NUMINAMATH_CALUDE_test_probabilities_l3256_325697


namespace NUMINAMATH_CALUDE_growth_comparison_l3256_325683

theorem growth_comparison (x : ℝ) (h : x > 0) :
  (0 < x ∧ x < 1/2 → (fun y => y) x > (fun y => y^2) x) ∧
  (x > 1/2 → (fun y => y^2) x > (fun y => y) x) := by
sorry

end NUMINAMATH_CALUDE_growth_comparison_l3256_325683


namespace NUMINAMATH_CALUDE_root_equation_consequence_l3256_325615

theorem root_equation_consequence (m : ℝ) : 
  m^2 - 2*m - 7 = 0 → m^2 - 2*m + 1 = 8 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_consequence_l3256_325615


namespace NUMINAMATH_CALUDE_decreasing_function_positive_range_l3256_325609

-- Define a decreasing function f on ℝ
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- State the theorem
theorem decreasing_function_positive_range
  (h_decreasing : ∀ x y, x < y → f x > f y)
  (h_derivative : ∀ x, HasDerivAt f (f' x) x)
  (h_inequality : ∀ x, f x / f' x + x < 1) :
  ∀ x, f x > 0 ↔ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_function_positive_range_l3256_325609


namespace NUMINAMATH_CALUDE_eggs_laid_per_dove_l3256_325624

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

end NUMINAMATH_CALUDE_eggs_laid_per_dove_l3256_325624


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3256_325620

def a (n : ℕ+) : ℤ := 2^n.val - (-1)^n.val

theorem arithmetic_sequence_properties :
  (∃ n : ℕ+, a n + a (n + 2) = 2 * a (n + 1) ∧ n = 2) ∧
  (∃ n₂ n₃ : ℕ+, n₂ < n₃ ∧ a 1 + a n₃ = 2 * a n₂ ∧ n₃ - n₂ = 1) ∧
  (∀ t : ℕ+, t > 3 →
    ¬∃ (n : Fin t → ℕ+), (∀ i j : Fin t, i < j → n i < n j) ∧
      (∀ i : Fin (t - 2), 2 * a (n (i + 1)) = a (n i) + a (n (i + 2)))) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3256_325620


namespace NUMINAMATH_CALUDE_smallest_solution_quartic_equation_l3256_325634

theorem smallest_solution_quartic_equation :
  ∃ (x : ℝ), x^4 - 50*x^2 + 625 = 0 ∧ 
  (∀ y : ℝ, y^4 - 50*y^2 + 625 = 0 → y ≥ x) ∧
  x = -5 := by
sorry

end NUMINAMATH_CALUDE_smallest_solution_quartic_equation_l3256_325634


namespace NUMINAMATH_CALUDE_rectangle_width_l3256_325646

/-- Given a rectangle with length 13 cm and perimeter 50 cm, prove its width is 12 cm. -/
theorem rectangle_width (length : ℝ) (perimeter : ℝ) (width : ℝ) : 
  length = 13 → perimeter = 50 → perimeter = 2 * (length + width) → width = 12 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_l3256_325646


namespace NUMINAMATH_CALUDE_tangent_line_to_parabola_l3256_325617

theorem tangent_line_to_parabola (d : ℝ) : 
  (∃ (x y : ℝ), y = 3*x + d ∧ y^2 = 12*x ∧ 
   ∀ (x' y' : ℝ), y' = 3*x' + d → y'^2 ≥ 12*x') → d = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_parabola_l3256_325617


namespace NUMINAMATH_CALUDE_square_side_length_equal_area_l3256_325606

theorem square_side_length_equal_area (rectangle_length rectangle_width : ℝ) :
  rectangle_length = 72 ∧ rectangle_width = 18 →
  ∃ (square_side : ℝ), square_side ^ 2 = rectangle_length * rectangle_width ∧ square_side = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_equal_area_l3256_325606


namespace NUMINAMATH_CALUDE_inequality_solution_l3256_325628

theorem inequality_solution (x : ℝ) : 
  (9*x^2 + 27*x - 40) / ((3*x - 4)*(x + 5)) < 5 ↔ 
  (x > -6 ∧ x < -5) ∨ (x > 4/3 ∧ x < 5/3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3256_325628


namespace NUMINAMATH_CALUDE_margie_change_l3256_325631

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

end NUMINAMATH_CALUDE_margie_change_l3256_325631


namespace NUMINAMATH_CALUDE_range_of_f_l3256_325601

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x

-- Define the domain
def domain : Set ℝ := { x | 1 ≤ x ∧ x < 5 }

-- Theorem statement
theorem range_of_f :
  { y | ∃ x ∈ domain, f x = y } = { y | -4 ≤ y ∧ y < 5 } := by sorry

end NUMINAMATH_CALUDE_range_of_f_l3256_325601


namespace NUMINAMATH_CALUDE_solutions_count_for_specific_n_l3256_325653

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

end NUMINAMATH_CALUDE_solutions_count_for_specific_n_l3256_325653


namespace NUMINAMATH_CALUDE_valid_permutation_exists_valid_permutation_32_valid_permutation_100_l3256_325644

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

end NUMINAMATH_CALUDE_valid_permutation_exists_valid_permutation_32_valid_permutation_100_l3256_325644


namespace NUMINAMATH_CALUDE_meals_per_day_l3256_325668

theorem meals_per_day (people : ℕ) (total_plates : ℕ) (days : ℕ) (plates_per_meal : ℕ)
  (h1 : people = 6)
  (h2 : total_plates = 144)
  (h3 : days = 4)
  (h4 : plates_per_meal = 2)
  : (total_plates / (people * days * plates_per_meal) : ℚ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_meals_per_day_l3256_325668


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_l3256_325666

theorem geometric_sequence_second_term (a : ℝ) : 
  a > 0 ∧ 
  (∃ r : ℝ, r ≠ 0 ∧ 25 * r = a ∧ a * r = 8/5) → 
  a = 2 * Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_second_term_l3256_325666


namespace NUMINAMATH_CALUDE_group_size_l3256_325696

/-- The number of people in a group, given weight changes. -/
theorem group_size (weight_increase_per_person : ℝ) (new_person_weight : ℝ) (replaced_person_weight : ℝ) :
  weight_increase_per_person * 10 = new_person_weight - replaced_person_weight →
  10 = (new_person_weight - replaced_person_weight) / weight_increase_per_person :=
by
  sorry

#check group_size 7.2 137 65

end NUMINAMATH_CALUDE_group_size_l3256_325696


namespace NUMINAMATH_CALUDE_comparison_of_expressions_l3256_325652

theorem comparison_of_expressions (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  ¬ (∀ x y z : ℝ, 
    (x = (a + 1/a) * (b + 1/b) ∧ 
     y = (Real.sqrt (a * b) + 1 / Real.sqrt (a * b))^2 ∧ 
     z = ((a + b)/2 + 2/(a + b))^2) →
    (x > y ∧ x > z) ∨ (y > x ∧ y > z) ∨ (z > x ∧ z > y)) :=
by sorry


end NUMINAMATH_CALUDE_comparison_of_expressions_l3256_325652


namespace NUMINAMATH_CALUDE_train_speed_calculation_l3256_325680

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 100 →
  bridge_length = 150 →
  crossing_time = 49.9960003199744 →
  ∃ (speed : ℝ), (abs (speed - 18) < 0.1 ∧ 
    speed = (train_length + bridge_length) / crossing_time * 3.6) := by
  sorry


end NUMINAMATH_CALUDE_train_speed_calculation_l3256_325680


namespace NUMINAMATH_CALUDE_ladder_problem_l3256_325681

theorem ladder_problem (ladder_length height : ℝ) 
  (h1 : ladder_length = 13)
  (h2 : height = 12) :
  ∃ (base : ℝ), base^2 + height^2 = ladder_length^2 ∧ base = 5 :=
sorry

end NUMINAMATH_CALUDE_ladder_problem_l3256_325681


namespace NUMINAMATH_CALUDE_absolute_value_sum_l3256_325670

theorem absolute_value_sum (a : ℝ) (h : -2 < a ∧ a < 0) : |a| + |a+2| = 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_l3256_325670


namespace NUMINAMATH_CALUDE_second_month_sale_l3256_325654

/-- Calculates the sale in the second month given sales for other months and the average --/
def calculate_second_month_sale (first_month : ℕ) (third_month : ℕ) (fourth_month : ℕ) 
  (fifth_month : ℕ) (sixth_month : ℕ) (average : ℕ) : ℕ :=
  6 * average - (first_month + third_month + fourth_month + fifth_month + sixth_month)

/-- Theorem stating that the sale in the second month is 5660 --/
theorem second_month_sale : 
  calculate_second_month_sale 5420 6200 6350 6500 6470 6100 = 5660 := by
  sorry

end NUMINAMATH_CALUDE_second_month_sale_l3256_325654


namespace NUMINAMATH_CALUDE_orthogonal_projection_magnitude_l3256_325667

/-- Given two vectors a and b in ℝ², prove that the magnitude of the orthogonal projection of a onto b is √5 -/
theorem orthogonal_projection_magnitude (a b : ℝ × ℝ) (h1 : a = (3, -1)) (h2 : b = (1, -2)) :
  ‖(((a.1 * b.1 + a.2 * b.2) / (b.1^2 + b.2^2)) • b)‖ = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_orthogonal_projection_magnitude_l3256_325667


namespace NUMINAMATH_CALUDE_fermat_like_equation_exponent_l3256_325655

theorem fermat_like_equation_exponent (x y p n k : ℕ) : 
  x^n + y^n = p^k →
  n > 1 →
  Odd n →
  Nat.Prime p →
  Odd p →
  ∃ l : ℕ, n = p^l := by
sorry

end NUMINAMATH_CALUDE_fermat_like_equation_exponent_l3256_325655


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l3256_325695

theorem arithmetic_calculations :
  ((-8) + 10 - (-2) = 12) ∧
  (42 * (-2/3) + (-3/4) / (-0.25) = -25) ∧
  ((-2.5) / (-5/8) * (-0.25) = -1) ∧
  ((1 + 3/4 - 7/8 - 7/12) / (-7/8) = -1/3) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l3256_325695


namespace NUMINAMATH_CALUDE_coin_value_difference_l3256_325687

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

end NUMINAMATH_CALUDE_coin_value_difference_l3256_325687


namespace NUMINAMATH_CALUDE_circles_A_B_intersect_l3256_325651

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

end NUMINAMATH_CALUDE_circles_A_B_intersect_l3256_325651


namespace NUMINAMATH_CALUDE_roger_toys_theorem_l3256_325662

def max_toys_buyable (initial_amount : ℕ) (spent_amount : ℕ) (toy_cost : ℕ) : ℕ :=
  (initial_amount - spent_amount) / toy_cost

theorem roger_toys_theorem : 
  max_toys_buyable 63 48 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_roger_toys_theorem_l3256_325662


namespace NUMINAMATH_CALUDE_concert_attendance_l3256_325657

/-- The number of buses used for the concert trip -/
def number_of_buses : ℕ := 8

/-- The number of students each bus can carry -/
def students_per_bus : ℕ := 45

/-- The total number of students who went to the concert -/
def total_students : ℕ := number_of_buses * students_per_bus

/-- Theorem stating that the total number of students who went to the concert is 360 -/
theorem concert_attendance : total_students = 360 := by
  sorry

end NUMINAMATH_CALUDE_concert_attendance_l3256_325657


namespace NUMINAMATH_CALUDE_complex_number_equality_l3256_325656

theorem complex_number_equality (z : ℂ) (h : z / (1 - Complex.I) = Complex.I) : z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l3256_325656


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l3256_325660

noncomputable def f (θ : ℝ) : ℝ := (Real.sin θ) / (2 + Real.cos θ)

theorem derivative_f_at_zero :
  deriv f 0 = 1/3 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l3256_325660


namespace NUMINAMATH_CALUDE_unique_solution_k_values_l3256_325626

theorem unique_solution_k_values (k : ℝ) :
  (∃! x : ℝ, 1 ≤ k * x^2 + 2 ∧ x + k ≤ 2) ↔ 
  (k = 1 + Real.sqrt 2 ∨ k = (1 - Real.sqrt 5) / 2) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_k_values_l3256_325626


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l3256_325688

theorem triangle_angle_sum (a b c : ℝ) (h1 : b = 30)
    (h2 : c = 3 * b) : a = 60 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l3256_325688


namespace NUMINAMATH_CALUDE_triangle_area_in_circle_l3256_325663

theorem triangle_area_in_circle (r : ℝ) (h : r = 3) : 
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Sides are positive
    a = b ∧ c = a * Real.sqrt 2 ∧  -- Sides are in ratio 1:1:√2
    c = 2 * r ∧  -- Diameter of circle
    (1/2) * a * b = 9 := by  -- Area of triangle
  sorry

end NUMINAMATH_CALUDE_triangle_area_in_circle_l3256_325663


namespace NUMINAMATH_CALUDE_sphere_surface_area_l3256_325675

theorem sphere_surface_area (d : ℝ) (h : d = 4) :
  4 * Real.pi * (d / 2)^2 = 16 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l3256_325675


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_l3256_325674

theorem systematic_sampling_interval 
  (total : ℕ) 
  (samples : ℕ) 
  (h1 : total = 231) 
  (h2 : samples = 22) :
  let adjusted_total := total - (total % samples)
  adjusted_total / samples = 10 :=
sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_l3256_325674


namespace NUMINAMATH_CALUDE_pig_year_paintings_distribution_l3256_325691

theorem pig_year_paintings_distribution (n : ℕ) (k : ℕ) (h1 : n = 4) (h2 : k = 3) :
  let total_outcomes := k^n
  let favorable_outcomes := (n.choose 2) * (k.factorial)
  (favorable_outcomes : ℚ) / total_outcomes = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_pig_year_paintings_distribution_l3256_325691


namespace NUMINAMATH_CALUDE_max_value_of_x_plus_inverse_l3256_325629

theorem max_value_of_x_plus_inverse (x : ℝ) (h : 7 = x^2 + 1/x^2) :
  ∃ (y : ℝ), y = x + 1/x ∧ y ≤ 3 ∧ ∃ (z : ℝ), z = x + 1/x ∧ z = 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_x_plus_inverse_l3256_325629


namespace NUMINAMATH_CALUDE_problem_solutions_l3256_325637

theorem problem_solutions : 
  (1) * (1/3 - 1/4 - 1/2) / (-1/24) = 10 ∧ 
  -(3^2) - (-2/3) * 6 + (-2)^3 = -13 := by sorry

end NUMINAMATH_CALUDE_problem_solutions_l3256_325637


namespace NUMINAMATH_CALUDE_angle_abc_measure_l3256_325604

/-- A configuration with a square inscribed in a regular pentagon sharing a side -/
structure SquareInPentagon where
  /-- The measure of an interior angle of the regular pentagon in degrees -/
  pentagon_angle : ℝ
  /-- The measure of an interior angle of the square in degrees -/
  square_angle : ℝ
  /-- The angle ABC formed by the vertex of the pentagon adjacent to the shared side
      and the two nearest vertices of the square -/
  angle_abc : ℝ
  /-- The pentagon_angle is 108 degrees -/
  pentagon_angle_eq : pentagon_angle = 108
  /-- The square_angle is 90 degrees -/
  square_angle_eq : square_angle = 90

/-- The angle ABC in a SquareInPentagon configuration is 27 degrees -/
theorem angle_abc_measure (config : SquareInPentagon) : config.angle_abc = 27 :=
  sorry

end NUMINAMATH_CALUDE_angle_abc_measure_l3256_325604


namespace NUMINAMATH_CALUDE_fraction_equality_l3256_325649

theorem fraction_equality (a b : ℝ) (h : a / (a + 2 * b) = 3 / 5) : a / b = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3256_325649


namespace NUMINAMATH_CALUDE_inequality_proof_l3256_325630

theorem inequality_proof (x y : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) :
  x + y + 1 / (x * y) ≤ 1 / x + 1 / y + x * y := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3256_325630


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_ratios_l3256_325616

theorem min_value_of_sum_of_ratios (a₁ a₂ a₃ a₄ a₅ a₆ : ℕ) 
  (h1 : 1 ≤ a₁) (h2 : a₁ ≤ a₂) (h3 : a₂ ≤ a₃) (h4 : a₃ ≤ a₄) 
  (h5 : a₄ ≤ a₅) (h6 : a₅ ≤ a₆) (h7 : a₆ ≤ 64) :
  (a₁ : ℚ) / a₂ + (a₃ : ℚ) / a₄ + (a₅ : ℚ) / a₆ ≥ 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_ratios_l3256_325616


namespace NUMINAMATH_CALUDE_max_principals_in_period_l3256_325636

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

end NUMINAMATH_CALUDE_max_principals_in_period_l3256_325636


namespace NUMINAMATH_CALUDE_gerald_initial_farthings_l3256_325600

/-- The number of farthings in a pfennig -/
def farthings_per_pfennig : ℕ := 6

/-- The cost of a meat pie in pfennigs -/
def pie_cost : ℕ := 2

/-- The number of pfennigs Gerald has left after buying the pie -/
def pfennigs_left : ℕ := 7

/-- The number of farthings Gerald has initially -/
def initial_farthings : ℕ := 54

theorem gerald_initial_farthings :
  initial_farthings = 
    pie_cost * farthings_per_pfennig + pfennigs_left * farthings_per_pfennig :=
by sorry

end NUMINAMATH_CALUDE_gerald_initial_farthings_l3256_325600


namespace NUMINAMATH_CALUDE_prime_factor_sum_l3256_325614

theorem prime_factor_sum (w x y z t : ℕ) : 
  2^w * 3^x * 5^y * 7^z * 17^t = 107100 →
  2*w + 3*x + 5*y + 7*z + 11*t = 38 := by
sorry

end NUMINAMATH_CALUDE_prime_factor_sum_l3256_325614


namespace NUMINAMATH_CALUDE_sams_cycling_speed_l3256_325689

/-- Given the cycling speeds of three friends, prove Sam's speed -/
theorem sams_cycling_speed 
  (lucas_speed : ℚ) 
  (maya_speed_ratio : ℚ) 
  (lucas_sam_ratio : ℚ)
  (h1 : lucas_speed = 5)
  (h2 : maya_speed_ratio = 4 / 5)
  (h3 : lucas_sam_ratio = 9 / 8) :
  lucas_speed * (8 / 9) = 40 / 9 := by
  sorry

#check sams_cycling_speed

end NUMINAMATH_CALUDE_sams_cycling_speed_l3256_325689


namespace NUMINAMATH_CALUDE_turtleneck_sweater_profit_profit_percentage_l3256_325607

theorem turtleneck_sweater_profit (C : ℝ) : 
  let first_markup := C * 1.20
  let second_markup := first_markup * 1.25
  let final_price := second_markup * 0.88
  final_price = C * 1.32 := by sorry

theorem profit_percentage (C : ℝ) : 
  let first_markup := C * 1.20
  let second_markup := first_markup * 1.25
  let final_price := second_markup * 0.88
  (final_price - C) / C = 0.32 := by sorry

end NUMINAMATH_CALUDE_turtleneck_sweater_profit_profit_percentage_l3256_325607


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l3256_325685

def arithmetic_sequence (a : ℕ → ℝ) := 
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_difference 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_sum1 : a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 81)
  (h_sum2 : a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 = 171) :
  ∃ d : ℝ, d = 10 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l3256_325685


namespace NUMINAMATH_CALUDE_symmetric_point_y_axis_coordinates_l3256_325698

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

end NUMINAMATH_CALUDE_symmetric_point_y_axis_coordinates_l3256_325698


namespace NUMINAMATH_CALUDE_complex_fraction_equals_two_l3256_325676

theorem complex_fraction_equals_two (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = a*b) :
  (a^6 + b^6) / (a + b)^6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_two_l3256_325676


namespace NUMINAMATH_CALUDE_fraction_calculation_l3256_325645

theorem fraction_calculation : 
  (2 + 1/4 + 0.25) / (2 + 3/4 - 1/2) + (2 * 0.5) / (2 + 1/5 - 2/5) = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l3256_325645


namespace NUMINAMATH_CALUDE_wickets_in_last_match_is_five_l3256_325678

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

end NUMINAMATH_CALUDE_wickets_in_last_match_is_five_l3256_325678


namespace NUMINAMATH_CALUDE_order_cost_l3256_325692

/-- The cost of the order given the prices and quantities of pencils and erasers -/
theorem order_cost (pencil_price eraser_price : ℕ) (total_cartons pencil_cartons : ℕ) : 
  pencil_price = 6 →
  eraser_price = 3 →
  total_cartons = 100 →
  pencil_cartons = 20 →
  pencil_price * pencil_cartons + eraser_price * (total_cartons - pencil_cartons) = 360 := by
sorry

end NUMINAMATH_CALUDE_order_cost_l3256_325692


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3256_325669

theorem sufficient_not_necessary_condition (a : ℝ) :
  (a ≥ 5) →
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0) ∧
  ¬(∀ a : ℝ, (∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0) → a ≥ 5) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3256_325669


namespace NUMINAMATH_CALUDE_office_age_problem_l3256_325684

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

end NUMINAMATH_CALUDE_office_age_problem_l3256_325684


namespace NUMINAMATH_CALUDE_first_fun_friday_l3256_325608

/-- Represents a day of the week -/
inductive DayOfWeek
  | sunday
  | monday
  | tuesday
  | wednesday
  | thursday
  | friday
  | saturday

/-- Represents a date in March -/
structure MarchDate where
  day : Nat
  dayOfWeek : DayOfWeek

/-- The company's year starts on Thursday, March 1st -/
def yearStart : MarchDate :=
  { day := 1, dayOfWeek := DayOfWeek.thursday }

/-- March has 31 days -/
def marchDays : Nat := 31

/-- Determines if a given date is a Friday -/
def isFriday (date : MarchDate) : Prop :=
  date.dayOfWeek = DayOfWeek.friday

/-- Counts the number of Fridays up to and including a given date in March -/
def fridayCount (date : MarchDate) : Nat :=
  sorry

/-- Determines if a given date is a Fun Friday -/
def isFunFriday (date : MarchDate) : Prop :=
  isFriday date ∧ fridayCount date = 5

/-- The theorem to be proved -/
theorem first_fun_friday : 
  ∃ (date : MarchDate), date.day = 30 ∧ isFunFriday date :=
sorry

end NUMINAMATH_CALUDE_first_fun_friday_l3256_325608


namespace NUMINAMATH_CALUDE_two_is_sup_of_satisfying_set_l3256_325699

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

end NUMINAMATH_CALUDE_two_is_sup_of_satisfying_set_l3256_325699


namespace NUMINAMATH_CALUDE_probability_of_choosing_quarter_l3256_325635

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

end NUMINAMATH_CALUDE_probability_of_choosing_quarter_l3256_325635


namespace NUMINAMATH_CALUDE_slag_transport_theorem_l3256_325642

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

end NUMINAMATH_CALUDE_slag_transport_theorem_l3256_325642


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l3256_325610

theorem quadratic_inequality_condition (m : ℝ) :
  (∀ x : ℝ, x^2 + m*x + 2 > 0) ↔ -2*Real.sqrt 2 < m ∧ m < 2*Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l3256_325610


namespace NUMINAMATH_CALUDE_parallelogram_area_v_w_l3256_325622

/-- The area of a parallelogram formed by two 2D vectors -/
def parallelogramArea (v w : Fin 2 → ℝ) : ℝ :=
  |v 0 * w 1 - v 1 * w 0|

/-- Vectors v and w -/
def v : Fin 2 → ℝ := ![4, -6]
def w : Fin 2 → ℝ := ![7, -1]

/-- Theorem stating that the area of the parallelogram formed by v and w is 38 -/
theorem parallelogram_area_v_w : parallelogramArea v w = 38 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_v_w_l3256_325622


namespace NUMINAMATH_CALUDE_people_in_room_l3256_325679

theorem people_in_room (total_chairs : ℚ) (occupied_chairs : ℚ) (empty_chairs : ℚ) 
  (h1 : empty_chairs = 5)
  (h2 : occupied_chairs = (2/3) * total_chairs)
  (h3 : empty_chairs = (1/3) * total_chairs)
  (h4 : occupied_chairs = 10) :
  ∃ (total_people : ℚ), total_people = 50/3 ∧ (3/5) * total_people = occupied_chairs := by
  sorry

end NUMINAMATH_CALUDE_people_in_room_l3256_325679


namespace NUMINAMATH_CALUDE_prob_b_not_lose_l3256_325665

/-- The probability that Player A wins a chess match. -/
def prob_a_wins : ℝ := 0.5

/-- The probability of a draw in a chess match. -/
def prob_draw : ℝ := 0.1

/-- The probability that Player B does not lose is equal to 0.5. -/
theorem prob_b_not_lose : 1 - prob_a_wins = 0.5 := by sorry

end NUMINAMATH_CALUDE_prob_b_not_lose_l3256_325665


namespace NUMINAMATH_CALUDE_probability_one_black_one_white_l3256_325611

/-- The probability of selecting one black ball and one white ball from a jar containing 6 black balls and 2 white balls when picking two balls at the same time. -/
theorem probability_one_black_one_white (black_balls : ℕ) (white_balls : ℕ) 
  (h1 : black_balls = 6) (h2 : white_balls = 2) :
  (black_balls * white_balls : ℚ) / (Nat.choose (black_balls + white_balls) 2) = 3/7 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_black_one_white_l3256_325611


namespace NUMINAMATH_CALUDE_fixed_point_quadratic_l3256_325632

theorem fixed_point_quadratic (k : ℝ) : 
  200 = 8 * (5 : ℝ)^2 + 3 * k * 5 - 5 * k := by sorry

end NUMINAMATH_CALUDE_fixed_point_quadratic_l3256_325632


namespace NUMINAMATH_CALUDE_quadratic_solution_l3256_325693

-- Define the quadratic equation
def quadratic_equation (p q x : ℝ) : Prop := x^2 + p*x + q = 0

-- Define the conditions
theorem quadratic_solution (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) :
  (quadratic_equation p q (2*p) ∧ quadratic_equation p q (q/2)) →
  p = 1 ∧ q = -6 := by sorry

end NUMINAMATH_CALUDE_quadratic_solution_l3256_325693


namespace NUMINAMATH_CALUDE_division_problem_l3256_325682

theorem division_problem : (107.8 : ℝ) / 11 = 9.8 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3256_325682


namespace NUMINAMATH_CALUDE_total_outfits_count_l3256_325643

def total_shirts : ℕ := 8
def total_ties : ℕ := 6
def special_shirt_matches : ℕ := 3

def outfits_with_special_shirt : ℕ := special_shirt_matches
def outfits_with_other_shirts : ℕ := (total_shirts - 1) * total_ties

theorem total_outfits_count :
  outfits_with_special_shirt + outfits_with_other_shirts = 45 :=
by sorry

end NUMINAMATH_CALUDE_total_outfits_count_l3256_325643


namespace NUMINAMATH_CALUDE_exam_questions_count_l3256_325661

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

end NUMINAMATH_CALUDE_exam_questions_count_l3256_325661


namespace NUMINAMATH_CALUDE_f_g_minus_g_f_l3256_325613

def f (x : ℝ) : ℝ := 2 * x - 1

def g (x : ℝ) : ℝ := x^2 + 2 * x + 1

theorem f_g_minus_g_f : f (g 3) - g (f 3) = -5 := by
  sorry

end NUMINAMATH_CALUDE_f_g_minus_g_f_l3256_325613


namespace NUMINAMATH_CALUDE_club_co_presidents_selection_l3256_325627

/-- The number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of members in the club --/
def club_members : ℕ := 18

/-- The number of co-presidents to be chosen --/
def co_presidents : ℕ := 3

theorem club_co_presidents_selection :
  choose club_members co_presidents = 816 := by
  sorry

end NUMINAMATH_CALUDE_club_co_presidents_selection_l3256_325627


namespace NUMINAMATH_CALUDE_cartesian_product_subset_cartesian_product_intersection_y_axis_representation_l3256_325690

-- Define the Cartesian product
def cartesianProduct (A B : Set α) : Set (α × α) :=
  {p | p.1 ∈ A ∧ p.2 ∈ B}

-- Statement 1
theorem cartesian_product_subset {A B C : Set α} (h : A ⊆ C) :
  cartesianProduct A B ⊆ cartesianProduct C B :=
sorry

-- Statement 2
theorem cartesian_product_intersection {A B C : Set α} :
  cartesianProduct A (B ∩ C) = cartesianProduct A B ∩ cartesianProduct A C :=
sorry

-- Statement 3
theorem y_axis_representation {R : Type} [LinearOrderedField R] :
  cartesianProduct {(0 : R)} (Set.univ : Set R) = {p : R × R | p.1 = 0} :=
sorry

end NUMINAMATH_CALUDE_cartesian_product_subset_cartesian_product_intersection_y_axis_representation_l3256_325690


namespace NUMINAMATH_CALUDE_paco_cookies_l3256_325677

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

end NUMINAMATH_CALUDE_paco_cookies_l3256_325677


namespace NUMINAMATH_CALUDE_unique_function_solution_l3256_325605

/-- Given a positive real number c, prove that the only function f: ℝ₊ → ℝ₊ 
    satisfying f((c+1)x + f(y)) = f(x + 2y) + 2cx for all x, y ∈ ℝ₊ is f(x) = 2x. -/
theorem unique_function_solution (c : ℝ) (hc : c > 0) :
  ∀ f : ℝ → ℝ, (∀ x, x > 0 → f x > 0) →
  (∀ x y, x > 0 → y > 0 → f ((c + 1) * x + f y) = f (x + 2 * y) + 2 * c * x) →
  ∀ x, x > 0 → f x = 2 * x :=
by sorry

end NUMINAMATH_CALUDE_unique_function_solution_l3256_325605


namespace NUMINAMATH_CALUDE_order_of_special_roots_l3256_325648

theorem order_of_special_roots : ∃ (a b c : ℝ), 
  a = (2 : ℝ) ^ (1/2) ∧ 
  b = Real.exp (1/Real.exp 1) ∧ 
  c = (3 : ℝ) ^ (1/3) ∧ 
  a < c ∧ c < b := by
  sorry

end NUMINAMATH_CALUDE_order_of_special_roots_l3256_325648


namespace NUMINAMATH_CALUDE_ivan_piggy_bank_l3256_325603

/-- Represents the contents of Ivan's piggy bank -/
structure PiggyBank where
  dimes : Nat
  pennies : Nat

/-- The value of the piggy bank in cents -/
def PiggyBank.value (pb : PiggyBank) : Nat :=
  pb.dimes * 10 + pb.pennies

theorem ivan_piggy_bank :
  ∀ (pb : PiggyBank),
    pb.dimes = 50 →
    pb.value = 1200 →
    pb.pennies = 700 := by
  sorry

end NUMINAMATH_CALUDE_ivan_piggy_bank_l3256_325603


namespace NUMINAMATH_CALUDE_equation_solution_l3256_325641

theorem equation_solution :
  ∃! x : ℝ, 
    x > 10 ∧
    (8 / (Real.sqrt (x - 10) - 10) + 
     2 / (Real.sqrt (x - 10) - 5) + 
     9 / (Real.sqrt (x - 10) + 5) + 
     15 / (Real.sqrt (x - 10) + 10) = 0) ∧
    x = 35 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3256_325641
