import Mathlib

namespace fencing_cost_l927_92750

/-- Given a rectangular field with sides in ratio 3:4 and area 10092 sq. m,
    prove that the cost of fencing at 25 paise per metre is 101.5 rupees. -/
theorem fencing_cost (length width : ℝ) (h1 : length / width = 3 / 4)
  (h2 : length * width = 10092) : 
  (2 * (length + width) * 25 / 100) = 101.5 :=
by sorry

end fencing_cost_l927_92750


namespace carols_mother_carrots_l927_92794

theorem carols_mother_carrots : 
  ∀ (carol_carrots good_carrots bad_carrots total_carrots mother_carrots : ℕ),
    carol_carrots = 29 →
    good_carrots = 38 →
    bad_carrots = 7 →
    total_carrots = good_carrots + bad_carrots →
    mother_carrots = total_carrots - carol_carrots →
    mother_carrots = 16 := by
  sorry

end carols_mother_carrots_l927_92794


namespace minimum_words_to_learn_l927_92789

theorem minimum_words_to_learn (total_words : ℕ) (required_percentage : ℚ) : 
  total_words = 600 → required_percentage = 90 / 100 → 
  ∃ (min_words : ℕ), min_words * 100 ≥ total_words * required_percentage ∧
    ∀ (n : ℕ), n * 100 ≥ total_words * required_percentage → n ≥ min_words :=
by sorry

end minimum_words_to_learn_l927_92789


namespace intersection_points_count_l927_92762

/-- Definition of the three lines -/
def line1 (x y : ℝ) : Prop := 2 * y - 3 * x = 4
def line2 (x y : ℝ) : Prop := 5 * x + y = 1
def line3 (x y : ℝ) : Prop := 6 * x - 4 * y = 2

/-- A point lies on at least two of the three lines -/
def point_on_two_lines (x y : ℝ) : Prop :=
  (line1 x y ∧ line2 x y) ∨ (line1 x y ∧ line3 x y) ∨ (line2 x y ∧ line3 x y)

/-- The main theorem to prove -/
theorem intersection_points_count :
  ∃ (p1 p2 : ℝ × ℝ), p1 ≠ p2 ∧
    point_on_two_lines p1.1 p1.2 ∧
    point_on_two_lines p2.1 p2.2 ∧
    ∀ (p : ℝ × ℝ), point_on_two_lines p.1 p.2 → p = p1 ∨ p = p2 := by
  sorry


end intersection_points_count_l927_92762


namespace bus_rental_combinations_l927_92736

theorem bus_rental_combinations :
  let total_people : ℕ := 482
  let large_bus_capacity : ℕ := 42
  let medium_bus_capacity : ℕ := 20
  ∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ =>
    p.1 * large_bus_capacity + p.2 * medium_bus_capacity = total_people
  ) (Finset.product (Finset.range (total_people + 1)) (Finset.range (total_people + 1)))).card ∧ n = 2 :=
by sorry

end bus_rental_combinations_l927_92736


namespace empire_state_building_race_l927_92755

-- Define the total number of steps
def total_steps : ℕ := 1576

-- Define the total time in seconds
def total_time_seconds : ℕ := 11 * 60 + 57

-- Define the function to calculate steps per minute
def steps_per_minute (steps : ℕ) (time_seconds : ℕ) : ℚ :=
  (steps : ℚ) / ((time_seconds : ℚ) / 60)

-- Theorem statement
theorem empire_state_building_race :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ 
  |steps_per_minute total_steps total_time_seconds - 130| < ε :=
sorry

end empire_state_building_race_l927_92755


namespace infinitely_many_integers_with_zero_padic_valuation_mod_d_l927_92711

/-- The p-adic valuation of n! -/
def ν (p : Nat) (n : Nat) : Nat := sorry

theorem infinitely_many_integers_with_zero_padic_valuation_mod_d 
  (d : Nat) (primes : Finset Nat) (h_d : d > 0) (h_primes : ∀ p ∈ primes, Nat.Prime p) :
  ∃ (S : Set Nat), Set.Infinite S ∧ 
    ∀ n ∈ S, ∀ p ∈ primes, (ν p n) % d = 0 := by
  sorry

end infinitely_many_integers_with_zero_padic_valuation_mod_d_l927_92711


namespace cubic_factorization_l927_92799

theorem cubic_factorization (x : ℝ) : x^3 - 9*x = x*(x+3)*(x-3) := by
  sorry

end cubic_factorization_l927_92799


namespace initial_geese_count_l927_92798

theorem initial_geese_count (G : ℕ) : 
  (G / 2 + 4 = 12) → G = 16 := by
  sorry

end initial_geese_count_l927_92798


namespace damien_tall_cupboard_glasses_l927_92757

/-- Represents the number of glasses in different cupboards --/
structure Cupboards where
  tall : ℕ
  wide : ℕ
  narrow : ℕ

/-- The setup of Damien's glass collection --/
def damien_cupboards : Cupboards where
  tall := 5
  wide := 10
  narrow := 10

/-- Theorem stating the number of glasses in Damien's tall cupboard --/
theorem damien_tall_cupboard_glasses :
  ∃ (c : Cupboards), 
    c.wide = 2 * c.tall ∧ 
    c.narrow = 10 ∧ 
    15 % 3 = 0 ∧ 
    c = damien_cupboards :=
by
  sorry

#check damien_tall_cupboard_glasses

end damien_tall_cupboard_glasses_l927_92757


namespace number_of_boys_in_class_l927_92724

/-- Given the conditions of a class height measurement error, prove the number of boys in the class. -/
theorem number_of_boys_in_class 
  (n : ℕ) -- number of boys
  (initial_average : ℝ) -- initial average height
  (wrong_height : ℝ) -- wrongly recorded height
  (correct_height : ℝ) -- correct height of the boy
  (actual_average : ℝ) -- actual average height
  (h1 : initial_average = 182)
  (h2 : wrong_height = 166)
  (h3 : correct_height = 106)
  (h4 : actual_average = 180)
  (h5 : n * initial_average - wrong_height + correct_height = n * actual_average) :
  n = 30 := by
sorry

end number_of_boys_in_class_l927_92724


namespace probability_same_color_is_240_11970_l927_92777

/-- The number of red marbles in the bag -/
def red_marbles : ℕ := 6

/-- The number of white marbles in the bag -/
def white_marbles : ℕ := 7

/-- The number of blue marbles in the bag -/
def blue_marbles : ℕ := 8

/-- The total number of marbles in the bag -/
def total_marbles : ℕ := red_marbles + white_marbles + blue_marbles

/-- The number of marbles drawn -/
def marbles_drawn : ℕ := 4

/-- The probability of drawing four marbles of the same color -/
def probability_same_color : ℚ :=
  (red_marbles * (red_marbles - 1) * (red_marbles - 2) * (red_marbles - 3) +
   white_marbles * (white_marbles - 1) * (white_marbles - 2) * (white_marbles - 3) +
   blue_marbles * (blue_marbles - 1) * (blue_marbles - 2) * (blue_marbles - 3)) /
  (total_marbles * (total_marbles - 1) * (total_marbles - 2) * (total_marbles - 3))

theorem probability_same_color_is_240_11970 :
  probability_same_color = 240 / 11970 := by
  sorry

end probability_same_color_is_240_11970_l927_92777


namespace complex_multiplication_l927_92753

theorem complex_multiplication (z : ℂ) (h : z = 1 + I) : (1 + z) * z = 1 + 3*I := by sorry

end complex_multiplication_l927_92753


namespace mike_siblings_l927_92772

-- Define the characteristics
inductive EyeColor
| Blue
| Green

inductive HairColor
| Black
| Blonde

inductive Sport
| Soccer
| Basketball

-- Define a child's characteristics
structure ChildCharacteristics where
  eyeColor : EyeColor
  hairColor : HairColor
  favoriteSport : Sport

-- Define the children
def Lily : ChildCharacteristics := ⟨EyeColor.Green, HairColor.Black, Sport.Soccer⟩
def Mike : ChildCharacteristics := ⟨EyeColor.Blue, HairColor.Blonde, Sport.Basketball⟩
def Oliver : ChildCharacteristics := ⟨EyeColor.Blue, HairColor.Black, Sport.Soccer⟩
def Emma : ChildCharacteristics := ⟨EyeColor.Green, HairColor.Blonde, Sport.Basketball⟩
def Jacob : ChildCharacteristics := ⟨EyeColor.Blue, HairColor.Blonde, Sport.Soccer⟩
def Sophia : ChildCharacteristics := ⟨EyeColor.Green, HairColor.Blonde, Sport.Soccer⟩

-- Define a function to check if two children share at least one characteristic
def shareCharacteristic (child1 child2 : ChildCharacteristics) : Prop :=
  child1.eyeColor = child2.eyeColor ∨ 
  child1.hairColor = child2.hairColor ∨ 
  child1.favoriteSport = child2.favoriteSport

-- Define the theorem
theorem mike_siblings : 
  shareCharacteristic Mike Emma ∧ 
  shareCharacteristic Mike Jacob ∧ 
  shareCharacteristic Emma Jacob ∧
  ¬(shareCharacteristic Mike Lily ∧ shareCharacteristic Mike Oliver ∧ shareCharacteristic Mike Sophia) :=
by sorry

end mike_siblings_l927_92772


namespace ticket_cost_after_30_years_l927_92766

/-- The cost of a ticket to Mars after a given number of years, given an initial cost and a halving period --/
def ticket_cost (initial_cost : ℕ) (halving_period : ℕ) (years : ℕ) : ℕ :=
  initial_cost / (2 ^ (years / halving_period))

/-- Theorem stating that the cost of a ticket to Mars after 30 years is $125,000 --/
theorem ticket_cost_after_30_years :
  ticket_cost 1000000 10 30 = 125000 := by
  sorry

end ticket_cost_after_30_years_l927_92766


namespace smallest_a_unique_b_l927_92703

def is_all_real_roots (a b : ℝ) : Prop :=
  ∀ x : ℂ, x^4 - a*x^3 + b*x^2 - a*x + 1 = 0 → x.im = 0

theorem smallest_a_unique_b :
  ∃! (a : ℝ), a > 0 ∧
    (∃ (b : ℝ), b > 0 ∧ is_all_real_roots a b) ∧
    (∀ (a' : ℝ), 0 < a' ∧ a' < a →
      ¬∃ (b : ℝ), b > 0 ∧ is_all_real_roots a' b) ∧
    (∃! (b : ℝ), b > 0 ∧ is_all_real_roots a b) ∧
    a = 4 :=
sorry

end smallest_a_unique_b_l927_92703


namespace max_profit_140000_l927_92725

structure ProductionPlan where
  productA : ℕ
  productB : ℕ

def componentAUsage (plan : ProductionPlan) : ℕ := 4 * plan.productA
def componentBUsage (plan : ProductionPlan) : ℕ := 4 * plan.productB
def totalHours (plan : ProductionPlan) : ℕ := plan.productA + 2 * plan.productB
def profit (plan : ProductionPlan) : ℕ := 20000 * plan.productA + 30000 * plan.productB

def isValidPlan (plan : ProductionPlan) : Prop :=
  componentAUsage plan ≤ 16 ∧
  componentBUsage plan ≤ 12 ∧
  totalHours plan ≤ 8

theorem max_profit_140000 :
  ∃ (optimalPlan : ProductionPlan),
    isValidPlan optimalPlan ∧
    profit optimalPlan = 140000 ∧
    ∀ (plan : ProductionPlan), isValidPlan plan → profit plan ≤ profit optimalPlan :=
sorry

end max_profit_140000_l927_92725


namespace decimal_to_fraction_l927_92713

theorem decimal_to_fraction : (2.75 : ℚ) = 11 / 4 := by sorry

end decimal_to_fraction_l927_92713


namespace quadratic_root_in_unit_interval_l927_92796

theorem quadratic_root_in_unit_interval 
  (a b c m : ℝ) 
  (ha : a > 0) 
  (hm : m > 0) 
  (h_sum : a / (m + 2) + b / (m + 1) + c / m = 0) : 
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a * x^2 + b * x + c = 0 :=
by sorry

end quadratic_root_in_unit_interval_l927_92796


namespace arithmetic_geometric_mean_sum_squares_l927_92731

theorem arithmetic_geometric_mean_sum_squares (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 20)
  (h_geometric : Real.sqrt (x * y) = Real.sqrt 120) : 
  x^2 + y^2 = 1360 := by
  sorry

end arithmetic_geometric_mean_sum_squares_l927_92731


namespace sequence_difference_theorem_l927_92752

def is_valid_sequence (x : ℕ → ℕ) : Prop :=
  x 1 = 1 ∧ 
  (∀ n, x n < x (n + 1)) ∧
  (∀ n, x (2 * n + 1) ≤ 2 * n)

theorem sequence_difference_theorem (x : ℕ → ℕ) (h : is_valid_sequence x) :
  ∀ k : ℕ, ∃ r s : ℕ, x r - x s = k :=
sorry

end sequence_difference_theorem_l927_92752


namespace sum_of_four_integers_l927_92708

theorem sum_of_four_integers (a b c d : ℤ) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (a - 1) * (b - 1) * (c - 1) * (d - 1) = 25 →
  a + b + c + d = 4 := by
sorry

end sum_of_four_integers_l927_92708


namespace subtraction_of_fractions_l927_92706

theorem subtraction_of_fractions : (8 : ℚ) / 19 - (5 : ℚ) / 57 = (1 : ℚ) / 3 := by
  sorry

end subtraction_of_fractions_l927_92706


namespace eight_distinct_lengths_l927_92732

/-- Represents an isosceles right triangle with side length 24 -/
structure IsoscelesRightTriangle :=
  (side : ℝ)
  (is_24 : side = 24)

/-- Counts the number of distinct integer lengths of line segments from a vertex to the hypotenuse -/
def count_distinct_integer_lengths (t : IsoscelesRightTriangle) : ℕ :=
  sorry

/-- Theorem stating that there are exactly 8 distinct integer lengths -/
theorem eight_distinct_lengths (t : IsoscelesRightTriangle) : 
  count_distinct_integer_lengths t = 8 := by sorry

end eight_distinct_lengths_l927_92732


namespace zach_allowance_is_five_l927_92749

/-- Calculates Zach's weekly allowance given the conditions of his savings and earnings -/
def zachsAllowance (bikeCost lawnMowingPay babysittingRatePerHour babysittingHours currentSavings additionalNeeded : ℕ) : ℕ :=
  let totalNeeded := bikeCost - additionalNeeded
  let remainingToEarn := totalNeeded - currentSavings
  let otherEarnings := lawnMowingPay + babysittingRatePerHour * babysittingHours
  remainingToEarn - otherEarnings

/-- Proves that Zach's weekly allowance is $5 given the specified conditions -/
theorem zach_allowance_is_five :
  zachsAllowance 100 10 7 2 65 6 = 5 := by
  sorry

end zach_allowance_is_five_l927_92749


namespace book_words_per_page_l927_92746

theorem book_words_per_page (total_pages : Nat) (max_words_per_page : Nat) (remainder : Nat) :
  total_pages = 150 →
  max_words_per_page = 100 →
  remainder = 198 →
  ∃ p : Nat,
    p ≤ max_words_per_page ∧
    (total_pages * p) % 221 = remainder ∧
    p = 93 :=
by sorry

end book_words_per_page_l927_92746


namespace revenue_difference_is_164_5_l927_92722

/-- Represents the types of fruits sold by Kevin --/
inductive Fruit
  | Grapes
  | Mangoes
  | PassionFruits

/-- Represents the pricing and quantity information for each fruit --/
structure FruitInfo where
  price : ℕ
  quantity : ℕ
  discountThreshold : ℕ
  discountRate : ℚ

/-- Calculates the revenue for a given fruit with or without discount --/
def calculateRevenue (info : FruitInfo) (applyDiscount : Bool) : ℚ :=
  let price := if applyDiscount && info.quantity > info.discountThreshold
    then info.price * (1 - info.discountRate)
    else info.price
  price * info.quantity

/-- Theorem: The difference between total revenue without and with discounts is $164.5 --/
theorem revenue_difference_is_164_5 (fruitData : Fruit → FruitInfo) 
    (h1 : fruitData Fruit.Grapes = { price := 15, quantity := 13, discountThreshold := 10, discountRate := 0.1 })
    (h2 : fruitData Fruit.Mangoes = { price := 20, quantity := 20, discountThreshold := 15, discountRate := 0.15 })
    (h3 : fruitData Fruit.PassionFruits = { price := 25, quantity := 17, discountThreshold := 5, discountRate := 0.2 })
    (h4 : (fruitData Fruit.Grapes).quantity + (fruitData Fruit.Mangoes).quantity + (fruitData Fruit.PassionFruits).quantity = 50) :
    (calculateRevenue (fruitData Fruit.Grapes) false +
     calculateRevenue (fruitData Fruit.Mangoes) false +
     calculateRevenue (fruitData Fruit.PassionFruits) false) -
    (calculateRevenue (fruitData Fruit.Grapes) true +
     calculateRevenue (fruitData Fruit.Mangoes) true +
     calculateRevenue (fruitData Fruit.PassionFruits) true) = 164.5 := by
  sorry


end revenue_difference_is_164_5_l927_92722


namespace trig_equation_result_l927_92791

theorem trig_equation_result (x : Real) : 
  2 * Real.cos x - 5 * Real.sin x = 3 → 
  Real.sin x + 2 * Real.cos x = 1/2 ∨ Real.sin x + 2 * Real.cos x = 83/29 := by
sorry

end trig_equation_result_l927_92791


namespace mixture_theorem_l927_92780

/-- Represents a mixture of three liquids -/
structure Mixture where
  lemon : ℚ
  oil : ℚ
  vinegar : ℚ

/-- Mix A composition -/
def mixA : Mixture := ⟨1, 2, 3⟩

/-- Mix B composition -/
def mixB : Mixture := ⟨3, 4, 5⟩

/-- Checks if it's possible to create a target mixture from Mix A and Mix B -/
def canCreateMixture (target : Mixture) : Prop :=
  ∃ (x y : ℚ), x ≥ 0 ∧ y ≥ 0 ∧
    x * mixA.lemon + y * mixB.lemon = (x + y) * target.lemon ∧
    x * mixA.oil + y * mixB.oil = (x + y) * target.oil ∧
    x * mixA.vinegar + y * mixB.vinegar = (x + y) * target.vinegar

theorem mixture_theorem :
  canCreateMixture ⟨3, 5, 7⟩ ∧
  ¬canCreateMixture ⟨2, 5, 8⟩ ∧
  ¬canCreateMixture ⟨4, 5, 6⟩ ∧
  ¬canCreateMixture ⟨5, 6, 7⟩ := by
  sorry


end mixture_theorem_l927_92780


namespace circle_equation_perpendicular_chord_values_l927_92763

-- Define the circle
def circle_center : ℝ × ℝ := (2, 0)
def circle_radius : ℝ := 5

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := 4 * x + 3 * y - 33 = 0

-- Define the intersecting line
def intersecting_line (a x y : ℝ) : Prop := a * x - y - 7 = 0

-- Theorem for the circle equation
theorem circle_equation : 
  ∀ x y : ℝ, (x - circle_center.1)^2 + (y - circle_center.2)^2 = circle_radius^2 ↔ 
  (x - 2)^2 + y^2 = 25 := by sorry

-- Theorem for the values of a
theorem perpendicular_chord_values (a : ℝ) :
  (∃ A B : ℝ × ℝ, 
    (A.1 - circle_center.1)^2 + (A.2 - circle_center.2)^2 = circle_radius^2 ∧
    (B.1 - circle_center.1)^2 + (B.2 - circle_center.2)^2 = circle_radius^2 ∧
    intersecting_line a A.1 A.2 ∧
    intersecting_line a B.1 B.2 ∧
    ((A.1 - circle_center.1) * (B.1 - circle_center.1) + (A.2 - circle_center.2) * (B.2 - circle_center.2) = 0)) →
  (a = 1 ∨ a = -73/17) := by sorry

end circle_equation_perpendicular_chord_values_l927_92763


namespace school_transfer_percentage_l927_92712

theorem school_transfer_percentage :
  ∀ (total_students : ℝ) (school_A_percentage : ℝ) (school_B_percentage : ℝ)
    (transfer_A_to_C_percentage : ℝ) (transfer_B_to_C_percentage : ℝ),
  school_A_percentage = 60 →
  school_B_percentage = 100 - school_A_percentage →
  transfer_A_to_C_percentage = 30 →
  transfer_B_to_C_percentage = 40 →
  let students_A := total_students * (school_A_percentage / 100)
  let students_B := total_students * (school_B_percentage / 100)
  let students_C := (students_A * (transfer_A_to_C_percentage / 100)) +
                    (students_B * (transfer_B_to_C_percentage / 100))
  (students_C / total_students) * 100 = 34 :=
by sorry

end school_transfer_percentage_l927_92712


namespace negative_quadratic_inequality_l927_92702

/-- A quadratic polynomial ax^2 + bx + c that is negative for all real x -/
structure NegativeQuadratic where
  a : ℝ
  b : ℝ
  c : ℝ
  is_negative : ∀ x : ℝ, a * x^2 + b * x + c < 0

/-- Theorem: For a negative quadratic polynomial, b/a < c/a + 1 -/
theorem negative_quadratic_inequality (q : NegativeQuadratic) : q.b / q.a < q.c / q.a + 1 := by
  sorry

end negative_quadratic_inequality_l927_92702


namespace calculate_expression_l927_92792

theorem calculate_expression (a : ℝ) : (-2 * a^2)^3 / a^3 = -8 * a^3 := by
  sorry

end calculate_expression_l927_92792


namespace equation_solution_l927_92719

theorem equation_solution (x y : ℝ) : 
  x^2 + (1 - y)^2 + (x - y)^2 = 1/3 ↔ x = 1/3 ∧ y = 2/3 := by
  sorry

end equation_solution_l927_92719


namespace infant_weight_at_four_months_l927_92717

/-- Represents the weight of an infant in grams at a given age in months. -/
def infantWeight (birthWeight : ℝ) (ageMonths : ℝ) : ℝ :=
  birthWeight + 700 * ageMonths

/-- Theorem stating that an infant with a birth weight of 3000 grams will weigh 5800 grams at 4 months. -/
theorem infant_weight_at_four_months :
  infantWeight 3000 4 = 5800 := by
  sorry

end infant_weight_at_four_months_l927_92717


namespace remainder_theorem_l927_92783

/-- Given a polynomial f(x) with the following properties:
    1) When divided by (x-1), the remainder is 8
    2) When divided by (x+1), the remainder is 1
    This theorem states that the remainder when f(x) is divided by (x^2-1) is -7x-9 -/
theorem remainder_theorem (f : ℝ → ℝ) 
  (h1 : ∃ g : ℝ → ℝ, ∀ x, f x = g x * (x - 1) + 8)
  (h2 : ∃ h : ℝ → ℝ, ∀ x, f x = h x * (x + 1) + 1) :
  ∃ q : ℝ → ℝ, ∀ x, f x = q x * (x^2 - 1) + (-7*x - 9) :=
sorry

end remainder_theorem_l927_92783


namespace settlement_area_theorem_l927_92765

/-- Represents the lengths of the sides of the fields and forest -/
structure SettlementGeometry where
  r : ℝ  -- Length of the side of the square field
  p : ℝ  -- Length of the shorter side of the rectangular field
  q : ℝ  -- Length of the longer side of the rectangular forest

/-- The total area of the forest and fields given the geometry -/
def totalArea (g : SettlementGeometry) : ℝ :=
  g.r^2 + 4*g.p^2 + 12*g.q

/-- The conditions given in the problem -/
def satisfiesConditions (g : SettlementGeometry) : Prop :=
  12*g.q = g.r^2 + 4*g.p^2 + 45 ∧
  g.r > 0 ∧ g.p > 0 ∧ g.q > 0

theorem settlement_area_theorem (g : SettlementGeometry) 
  (h : satisfiesConditions g) : totalArea g = 135 := by
  sorry

#check settlement_area_theorem

end settlement_area_theorem_l927_92765


namespace sum_even_integers_40_to_60_l927_92773

def evenIntegersFrom40To60 : List ℕ := [40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60]

def x : ℕ := evenIntegersFrom40To60.sum

def y : ℕ := evenIntegersFrom40To60.length

theorem sum_even_integers_40_to_60 : x + y = 561 := by
  sorry

end sum_even_integers_40_to_60_l927_92773


namespace circle_tangent_to_line_l927_92730

/-- The circle x^2 + y^2 = m^2 is tangent to the line x - y = m if and only if m = 0 -/
theorem circle_tangent_to_line (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = m^2 ∧ x - y = m ∧ 
    (∀ (x' y' : ℝ), x'^2 + y'^2 = m^2 → x' - y' = m → (x', y') = (x, y))) ↔ 
  m = 0 :=
sorry

end circle_tangent_to_line_l927_92730


namespace f_nonnegative_iff_l927_92742

/-- The function f(x) defined in the problem -/
def f (a x : ℝ) : ℝ := x^2 - 2*x - |x - 1 - a| - |x - 2| + 4

/-- Theorem stating the condition for f(x) to be non-negative for all real x -/
theorem f_nonnegative_iff (a : ℝ) : 
  (∀ x : ℝ, f a x ≥ 0) ↔ -2 ≤ a ∧ a ≤ 1 :=
sorry

end f_nonnegative_iff_l927_92742


namespace octal_4652_to_decimal_l927_92795

def octal_to_decimal (octal : List Nat) : Nat :=
  octal.enum.foldl (fun acc (i, digit) => acc + digit * (8 ^ i)) 0

theorem octal_4652_to_decimal :
  octal_to_decimal [2, 5, 6, 4] = 2474 := by
  sorry

end octal_4652_to_decimal_l927_92795


namespace negation_of_universal_statement_l927_92748

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x + 1 > 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 + x + 1 ≤ 0) :=
by sorry

end negation_of_universal_statement_l927_92748


namespace angle_c_possibilities_l927_92758

theorem angle_c_possibilities : ∃ (s : Finset ℕ), 
  (∀ c ∈ s, ∃ d : ℕ, 
    c > 0 ∧ d > 0 ∧ 
    c + d = 180 ∧ 
    ∃ k : ℕ, k > 0 ∧ c = k * d) ∧
  (∀ c : ℕ, 
    (∃ d : ℕ, c > 0 ∧ d > 0 ∧ c + d = 180 ∧ ∃ k : ℕ, k > 0 ∧ c = k * d) → 
    c ∈ s) ∧
  s.card = 17 :=
sorry

end angle_c_possibilities_l927_92758


namespace complex_symmetric_product_l927_92760

theorem complex_symmetric_product (z₁ z₂ : ℂ) :
  z₁.im = -z₂.im → z₁.re = z₂.re → z₁ = 2 - I → z₁ * z₂ = 5 := by sorry

end complex_symmetric_product_l927_92760


namespace no_solution_iff_m_eq_neg_six_l927_92790

theorem no_solution_iff_m_eq_neg_six (m : ℝ) :
  (∀ x : ℝ, x ≠ -2 → (x - 3) / (x + 2) + (x + 1) / (x + 2) ≠ m / (x + 2)) ↔ m = -6 := by
  sorry

end no_solution_iff_m_eq_neg_six_l927_92790


namespace midline_tetrahedra_volume_ratio_l927_92737

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  volume : ℝ

/-- A tetrahedron formed by connecting midpoints of midlines to vertices -/
structure MidlineTetrahedron where
  volume : ℝ

/-- The common part of three MidlineTetrahedra -/
structure CommonTetrahedron where
  volume : ℝ

/-- Given a regular tetrahedron, construct three MidlineTetrahedra -/
def construct_midline_tetrahedra (t : RegularTetrahedron) : 
  (MidlineTetrahedron × MidlineTetrahedron × MidlineTetrahedron) :=
  sorry

/-- Find the common part of three MidlineTetrahedra -/
def find_common_part (t1 t2 t3 : MidlineTetrahedron) : CommonTetrahedron :=
  sorry

/-- The theorem to be proved -/
theorem midline_tetrahedra_volume_ratio 
  (t : RegularTetrahedron) 
  (t1 t2 t3 : MidlineTetrahedron) 
  (c : CommonTetrahedron) :
  t1 = (construct_midline_tetrahedra t).1 ∧
  t2 = (construct_midline_tetrahedra t).2.1 ∧
  t3 = (construct_midline_tetrahedra t).2.2 ∧
  c = find_common_part t1 t2 t3 →
  c.volume = t.volume / 10 :=
by sorry

end midline_tetrahedra_volume_ratio_l927_92737


namespace power_difference_equals_l927_92771

theorem power_difference_equals (a b c : ℕ) :
  3^456 - 9^5 / 9^3 = 3^456 - 81 := by sorry

end power_difference_equals_l927_92771


namespace weight_of_seven_moles_l927_92715

/-- The weight of a given number of moles of a compound -/
def weight_of_moles (molecular_weight : ℕ) (moles : ℕ) : ℕ :=
  molecular_weight * moles

/-- Theorem: The weight of 7 moles of a compound with molecular weight 2856 is 19992 -/
theorem weight_of_seven_moles :
  weight_of_moles 2856 7 = 19992 := by
  sorry

end weight_of_seven_moles_l927_92715


namespace dentist_age_l927_92733

/-- The dentist's current age satisfies the given condition and is equal to 32. -/
theorem dentist_age : ∃ (x : ℕ), (x - 8) / 6 = (x + 8) / 10 ∧ x = 32 := by
  sorry

end dentist_age_l927_92733


namespace equation_solutions_l927_92778

theorem equation_solutions :
  let f (x : ℝ) := 4 * (3 * x)^2 + 3 * x + 6 - (3 * (9 * x^2 + 3 * x + 3))
  ∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = -1/3 := by
  sorry

end equation_solutions_l927_92778


namespace red_tint_percentage_l927_92759

/-- Given a paint mixture, calculate the percentage of red tint after adding more red tint -/
theorem red_tint_percentage (original_volume : ℝ) (original_red_percent : ℝ) (added_red_volume : ℝ) :
  original_volume = 40 →
  original_red_percent = 20 →
  added_red_volume = 10 →
  let original_red_volume := original_red_percent / 100 * original_volume
  let new_red_volume := original_red_volume + added_red_volume
  let new_total_volume := original_volume + added_red_volume
  (new_red_volume / new_total_volume) * 100 = 36 := by
  sorry

end red_tint_percentage_l927_92759


namespace intersection_points_theorem_l927_92782

-- Define the functions
def p (x : ℝ) : ℝ := x^2 - 4*x + 3
def q (x : ℝ) : ℝ := -p x + 2
def r (x : ℝ) : ℝ := p (-x)

-- Define the number of intersection points
def c : ℕ := 2  -- Number of intersections between p and q
def d : ℕ := 1  -- Number of intersections between p and r

-- Theorem statement
theorem intersection_points_theorem :
  (∀ x : ℝ, p x = q x → x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2) ∧
  (∀ x : ℝ, p x = r x → x = 0) ∧
  (10 * c + d = 21) :=
sorry

end intersection_points_theorem_l927_92782


namespace max_value_of_operation_achievable_max_value_l927_92729

theorem max_value_of_operation (n : ℕ) : 
  (10 ≤ n ∧ n ≤ 99) → 2 * (200 - n) ≤ 380 :=
by
  sorry

theorem achievable_max_value : 
  ∃ (n : ℕ), (10 ≤ n ∧ n ≤ 99) ∧ 2 * (200 - n) = 380 :=
by
  sorry

end max_value_of_operation_achievable_max_value_l927_92729


namespace fifteenth_student_age_l927_92775

theorem fifteenth_student_age 
  (total_students : Nat) 
  (avg_age_all : ℝ) 
  (group1_size : Nat) 
  (avg_age_group1 : ℝ) 
  (group2_size : Nat) 
  (avg_age_group2 : ℝ) 
  (h1 : total_students = 15)
  (h2 : avg_age_all = 15)
  (h3 : group1_size = 5)
  (h4 : avg_age_group1 = 12)
  (h5 : group2_size = 9)
  (h6 : avg_age_group2 = 16)
  (h7 : group1_size + group2_size + 1 = total_students) :
  (total_students : ℝ) * avg_age_all - 
  ((group1_size : ℝ) * avg_age_group1 + (group2_size : ℝ) * avg_age_group2) = 21 := by
  sorry

end fifteenth_student_age_l927_92775


namespace average_difference_l927_92747

theorem average_difference (x : ℝ) : 
  (20 + 40 + 60) / 3 = ((10 + 70 + x) / 3) + 8 → x = 16 := by
sorry

end average_difference_l927_92747


namespace sum_of_three_numbers_l927_92784

theorem sum_of_three_numbers (a b c : ℝ) 
  (sum1 : a + b = 36)
  (sum2 : b + c = 55)
  (sum3 : c + a = 60) : 
  a + b + c = 75.5 := by
sorry

end sum_of_three_numbers_l927_92784


namespace triangle_vertices_l927_92721

structure Triangle where
  a : ℝ
  m_a : ℝ
  s_a : ℝ

def is_valid_vertex (t : Triangle) (x y : ℝ) : Prop :=
  (x - 5)^2 + y^2 = t.s_a^2 ∧ 
  |y| = t.m_a

theorem triangle_vertices (t : Triangle) 
  (h1 : t.a = 10) 
  (h2 : t.m_a = 4) 
  (h3 : t.s_a = 5) : 
  (is_valid_vertex t 8 4 ∧ 
   is_valid_vertex t 8 (-4) ∧ 
   is_valid_vertex t 2 4 ∧ 
   is_valid_vertex t 2 (-4)) :=
by sorry

end triangle_vertices_l927_92721


namespace tan_addition_result_l927_92768

theorem tan_addition_result (x : Real) (h : Real.tan x = 3) :
  Real.tan (x + π / 3) = -(6 + 5 * Real.sqrt 3) / 13 := by
  sorry

end tan_addition_result_l927_92768


namespace square_area_from_perimeter_l927_92705

/-- For a square with perimeter 48 meters, its area is 144 square meters. -/
theorem square_area_from_perimeter : 
  ∀ (s : Real), 
    (4 * s = 48) →  -- perimeter = 4 * side length = 48
    (s * s = 144)   -- area = side length * side length = 144
:= by sorry

end square_area_from_perimeter_l927_92705


namespace stratified_sampling_l927_92738

theorem stratified_sampling (total_families : ℕ) (high_income : ℕ) (middle_income : ℕ) (low_income : ℕ) 
  (high_income_sampled : ℕ) (h1 : total_families = 500) (h2 : high_income = 125) (h3 : middle_income = 280) 
  (h4 : low_income = 95) (h5 : high_income_sampled = 25) :
  (high_income_sampled * low_income) / high_income = 19 := by
sorry

end stratified_sampling_l927_92738


namespace blue_lipstick_count_l927_92714

theorem blue_lipstick_count (total_students : ℕ) 
  (h1 : total_students = 200)
  (h2 : ∃ colored_lipstick : ℕ, colored_lipstick = total_students / 2)
  (h3 : ∃ red_lipstick : ℕ, red_lipstick = colored_lipstick / 4)
  (h4 : ∃ blue_lipstick : ℕ, blue_lipstick = red_lipstick / 5) :
  blue_lipstick = 5 := by
  sorry

end blue_lipstick_count_l927_92714


namespace abs_geq_one_necessary_not_sufficient_for_x_gt_two_l927_92720

theorem abs_geq_one_necessary_not_sufficient_for_x_gt_two :
  (∀ x : ℝ, x > 2 → |x| ≥ 1) ∧
  (∃ x : ℝ, |x| ≥ 1 ∧ ¬(x > 2)) :=
by sorry

end abs_geq_one_necessary_not_sufficient_for_x_gt_two_l927_92720


namespace quadratic_inequality_solution_l927_92728

/-- Given two incorrect solutions to a quadratic inequality, prove the correct solution -/
theorem quadratic_inequality_solution 
  (b c : ℝ) 
  (h1 : ∀ x, x^2 + b*x + c < 0 ↔ -6 < x ∧ x < 2)
  (h2 : ∃ c', ∀ x, x^2 + b*x + c' < 0 ↔ -3 < x ∧ x < 2) :
  ∀ x, x^2 + b*x + c < 0 ↔ -4 < x ∧ x < 3 :=
by sorry


end quadratic_inequality_solution_l927_92728


namespace linear_function_x_axis_intersection_l927_92756

/-- A linear function f(x) = -x + 2 -/
def f (x : ℝ) : ℝ := -x + 2

/-- The x-coordinate of the intersection point with the x-axis -/
def x_intersection : ℝ := 2

theorem linear_function_x_axis_intersection :
  f x_intersection = 0 ∧ x_intersection = 2 := by
  sorry

end linear_function_x_axis_intersection_l927_92756


namespace product_sum_division_l927_92754

theorem product_sum_division : (10 * 19 * 20 * 53 * 100 + 601) / 13 = 1549277 := by
  sorry

end product_sum_division_l927_92754


namespace sara_picked_24_more_peaches_l927_92793

/-- The number of additional peaches Sara picked at the orchard -/
def additional_peaches (initial_peaches total_peaches : ℝ) : ℝ :=
  total_peaches - initial_peaches

/-- Theorem: Sara picked 24 additional peaches at the orchard -/
theorem sara_picked_24_more_peaches (initial_peaches total_peaches : ℝ)
  (h1 : initial_peaches = 61.0)
  (h2 : total_peaches = 85.0) :
  additional_peaches initial_peaches total_peaches = 24 := by
  sorry

end sara_picked_24_more_peaches_l927_92793


namespace nina_has_24_dollars_l927_92785

/-- The amount of money Nina has -/
def nina_money : ℝ := 24

/-- The original price of a widget -/
def original_price : ℝ := 4

/-- Nina can purchase exactly 6 widgets at the original price -/
axiom nina_purchase_original : nina_money = 6 * original_price

/-- If each widget's price is reduced by $1, Nina can purchase exactly 8 widgets -/
axiom nina_purchase_reduced : nina_money = 8 * (original_price - 1)

/-- Proof that Nina has $24 -/
theorem nina_has_24_dollars : nina_money = 24 := by sorry

end nina_has_24_dollars_l927_92785


namespace rationalize_denominator_l927_92710

theorem rationalize_denominator :
  (Real.sqrt 12 + Real.sqrt 5) / (Real.sqrt 3 + Real.sqrt 5) = -1/2 + Real.sqrt 15 / 2 := by
  sorry

end rationalize_denominator_l927_92710


namespace min_value_product_l927_92734

theorem min_value_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_prod : a * b * c = 8) :
  (a + 3 * b) * (b + 3 * c) * (a * c + 2) ≥ 64 ∧
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ * b₀ * c₀ = 8 ∧
    (a₀ + 3 * b₀) * (b₀ + 3 * c₀) * (a₀ * c₀ + 2) = 64 :=
by sorry

end min_value_product_l927_92734


namespace fraction_equality_l927_92700

theorem fraction_equality (p q r s : ℚ) 
  (h1 : p / q = 2)
  (h2 : q / r = 4 / 5)
  (h3 : r / s = 3) :
  s / p = 5 / 24 := by
  sorry

end fraction_equality_l927_92700


namespace tangent_line_properties_l927_92787

-- Define the function
def f (m : ℝ) (x : ℝ) : ℝ := m * x^3 + 2 * x + 1

-- Define the derivative of the function
def f' (m : ℝ) (x : ℝ) : ℝ := 3 * m * x^2 + 2

theorem tangent_line_properties (m : ℝ) :
  -- Part 1: Parallel to y = 3x
  (f' m 1 = 3 → m = 1/3) ∧
  -- Part 2: Perpendicular to y = -1/2x
  (f' m 1 = 2 → ∃ b : ℝ, ∀ x y : ℝ, y = 2 * x + b ↔ y - f m 1 = f' m 1 * (x - 1)) :=
by sorry

end tangent_line_properties_l927_92787


namespace magnitude_of_c_l927_92709

/-- Given vectors a and b, with c parallel to b and its projection onto a being 2, 
    prove that the magnitude of c is 2√5 -/
theorem magnitude_of_c (a b c : ℝ × ℝ) : 
  a = (1, 0) → 
  b = (1, 2) → 
  (c.1 / b.1 = c.2 / b.2) →  -- c is parallel to b
  (c.1 * a.1 + c.2 * a.2) / Real.sqrt (a.1^2 + a.2^2) = 2 →  -- projection of c onto a is 2
  Real.sqrt (c.1^2 + c.2^2) = 2 * Real.sqrt 5 := by
sorry

end magnitude_of_c_l927_92709


namespace sum_of_coefficients_l927_92786

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (2*x + 1)^9 = a₀ + a₁*(x+2) + a₂*(x+2)^2 + a₃*(x+2)^3 + a₄*(x+2)^4 + 
    a₅*(x+2)^5 + a₆*(x+2)^6 + a₇*(x+2)^7 + a₈*(x+2)^8 + a₉*(x+2)^9 + a₁₀*(x+2)^10 + a₁₁*(x+2)^11) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ = -2 :=
by
  sorry

end sum_of_coefficients_l927_92786


namespace distance_to_sea_world_l927_92744

/-- Calculates the distance to Sea World based on given conditions --/
theorem distance_to_sea_world 
  (savings : ℕ) 
  (parking_cost : ℕ) 
  (entrance_cost : ℕ) 
  (meal_pass_cost : ℕ) 
  (car_efficiency : ℕ) 
  (gas_price : ℕ) 
  (additional_savings_needed : ℕ) 
  (h1 : savings = 28)
  (h2 : parking_cost = 10)
  (h3 : entrance_cost = 55)
  (h4 : meal_pass_cost = 25)
  (h5 : car_efficiency = 30)
  (h6 : gas_price = 3)
  (h7 : additional_savings_needed = 95)
  : ℕ := by
  sorry

#check distance_to_sea_world

end distance_to_sea_world_l927_92744


namespace gwen_money_left_l927_92735

/-- The amount of money Gwen has left after spending some of her birthday money -/
def money_left (received : ℕ) (spent : ℕ) : ℕ :=
  received - spent

/-- Theorem stating that Gwen has 2 dollars left -/
theorem gwen_money_left :
  money_left 5 3 = 2 := by
  sorry

end gwen_money_left_l927_92735


namespace eleven_million_nine_hundred_thousand_scientific_notation_l927_92723

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem eleven_million_nine_hundred_thousand_scientific_notation :
  toScientificNotation 11090000 = ScientificNotation.mk 1.109 7 (by norm_num) :=
sorry

end eleven_million_nine_hundred_thousand_scientific_notation_l927_92723


namespace power_tower_mod_2000_l927_92751

theorem power_tower_mod_2000 : 
  (5 : ℕ) ^ (5 ^ (5 ^ 5)) ≡ 625 [MOD 2000] := by
  sorry

end power_tower_mod_2000_l927_92751


namespace factorization_equality_l927_92743

theorem factorization_equality (x y : ℝ) : -4*x^2 + 12*x*y - 9*y^2 = -(2*x - 3*y)^2 := by
  sorry

end factorization_equality_l927_92743


namespace total_cost_matches_expected_l927_92788

/-- Calculate the total cost of an order with given conditions --/
def calculate_total_cost (burger_price : ℚ) (soda_price : ℚ) (chicken_sandwich_price : ℚ) 
  (happy_hour_discount : ℚ) (coupon_discount : ℚ) (sales_tax : ℚ) 
  (paulo_burgers : ℕ) (paulo_sodas : ℕ) (jeremy_burgers : ℕ) (jeremy_sodas : ℕ) 
  (stephanie_burgers : ℕ) (stephanie_sodas : ℕ) (stephanie_chicken : ℕ) : ℚ :=
  let total_burgers := paulo_burgers + jeremy_burgers + stephanie_burgers
  let total_sodas := paulo_sodas + jeremy_sodas + stephanie_sodas
  let subtotal := burger_price * total_burgers + soda_price * total_sodas + 
                  chicken_sandwich_price * stephanie_chicken
  let tax_amount := sales_tax * subtotal
  let total_with_tax := subtotal + tax_amount
  let coupon_applied := if total_with_tax > 25 then total_with_tax - coupon_discount else total_with_tax
  let happy_hour_discount_amount := if total_burgers > 2 then happy_hour_discount * (burger_price * total_burgers) else 0
  coupon_applied - happy_hour_discount_amount

/-- Theorem stating that the total cost matches the expected result --/
theorem total_cost_matches_expected : 
  calculate_total_cost 6 2 7.5 0.1 5 0.05 1 1 2 2 3 1 1 = 45.48 := by
  sorry

end total_cost_matches_expected_l927_92788


namespace part1_part2_part3_l927_92727

-- Define the system of linear equations
def system (x y : ℝ) : Prop :=
  2 * x + 3 * y = 6 ∧ 3 * x + 2 * y = 4

-- Define the new operation *
def star (a b c : ℝ) (x y : ℝ) : ℝ := a * x + b * y + c

-- Theorem for part 1
theorem part1 (x y : ℝ) (h : system x y) : x + y = 2 ∧ x - y = -2 := by
  sorry

-- Theorem for part 2
theorem part2 : ∃ x y : ℝ, 2024 * x + 2025 * y = 2023 ∧ 2022 * x + 2023 * y = 2021 ∧ x = 2 ∧ y = -1 := by
  sorry

-- Theorem for part 3
theorem part3 (a b c : ℝ) (h1 : star a b c 2 4 = 15) (h2 : star a b c 3 7 = 27) : 
  star a b c 1 1 = 3 := by
  sorry

end part1_part2_part3_l927_92727


namespace boat_speed_in_still_water_l927_92707

/-- The speed of a boat in still water, given its downstream and upstream distances traveled in one hour. -/
theorem boat_speed_in_still_water (downstream upstream : ℝ) (h1 : downstream = 11) (h2 : upstream = 5) :
  let boat_speed := (downstream + upstream) / 2
  boat_speed = 8 := by sorry

end boat_speed_in_still_water_l927_92707


namespace exists_n_fractional_part_greater_than_bound_l927_92779

theorem exists_n_fractional_part_greater_than_bound : 
  ∃ n : ℕ+, (2 + Real.sqrt 2)^(n : ℝ) - ⌊(2 + Real.sqrt 2)^(n : ℝ)⌋ > 0.999999 := by
  sorry

end exists_n_fractional_part_greater_than_bound_l927_92779


namespace distribute_students_count_l927_92776

/-- The number of ways to distribute 5 students into 3 groups -/
def distribute_students : ℕ :=
  let n : ℕ := 5  -- Total number of students
  let k : ℕ := 3  -- Number of groups
  let min_a : ℕ := 2  -- Minimum number of students in Group A
  let min_bc : ℕ := 1  -- Minimum number of students in Groups B and C
  sorry

/-- Theorem stating that the number of distribution schemes is 80 -/
theorem distribute_students_count : distribute_students = 80 := by
  sorry

end distribute_students_count_l927_92776


namespace green_hats_count_l927_92701

/-- The number of green hard hats initially in the truck -/
def initial_green_hats : ℕ := sorry

/-- The number of pink hard hats initially in the truck -/
def initial_pink_hats : ℕ := 26

/-- The number of yellow hard hats in the truck -/
def yellow_hats : ℕ := 24

/-- The number of pink hard hats Carl takes away -/
def carl_pink_hats : ℕ := 4

/-- The number of pink hard hats John takes away -/
def john_pink_hats : ℕ := 6

/-- The number of green hard hats John takes away -/
def john_green_hats : ℕ := 2 * john_pink_hats

/-- The total number of hard hats remaining in the truck -/
def remaining_hats : ℕ := 43

theorem green_hats_count : initial_green_hats = 15 :=
  by sorry

end green_hats_count_l927_92701


namespace unique_magnitude_complex_roots_l927_92797

theorem unique_magnitude_complex_roots (z : ℂ) :
  (3 * z^2 - 18 * z + 55 = 0) →
  ∃! m : ℝ, ∃ z₁ z₂ : ℂ, (3 * z₁^2 - 18 * z₁ + 55 = 0) ∧
                         (3 * z₂^2 - 18 * z₂ + 55 = 0) ∧
                         (Complex.abs z₁ = m) ∧
                         (Complex.abs z₂ = m) :=
by sorry

end unique_magnitude_complex_roots_l927_92797


namespace dodecahedron_faces_l927_92767

/-- A regular dodecahedron is a Platonic solid with 12 faces. -/
def RegularDodecahedron : Type := Unit

/-- The number of faces in a regular dodecahedron. -/
def num_faces (d : RegularDodecahedron) : ℕ := 12

/-- Theorem: A regular dodecahedron has 12 faces. -/
theorem dodecahedron_faces :
  ∀ (d : RegularDodecahedron), num_faces d = 12 := by
  sorry

end dodecahedron_faces_l927_92767


namespace product_of_middle_terms_l927_92774

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem product_of_middle_terms 
  (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_prod : a 7 * a 12 = 5) : 
  a 8 * a 9 * a 10 * a 11 = 25 := by
sorry

end product_of_middle_terms_l927_92774


namespace smallest_square_area_for_radius_6_l927_92716

/-- The area of the smallest square that can contain a circle with a given radius -/
def smallest_square_area (radius : ℝ) : ℝ :=
  (2 * radius) ^ 2

/-- Theorem: The area of the smallest square that can contain a circle with a radius of 6 is 144 -/
theorem smallest_square_area_for_radius_6 :
  smallest_square_area 6 = 144 := by
  sorry

end smallest_square_area_for_radius_6_l927_92716


namespace square_area_error_percentage_l927_92741

theorem square_area_error_percentage (s : ℝ) (h : s > 0) :
  let measured_side := s * (1 + 0.02)
  let actual_area := s^2
  let calculated_area := measured_side^2
  let area_error := calculated_area - actual_area
  let area_error_percentage := (area_error / actual_area) * 100
  area_error_percentage = 4.04 := by
sorry

end square_area_error_percentage_l927_92741


namespace statement_A_statement_B_l927_92770

-- Define the parabola E
def E (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of E
def F : ℝ × ℝ := (1, 0)

-- Define the circle F
def circle_F (x y r : ℝ) : Prop := (x - 1)^2 + y^2 = r^2

-- Define the line l_0
def l_0 (t : ℝ) (x y : ℝ) : Prop := x = t*y + 1

-- Define the intersection points A and B
def A (t : ℝ) : ℝ × ℝ := (t^2 + 1, 2*t)
def B (t : ℝ) : ℝ × ℝ := (t^2 + 1, -2*t)

-- Define the midpoint M
def M (t : ℝ) : ℝ × ℝ := (2*t^2 + 1, 2*t)

-- Define point T
def T : ℝ × ℝ := (0, 1)

-- Theorem for statement A
theorem statement_A (t : ℝ) : 
  let y_1 := (A t).2
  let y_2 := (B t).2
  let y_3 := -1/t
  1/y_1 + 1/y_2 = 1/y_3 :=
sorry

-- Theorem for statement B
theorem statement_B : 
  ∃ a b c : ℝ, ∀ t : ℝ, 
    let (x, y) := M t
    y^2 = a*x + b*y + c :=
sorry

end statement_A_statement_B_l927_92770


namespace cube_surface_area_for_given_volume_l927_92739

def cube_volume : ℝ := 3375

def cube_surface_area (v : ℝ) : ℝ :=
  6 * (v ^ (1/3)) ^ 2

theorem cube_surface_area_for_given_volume :
  cube_surface_area cube_volume = 1350 := by
  sorry

end cube_surface_area_for_given_volume_l927_92739


namespace pizza_order_theorem_l927_92745

/-- The number of pizzas needed for a group of people -/
def pizzas_needed (num_people : ℕ) (slices_per_person : ℕ) (slices_per_pizza : ℕ) : ℕ :=
  (num_people * slices_per_person + slices_per_pizza - 1) / slices_per_pizza

/-- Theorem: The number of pizzas needed for 18 people, where each person gets 3 slices
    and each pizza has 9 slices, is equal to 6 -/
theorem pizza_order_theorem :
  pizzas_needed 18 3 9 = 6 := by
  sorry

end pizza_order_theorem_l927_92745


namespace quadratic_inequality_solution_set_l927_92718

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c

-- Define the solution set of the first inequality
def solution_set (a b c : ℝ) := {x : ℝ | f a b c x > 0}

-- State the theorem
theorem quadratic_inequality_solution_set 
  (a b c : ℝ) (h : solution_set a b c = {x : ℝ | 2 < x ∧ x < 3}) :
  {x : ℝ | c * x^2 - b * x + a > 0} = {x : ℝ | -1/2 < x ∧ x < -1/3} := by
  sorry

end quadratic_inequality_solution_set_l927_92718


namespace negation_of_existence_negation_of_quadratic_inequality_l927_92781

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) :=
by
  sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 + x - 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + x - 2 > 0) :=
by
  sorry

end negation_of_existence_negation_of_quadratic_inequality_l927_92781


namespace extreme_point_of_f_l927_92769

/-- The function f(x) = 3/2 * x^2 - ln(x) for x > 0 has an extreme point at x = √3/3 -/
theorem extreme_point_of_f (x : ℝ) (h : x > 0) : 
  let f := fun (x : ℝ) => 3/2 * x^2 - Real.log x
  ∃ (c : ℝ), c = Real.sqrt 3 / 3 ∧ 
    (∀ y > 0, f y ≥ f c) ∨ (∀ y > 0, f y ≤ f c) := by
  sorry


end extreme_point_of_f_l927_92769


namespace domain_intersection_complement_l927_92704

-- Define the universal set as real numbers
def U : Type := ℝ

-- Define the function f(x) = ln(1-x)
noncomputable def f (x : ℝ) : ℝ := Real.log (1 - x)

-- Define the domain M of f
def M : Set ℝ := {x | x < 1}

-- Define the set N
def N : Set ℝ := {x | 0 < x ∧ x < 2}

-- State the theorem
theorem domain_intersection_complement :
  M ∩ (Set.univ \ N) = Set.Iic 0 :=
sorry

end domain_intersection_complement_l927_92704


namespace kerrys_age_l927_92761

/-- Proves Kerry's age given the conditions of the birthday candle problem -/
theorem kerrys_age (num_cakes : ℕ) (candles_per_box : ℕ) (cost_per_box : ℚ) (total_cost : ℚ) :
  num_cakes = 3 →
  candles_per_box = 12 →
  cost_per_box = 5/2 →
  total_cost = 5 →
  (total_cost / cost_per_box * candles_per_box) / num_cakes = 8 := by
sorry

end kerrys_age_l927_92761


namespace parallelogram_reflection_theorem_l927_92740

-- Define the type for 2D points
def Point := ℝ × ℝ

-- Define the reflection across x-axis
def reflectX (p : Point) : Point :=
  (p.1, -p.2)

-- Define the reflection across y = x - 2
def reflectYXMinus2 (p : Point) : Point :=
  let p' := (p.1, p.2 + 2)  -- Translate up by 2
  let p'' := (p'.2, p'.1)   -- Reflect across y = x
  (p''.1, p''.2 - 2)        -- Translate back down by 2

-- Define the theorem
theorem parallelogram_reflection_theorem (A B C D : Point)
  (hA : A = (3, 7))
  (hB : B = (5, 11))
  (hC : C = (7, 7))
  (hD : D = (5, 3))
  : reflectYXMinus2 (reflectX D) = (-1, 3) := by
  sorry

end parallelogram_reflection_theorem_l927_92740


namespace volunteer_allocation_schemes_l927_92764

/-- The number of ways to allocate volunteers to projects -/
def allocate_volunteers (n_volunteers : ℕ) (n_projects : ℕ) : ℕ :=
  (n_volunteers.choose 2) * (n_projects.factorial)

/-- Theorem stating that allocating 5 volunteers to 4 projects results in 240 schemes -/
theorem volunteer_allocation_schemes :
  allocate_volunteers 5 4 = 240 :=
by sorry

end volunteer_allocation_schemes_l927_92764


namespace decreasing_cubic_function_parameter_bound_l927_92726

/-- Given a function f(x) = ax³ - x that is decreasing on ℝ, prove that a ≤ 0 -/
theorem decreasing_cubic_function_parameter_bound (a : ℝ) :
  (∀ x : ℝ, HasDerivAt (fun x => a * x^3 - x) (3 * a * x^2 - 1) x) →
  (∀ x y : ℝ, x < y → (a * x^3 - x) > (a * y^3 - y)) →
  a ≤ 0 :=
by sorry

end decreasing_cubic_function_parameter_bound_l927_92726
