import Mathlib

namespace legs_minus_twice_heads_l926_92622

/-- Represents the number of legs for each animal type -/
def legs_per_animal : Fin 3 → ℕ
  | 0 => 2  -- Ducks
  | 1 => 4  -- Cows
  | 2 => 4  -- Buffaloes

/-- The group of animals -/
structure AnimalGroup where
  ducks : ℕ
  cows : ℕ
  buffaloes : ℕ
  buffalo_count_eq : buffaloes = 24

/-- Total number of heads in the group -/
def total_heads (g : AnimalGroup) : ℕ :=
  g.ducks + g.cows + g.buffaloes

/-- Total number of legs in the group -/
def total_legs (g : AnimalGroup) : ℕ :=
  g.ducks * legs_per_animal 0 + g.cows * legs_per_animal 1 + g.buffaloes * legs_per_animal 2

/-- The statement to be proven -/
theorem legs_minus_twice_heads (g : AnimalGroup) :
  total_legs g > 2 * total_heads g →
  total_legs g - 2 * total_heads g = 2 * g.cows + 48 :=
sorry

end legs_minus_twice_heads_l926_92622


namespace total_tiles_on_floor_l926_92670

/-- Represents a square floor with a border of tiles -/
structure BorderedFloor where
  /-- The side length of the square floor -/
  side_length : ℕ
  /-- The width of the border in tiles -/
  border_width : ℕ
  /-- The number of tiles in the border -/
  border_tiles : ℕ

/-- Theorem: Given a square floor with a 1-tile wide black border containing 204 tiles, 
    the total number of tiles on the floor is 2704 -/
theorem total_tiles_on_floor (floor : BorderedFloor) 
  (h1 : floor.border_width = 1)
  (h2 : floor.border_tiles = 204) : 
  floor.side_length^2 = 2704 := by
  sorry

#check total_tiles_on_floor

end total_tiles_on_floor_l926_92670


namespace sin_translation_l926_92671

/-- Given a function f obtained by translating the graph of y = sin 2x
    1 unit left and 1 unit upward, prove that f(x) = sin(2x+2)+1 for all real x. -/
theorem sin_translation (f : ℝ → ℝ) 
  (h : ∀ x, f x = (fun y ↦ Real.sin (2 * y)) (x + 1) + 1) :
  ∀ x, f x = Real.sin (2 * x + 2) + 1 := by
  sorry

end sin_translation_l926_92671


namespace valid_sequences_of_length_21_l926_92610

/-- Counts valid sequences of 0s and 1s of given length -/
def countValidSequences (n : ℕ) : ℕ :=
  if n ≤ 4 then 0
  else if n = 5 then 1
  else if n = 6 then 2
  else if n = 7 then 2
  else countValidSequences (n - 4) + 2 * countValidSequences (n - 5) + countValidSequences (n - 6)

/-- Theorem stating the number of valid sequences of length 21 -/
theorem valid_sequences_of_length_21 :
  countValidSequences 21 = 114 := by
  sorry

end valid_sequences_of_length_21_l926_92610


namespace solve_for_A_l926_92603

theorem solve_for_A : ∃ A : ℝ, (10 - A = 6) ∧ (A = 4) := by sorry

end solve_for_A_l926_92603


namespace elderly_sample_count_l926_92636

/-- Represents the composition of employees in a unit -/
structure EmployeeComposition where
  total : ℕ
  young : ℕ
  middleAged : ℕ
  elderly : ℕ

/-- Represents a stratified sample of employees -/
structure StratifiedSample where
  youngSampled : ℕ
  elderlySampled : ℕ

/-- Calculates the number of elderly employees in a stratified sample -/
def calculateElderlySampled (comp : EmployeeComposition) (sample : StratifiedSample) : ℚ :=
  (comp.elderly : ℚ) / comp.total * sample.youngSampled

theorem elderly_sample_count (comp : EmployeeComposition) (sample : StratifiedSample) :
  comp.total = 430 →
  comp.young = 160 →
  comp.middleAged = 2 * comp.elderly →
  sample.youngSampled = 32 →
  calculateElderlySampled comp sample = 18 := by
  sorry

end elderly_sample_count_l926_92636


namespace part_one_part_two_l926_92692

/-- The function f(x) = |a-4x| + |2a+x| -/
def f (a : ℝ) (x : ℝ) : ℝ := |a - 4*x| + |2*a + x|

/-- Part I: When a = 1, f(x) ≥ 3 if and only if x ≤ 0 or x ≥ 2/5 -/
theorem part_one : 
  ∀ x : ℝ, f 1 x ≥ 3 ↔ x ≤ 0 ∨ x ≥ 2/5 := by sorry

/-- Part II: For all x ≠ 0 and all a, f(x) + f(-1/x) ≥ 10 -/
theorem part_two : 
  ∀ a x : ℝ, x ≠ 0 → f a x + f a (-1/x) ≥ 10 := by sorry

end part_one_part_two_l926_92692


namespace min_sum_squares_l926_92649

theorem min_sum_squares (x y : ℝ) :
  x^2 - y^2 + 6*x + 4*y + 5 = 0 →
  ∃ (min : ℝ), min = 0.5 ∧ ∀ (a b : ℝ), a^2 - b^2 + 6*a + 4*b + 5 = 0 → a^2 + b^2 ≥ min :=
by sorry

end min_sum_squares_l926_92649


namespace extremum_value_theorem_l926_92607

/-- The function f(x) = x sin x achieves an extremum at x₀ -/
def has_extremum (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ ε > 0, ∀ x, |x - x₀| < ε → f x ≤ f x₀ ∨ f x ≥ f x₀

theorem extremum_value_theorem (x₀ : ℝ) :
  has_extremum (fun x => x * Real.sin x) x₀ →
  (1 + x₀^2) * (1 + Real.cos (2 * x₀)) = 2 :=
by sorry

end extremum_value_theorem_l926_92607


namespace factor_x_squared_minus_64_l926_92644

theorem factor_x_squared_minus_64 (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := by
  sorry

end factor_x_squared_minus_64_l926_92644


namespace not_monotone_decreasing_f_maps_A_to_B_l926_92680

-- Define the sets A and B
def A : Set ℝ := {1, 4}
def B : Set ℝ := {1, -1, 2, -2}

-- Define the function f
def f (x : ℝ) : ℝ := (x^2)^(1/7)

-- Theorem 1
theorem not_monotone_decreasing (f : ℝ → ℝ) (h : f 2 < f 3) :
  ¬(∀ x y : ℝ, x ≤ y → f x ≥ f y) :=
sorry

-- Theorem 2
theorem f_maps_A_to_B :
  ∀ x ∈ A, f x ∈ B :=
sorry

end not_monotone_decreasing_f_maps_A_to_B_l926_92680


namespace greatest_root_of_f_l926_92669

def f (x : ℝ) : ℝ := 21 * x^4 - 20 * x^2 + 3

theorem greatest_root_of_f :
  ∃ (r : ℝ), r = Real.sqrt 21 / 7 ∧
  f r = 0 ∧
  ∀ (x : ℝ), f x = 0 → x ≤ r :=
sorry

end greatest_root_of_f_l926_92669


namespace problem_1_l926_92693

theorem problem_1 : (-13/2 : ℚ) * (4/13 : ℚ) - 8 / |(-4 + 2)| = -6 := by sorry

end problem_1_l926_92693


namespace amusement_park_tickets_l926_92614

/-- Proves that the number of children in the group is 15 given the specified conditions -/
theorem amusement_park_tickets (total_cost adult_price child_price adult_child_difference : ℕ) 
  (h1 : total_cost = 720)
  (h2 : adult_price = 15)
  (h3 : child_price = 8)
  (h4 : adult_child_difference = 25) :
  ∃ (num_children : ℕ), 
    (num_children + adult_child_difference) * adult_price + num_children * child_price = total_cost ∧ 
    num_children = 15 := by
  sorry

end amusement_park_tickets_l926_92614


namespace solution_set_part1_range_of_a_part2_l926_92664

-- Define the function f
def f (a x : ℝ) : ℝ := |x + a| + |x - 1|

-- Part 1: Solution set when a = 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x < 3} = Set.Ioo (-3/2) (3/2) := by sorry

-- Part 2: Range of a for which f(x) ≥ 3 for all x
theorem range_of_a_part2 :
  ∀ a : ℝ, (∀ x : ℝ, f a x ≥ 3) ↔ (a ≥ 2 ∨ a ≤ -4) := by sorry

end solution_set_part1_range_of_a_part2_l926_92664


namespace min_distance_to_line_l926_92673

open Real

theorem min_distance_to_line :
  let line := {(x, y) : ℝ × ℝ | 4 * x - 3 * y - 5 * sqrt 2 = 0}
  ∃ (m n : ℝ), (m, n) ∈ line ∧ ∀ (x y : ℝ), (x, y) ∈ line → m^2 + n^2 ≤ x^2 + y^2 ∧ m^2 + n^2 = 2 :=
by sorry

end min_distance_to_line_l926_92673


namespace imaginary_part_of_one_plus_i_cubed_l926_92659

theorem imaginary_part_of_one_plus_i_cubed (i : ℂ) : Complex.im ((1 + i)^3) = 2 :=
by sorry

end imaginary_part_of_one_plus_i_cubed_l926_92659


namespace root_sum_theorem_l926_92654

theorem root_sum_theorem (a b : ℝ) (ha : a ≠ 0) (h : a^2 + b*a - 2*a = 0) : a + b = 2 := by
  sorry

end root_sum_theorem_l926_92654


namespace peach_tart_fraction_l926_92694

theorem peach_tart_fraction (total : ℝ) (cherry : ℝ) (blueberry : ℝ) 
  (h1 : total = 0.91)
  (h2 : cherry = 0.08)
  (h3 : blueberry = 0.75) :
  total - (cherry + blueberry) = 0.08 := by
  sorry

end peach_tart_fraction_l926_92694


namespace least_whole_number_ratio_l926_92615

theorem least_whole_number_ratio (x : ℕ) : 
  (x > 0 ∧ (6 - x : ℚ) / (7 - x) < 16 / 21) ↔ x ≥ 3 :=
by sorry

end least_whole_number_ratio_l926_92615


namespace greatest_n_for_2008_l926_92637

-- Define the sum of digits function
def sum_of_digits (a : ℕ) : ℕ := sorry

-- Define the sequence a_n
def a : ℕ → ℕ
  | 0 => sorry  -- Initial value, not specified in the problem
  | n + 1 => a n + sum_of_digits (a n)

-- Theorem statement
theorem greatest_n_for_2008 : (∃ n : ℕ, a n = 2008) ∧ (∀ m : ℕ, m > 6 → a m ≠ 2008) := by sorry

end greatest_n_for_2008_l926_92637


namespace horseshoe_profit_calculation_l926_92640

/-- Calculates the profit for a horseshoe manufacturing company --/
theorem horseshoe_profit_calculation 
  (initial_outlay : ℕ) 
  (manufacturing_cost_per_set : ℕ) 
  (selling_price_per_set : ℕ) 
  (sets_produced_and_sold : ℕ) :
  initial_outlay = 10000 →
  manufacturing_cost_per_set = 20 →
  selling_price_per_set = 50 →
  sets_produced_and_sold = 500 →
  (selling_price_per_set * sets_produced_and_sold) - 
  (initial_outlay + manufacturing_cost_per_set * sets_produced_and_sold) = 5000 :=
by sorry

end horseshoe_profit_calculation_l926_92640


namespace marks_total_spent_l926_92674

/-- Represents the purchase of a fruit with its quantity and price per pound -/
structure FruitPurchase where
  quantity : ℝ
  price_per_pound : ℝ

/-- Calculates the total cost of a fruit purchase -/
def total_cost (purchase : FruitPurchase) : ℝ :=
  purchase.quantity * purchase.price_per_pound

/-- Represents Mark's shopping list -/
structure ShoppingList where
  tomatoes : FruitPurchase
  apples : FruitPurchase
  oranges : FruitPurchase

/-- Calculates the total cost of all items in the shopping list -/
def total_spent (list : ShoppingList) : ℝ :=
  total_cost list.tomatoes + total_cost list.apples + total_cost list.oranges

/-- Mark's actual shopping list -/
def marks_shopping : ShoppingList :=
  { tomatoes := { quantity := 3, price_per_pound := 4.5 }
  , apples := { quantity := 7, price_per_pound := 3.25 }
  , oranges := { quantity := 4, price_per_pound := 2.75 }
  }

/-- Theorem: The total amount Mark spent is $47.25 -/
theorem marks_total_spent :
  total_spent marks_shopping = 47.25 := by
  sorry


end marks_total_spent_l926_92674


namespace consecutive_integers_deduction_l926_92606

theorem consecutive_integers_deduction (n : ℕ) (avg : ℚ) (new_avg : ℚ) : 
  n = 30 → 
  avg = 50 → 
  new_avg = 34.3 → 
  let sum := n * avg
  let first_deduction := 29
  let last_deduction := 1
  let deduction_sum := n.pred / 2 * (first_deduction + last_deduction)
  let final_deduction := 6 + 12 + 18
  let new_sum := sum - deduction_sum - final_deduction
  new_avg = new_sum / n := by sorry

end consecutive_integers_deduction_l926_92606


namespace no_hammers_loaded_l926_92623

theorem no_hammers_loaded (crate_capacity : ℕ) (num_crates : ℕ) (nail_bags : ℕ) (nail_weight : ℕ)
  (plank_bags : ℕ) (plank_weight : ℕ) (leave_out : ℕ) (hammer_weight : ℕ) :
  crate_capacity = 20 →
  num_crates = 15 →
  nail_bags = 4 →
  nail_weight = 5 →
  plank_bags = 10 →
  plank_weight = 30 →
  leave_out = 80 →
  hammer_weight = 5 →
  (∃ (loaded_planks : ℕ), 
    loaded_planks ≤ plank_bags * plank_weight ∧
    crate_capacity * num_crates - leave_out = nail_bags * nail_weight + loaded_planks) →
  (∀ (hammer_bags : ℕ), 
    crate_capacity * num_crates - leave_out < 
      nail_bags * nail_weight + plank_bags * plank_weight - leave_out + hammer_bags * hammer_weight) :=
by sorry

end no_hammers_loaded_l926_92623


namespace equation_solution_l926_92682

theorem equation_solution (x : ℝ) : (x - 2) * (x - 3) = 0 ↔ x = 2 ∨ x = 3 := by
  sorry

end equation_solution_l926_92682


namespace lena_always_greater_probability_lena_greater_l926_92638

def lena_set : Finset ℕ := {7, 8, 9}
def jonah_set : Finset ℕ := {2, 4, 6}

def lena_result (a b : ℕ) : ℕ := a * b

def jonah_result (a b c : ℕ) : ℕ := (a + b) * c

theorem lena_always_greater :
  ∀ (a b : ℕ) (c d e : ℕ),
    a ∈ lena_set → b ∈ lena_set → a ≠ b →
    c ∈ jonah_set → d ∈ jonah_set → e ∈ jonah_set →
    c ≠ d → c ≠ e → d ≠ e →
    lena_result a b > jonah_result c d e :=
by
  sorry

theorem probability_lena_greater : ℚ :=
  1

#check lena_always_greater
#check probability_lena_greater

end lena_always_greater_probability_lena_greater_l926_92638


namespace rectangle_max_area_l926_92685

theorem rectangle_max_area (length width : ℝ) :
  length > 0 → width > 0 → length + width = 18 →
  length * width ≤ 81 ∧
  (length * width = 81 ↔ length = 9 ∧ width = 9) :=
by sorry

end rectangle_max_area_l926_92685


namespace election_vote_count_l926_92620

/-- Represents an election with two candidates -/
structure TwoCandidateElection where
  totalVotes : ℕ
  loserPercentage : ℚ
  voteDifference : ℕ

/-- 
Theorem: In a two-candidate election where the losing candidate received 40% of the votes
and lost by 5000 votes, the total number of votes cast was 25000.
-/
theorem election_vote_count (e : TwoCandidateElection) 
  (h1 : e.loserPercentage = 40 / 100)
  (h2 : e.voteDifference = 5000) : 
  e.totalVotes = 25000 := by
  sorry

#eval (40 : ℚ) / 100  -- To verify the rational number representation

end election_vote_count_l926_92620


namespace n_in_interval_l926_92604

def is_repeating_decimal (q : ℚ) (period : ℕ) : Prop :=
  ∃ (k : ℕ), q * (10^period - 1) = k ∧ k < 10^period

theorem n_in_interval (n : ℕ) :
  n < 500 →
  is_repeating_decimal (1 / n) 4 →
  is_repeating_decimal (1 / (n + 4)) 2 →
  n ∈ Set.Icc 1 125 :=
sorry

end n_in_interval_l926_92604


namespace cos_A_eq_one_l926_92619

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_convex (q : Quadrilateral) : Prop := sorry

def angle_A_eq_C (q : Quadrilateral) : Prop :=
  sorry -- ∠A = ∠C

def side_AB_eq_240 (q : Quadrilateral) : ℝ :=
  sorry -- Distance between A and B

def side_CD_eq_240 (q : Quadrilateral) : ℝ :=
  sorry -- Distance between C and D

def side_AD_ne_BC (q : Quadrilateral) : Prop :=
  sorry -- AD ≠ BC

def perimeter_eq_960 (q : Quadrilateral) : ℝ :=
  sorry -- Perimeter of ABCD

-- Theorem statement
theorem cos_A_eq_one (q : Quadrilateral) 
  (h_convex : is_convex q)
  (h_angle : angle_A_eq_C q)
  (h_AB : side_AB_eq_240 q = 240)
  (h_CD : side_CD_eq_240 q = 240)
  (h_AD_ne_BC : side_AD_ne_BC q)
  (h_perimeter : perimeter_eq_960 q = 960) :
  let cos_A := sorry -- Definition of cos A for the quadrilateral
  cos_A = 1 := by sorry

end cos_A_eq_one_l926_92619


namespace tetrahedra_triangle_inequality_l926_92652

/-- A finite graph -/
structure FiniteGraph where
  -- We don't need to specify the internal structure of the graph
  -- as it's not directly used in the theorem statement

/-- The number of triangles in a finite graph -/
def numTriangles (G : FiniteGraph) : ℕ := sorry

/-- The number of tetrahedra in a finite graph -/
def numTetrahedra (G : FiniteGraph) : ℕ := sorry

/-- The main theorem stating the inequality between tetrahedra and triangles -/
theorem tetrahedra_triangle_inequality (G : FiniteGraph) :
  (numTetrahedra G)^3 ≤ (3/32) * (numTriangles G)^4 := by
  sorry

end tetrahedra_triangle_inequality_l926_92652


namespace square_side_length_l926_92699

/-- Given a rectangle with width 2 and a square placed next to it,
    if the total length of the bottom side is 7,
    then the side length of the square is 5. -/
theorem square_side_length (rectangle_width square_side total_length : ℝ) : 
  rectangle_width = 2 →
  total_length = 7 →
  total_length = rectangle_width + square_side →
  square_side = 5 := by
sorry


end square_side_length_l926_92699


namespace sequence_inequality_l926_92602

/-- S(n,m) is the number of sequences of length n consisting of 0 and 1 
    where there exists a 0 in any consecutive m digits -/
def S (n m : ℕ) : ℕ := sorry

/-- Theorem statement -/
theorem sequence_inequality (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  S (2015 * n) n * S (2015 * m) m ≥ S (2015 * n) m * S (2015 * m) n := by sorry

end sequence_inequality_l926_92602


namespace pizza_price_problem_l926_92653

/-- Proves the price of large pizza slices given the conditions of the problem -/
theorem pizza_price_problem (small_price : ℕ) (total_slices : ℕ) (total_revenue : ℕ) (small_slices : ℕ) :
  small_price = 150 →
  total_slices = 5000 →
  total_revenue = 1050000 →
  small_slices = 2000 →
  (total_revenue - small_price * small_slices) / (total_slices - small_slices) = 250 := by
sorry

end pizza_price_problem_l926_92653


namespace units_digit_G_1000_l926_92698

/-- The function G_n defined as 3^(3^n) + 1 -/
def G (n : ℕ) : ℕ := 3^(3^n) + 1

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- Theorem stating that the units digit of G(1000) is 2 -/
theorem units_digit_G_1000 : unitsDigit (G 1000) = 2 := by
  sorry

end units_digit_G_1000_l926_92698


namespace function_inequality_implies_squares_inequality_l926_92645

theorem function_inequality_implies_squares_inequality 
  (a b A B : ℝ) 
  (h : ∀ x : ℝ, 1 - a * Real.cos x - b * Real.sin x - A * Real.cos (2 * x) - B * Real.sin (2 * x) ≥ 0) : 
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 := by
  sorry

end function_inequality_implies_squares_inequality_l926_92645


namespace ten_factorial_minus_nine_factorial_l926_92609

theorem ten_factorial_minus_nine_factorial : Nat.factorial 10 - Nat.factorial 9 = 3265920 := by
  sorry

end ten_factorial_minus_nine_factorial_l926_92609


namespace father_double_son_age_l926_92691

/-- Represents the ages of a father and son, and the time until the father's age is twice the son's. -/
structure FatherSonAges where
  sonAge : ℕ
  fatherAge : ℕ
  yearsUntilDouble : ℕ

/-- The condition that the father is 25 years older than the son -/
def ageDifference (ages : FatherSonAges) : Prop :=
  ages.fatherAge = ages.sonAge + 25

/-- The condition that after a certain number of years, the father's age will be twice the son's -/
def doubleAgeCondition (ages : FatherSonAges) : Prop :=
  ages.fatherAge + ages.yearsUntilDouble = 2 * (ages.sonAge + ages.yearsUntilDouble)

/-- The main theorem stating that given the initial conditions, it will take 2 years for the father's age to be twice the son's -/
theorem father_double_son_age :
  ∀ (ages : FatherSonAges),
  ages.sonAge = 23 →
  ageDifference ages →
  doubleAgeCondition ages →
  ages.yearsUntilDouble = 2 :=
by
  sorry


end father_double_son_age_l926_92691


namespace davis_oldest_child_age_l926_92600

/-- The age of the oldest Davis child given the conditions -/
def oldest_child_age (avg_age : ℕ) (younger_child1 : ℕ) (younger_child2 : ℕ) : ℕ :=
  3 * avg_age - younger_child1 - younger_child2

/-- Theorem stating the age of the oldest Davis child -/
theorem davis_oldest_child_age :
  oldest_child_age 10 7 9 = 14 := by
  sorry

end davis_oldest_child_age_l926_92600


namespace sum_of_digits_odd_numbers_to_10000_l926_92678

/-- Sum of digits function for natural numbers -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Predicate to check if a natural number is odd -/
def isOdd (n : ℕ) : Prop := n % 2 = 1

/-- The sum of digits of all odd numbers from 1 to 10000 -/
def sumOfDigitsOddNumbers : ℕ := sorry

/-- Theorem stating that the sum of digits of all odd numbers from 1 to 10000 is 97500 -/
theorem sum_of_digits_odd_numbers_to_10000 :
  sumOfDigitsOddNumbers = 97500 := by sorry

end sum_of_digits_odd_numbers_to_10000_l926_92678


namespace count_sevens_up_to_2017_l926_92683

/-- Count of digit 7 in a natural number -/
def count_sevens (n : ℕ) : ℕ := sorry

/-- Sum of count_sevens for all numbers from 1 to n -/
def sum_count_sevens (n : ℕ) : ℕ := sorry

/-- The main theorem stating the count of digit 7 in numbers from 1 to 2017 -/
theorem count_sevens_up_to_2017 : sum_count_sevens 2017 = 602 := by sorry

end count_sevens_up_to_2017_l926_92683


namespace sine_cosine_inequality_range_l926_92647

theorem sine_cosine_inequality_range (θ : Real) :
  θ ∈ Set.Ioo 0 (2 * Real.pi) →
  Real.sin θ ^ 3 - Real.cos θ ^ 3 > (Real.cos θ ^ 5 - Real.sin θ ^ 5) / 7 →
  θ ∈ Set.Ioo (Real.pi / 4) (5 * Real.pi / 4) := by
  sorry

end sine_cosine_inequality_range_l926_92647


namespace jeff_rental_duration_l926_92648

/-- Represents the rental scenario for Jeff's apartment. -/
structure RentalScenario where
  initialRent : ℕ  -- Monthly rent for the first 3 years
  raisedRent : ℕ   -- Monthly rent after the raise
  initialYears : ℕ -- Number of years at the initial rent
  totalPaid : ℕ    -- Total amount paid over the entire rental period

/-- Calculates the total number of years Jeff rented the apartment. -/
def totalRentalYears (scenario : RentalScenario) : ℕ :=
  scenario.initialYears + 
  ((scenario.totalPaid - scenario.initialRent * scenario.initialYears * 12) / (scenario.raisedRent * 12))

/-- Theorem stating that Jeff rented the apartment for 5 years. -/
theorem jeff_rental_duration (scenario : RentalScenario) 
  (h1 : scenario.initialRent = 300)
  (h2 : scenario.raisedRent = 350)
  (h3 : scenario.initialYears = 3)
  (h4 : scenario.totalPaid = 19200) :
  totalRentalYears scenario = 5 := by
  sorry

end jeff_rental_duration_l926_92648


namespace polynomial_simplification_l926_92626

theorem polynomial_simplification (y : ℝ) :
  (3 * y - 2) * (5 * y^12 + 3 * y^11 + y^10 + 2 * y^9) =
  15 * y^13 - y^12 - 3 * y^11 + 4 * y^10 - 4 * y^9 := by
  sorry

end polynomial_simplification_l926_92626


namespace min_value_implies_a_l926_92676

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1/x + a * Real.log x

theorem min_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc (1/2 : ℝ) 1, f a x ≥ 0) ∧
  (∃ x ∈ Set.Icc (1/2 : ℝ) 1, f a x = 0) →
  a = 2 / Real.log 2 := by
  sorry

end min_value_implies_a_l926_92676


namespace sequence_second_term_l926_92616

/-- Given a sequence {a_n} with sum of first n terms S_n, prove a_2 = 4 -/
theorem sequence_second_term (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, S n = 2 * (a n - 1)) → a 2 = 4 := by
  sorry

end sequence_second_term_l926_92616


namespace smallest_average_is_16_5_l926_92679

/-- A function that generates all valid combinations of three single-digit
    and three double-digit numbers from the digits 1 to 9 without repetition -/
def generateValidCombinations : List (List ℕ) := sorry

/-- Calculates the average of a list of numbers -/
def average (numbers : List ℕ) : ℚ :=
  (numbers.sum : ℚ) / numbers.length

/-- Theorem stating that the smallest possible average is 16.5 -/
theorem smallest_average_is_16_5 :
  let allCombinations := generateValidCombinations
  let averages := allCombinations.map average
  averages.minimum? = some (33/2) := by sorry

end smallest_average_is_16_5_l926_92679


namespace f_has_unique_zero_l926_92663

noncomputable def f (x : ℝ) := Real.log x + 2 * x - 6

theorem f_has_unique_zero :
  ∃! x : ℝ, x > 0 ∧ f x = 0 :=
sorry

end f_has_unique_zero_l926_92663


namespace f_properties_l926_92657

noncomputable def f (x : ℝ) := (1/3) * x^3 - 2 * x^2 + 3 * x + 2/3

theorem f_properties :
  (∀ x : ℝ, f x ≤ 2) ∧
  (∀ b : ℝ, 
    (b ≤ 0 ∨ b > (9 + Real.sqrt 33) / 6 → 
      (∀ x ∈ Set.Icc b (b + 1), f x ≤ (b^3 / 3) - b^2 + 2) ∧
      ∃ x ∈ Set.Icc b (b + 1), f x = (b^3 / 3) - b^2 + 2) ∧
    (0 < b ∧ b ≤ 1 → 
      (∀ x ∈ Set.Icc b (b + 1), f x ≤ 2) ∧
      ∃ x ∈ Set.Icc b (b + 1), f x = 2) ∧
    (1 < b ∧ b ≤ (9 + Real.sqrt 33) / 6 → 
      (∀ x ∈ Set.Icc b (b + 1), f x ≤ (b^3 / 3) - 2 * b^2 + 3 * b + 2/3) ∧
      ∃ x ∈ Set.Icc b (b + 1), f x = (b^3 / 3) - 2 * b^2 + 3 * b + 2/3)) := by
  sorry

end f_properties_l926_92657


namespace complex_magnitude_problem_l926_92697

theorem complex_magnitude_problem (z : ℂ) : z = Complex.I * (Complex.I - 1) → Complex.abs (z - 1) = Real.sqrt 5 := by
  sorry

end complex_magnitude_problem_l926_92697


namespace ticket_distribution_count_l926_92633

/-- The number of ways to distribute 5 consecutive movie tickets to 5 people. -/
def distribute_tickets : ℕ :=
  /- Number of ways to group tickets -/ 4 *
  /- Number of ways to order A and B -/ 2 *
  /- Number of ways to permute remaining tickets -/ 6

/-- Theorem stating that there are 48 ways to distribute the tickets. -/
theorem ticket_distribution_count :
  distribute_tickets = 48 := by
  sorry

end ticket_distribution_count_l926_92633


namespace total_payment_calculation_l926_92689

def worker_count : ℝ := 2.5
def hourly_rate : ℝ := 15
def daily_hours : List ℝ := [12, 10, 8, 6, 14]

theorem total_payment_calculation :
  worker_count * hourly_rate * (daily_hours.sum) = 1875 := by sorry

end total_payment_calculation_l926_92689


namespace prob_at_least_one_head_in_five_tosses_l926_92624

/-- The probability of getting at least one head in five coin tosses -/
theorem prob_at_least_one_head_in_five_tosses : 
  let p_head : ℚ := 1/2  -- probability of getting heads on a single toss
  let n : ℕ := 5        -- number of coin tosses
  1 - (1 - p_head)^n = 31/32 :=
by sorry

end prob_at_least_one_head_in_five_tosses_l926_92624


namespace initial_alcohol_percentage_l926_92695

/-- Proves that the initial alcohol percentage is 5% given the problem conditions -/
theorem initial_alcohol_percentage
  (initial_volume : ℝ)
  (added_alcohol : ℝ)
  (added_water : ℝ)
  (final_percentage : ℝ)
  (h1 : initial_volume = 40)
  (h2 : added_alcohol = 5.5)
  (h3 : added_water = 4.5)
  (h4 : final_percentage = 15)
  (h5 : final_percentage / 100 * (initial_volume + added_alcohol + added_water) =
        initial_percentage / 100 * initial_volume + added_alcohol) :
  initial_percentage = 5 :=
by sorry


end initial_alcohol_percentage_l926_92695


namespace prism_volume_l926_92687

/-- The volume of a right rectangular prism given its face areas -/
theorem prism_volume (a b c : ℝ) 
  (h1 : a * b = 36)
  (h2 : a * c = 72)
  (h3 : b * c = 48) :
  a * b * c = 352.8 := by
sorry

end prism_volume_l926_92687


namespace composite_number_l926_92605

theorem composite_number : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ (2^17 + 2^5 - 1 = a * b) := by
  sorry

end composite_number_l926_92605


namespace reflection_of_circle_center_l926_92658

/-- Reflects a point (x, y) about the line y = -x --/
def reflect_about_negative_diagonal (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.2), -(p.1))

/-- The original center of the circle --/
def original_center : ℝ × ℝ := (4, -3)

/-- The expected reflected center of the circle --/
def expected_reflected_center : ℝ × ℝ := (3, -4)

theorem reflection_of_circle_center :
  reflect_about_negative_diagonal original_center = expected_reflected_center := by
  sorry

end reflection_of_circle_center_l926_92658


namespace coefficient_sum_of_squares_l926_92662

theorem coefficient_sum_of_squares (p q r s t u : ℤ) :
  (∀ x, 343 * x^3 + 64 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) →
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 3506 := by
  sorry

end coefficient_sum_of_squares_l926_92662


namespace no_integer_solution_l926_92618

theorem no_integer_solution : ∀ m n : ℤ, m^2 ≠ n^5 - 4 := by
  sorry

end no_integer_solution_l926_92618


namespace absolute_difference_bound_l926_92651

theorem absolute_difference_bound (x y s t : ℝ) 
  (hx : |x - s| < t) (hy : |y - s| < t) : |x - y| < 2*t := by
  sorry

end absolute_difference_bound_l926_92651


namespace trigonometric_equation_solution_l926_92639

open Real

theorem trigonometric_equation_solution (x : ℝ) :
  (2 * cos (π + x) - 5 * cos ((3/2) * π - x)) / (cos ((3/2) * π + x) - cos (π - x)) = 3/2 ↔
  ∃ k : ℤ, x = (π/4) * (4 * k + 1) := by sorry

end trigonometric_equation_solution_l926_92639


namespace cube_root_of_negative_eight_l926_92681

theorem cube_root_of_negative_eight (x : ℝ) : x^3 = -8 ↔ x = -2 := by
  sorry

end cube_root_of_negative_eight_l926_92681


namespace min_value_parabola_l926_92611

theorem min_value_parabola (x y : ℝ) (h : x^2 = 4*y) :
  ∃ (min : ℝ), min = 2 ∧ ∀ (x' y' : ℝ), x'^2 = 4*y' →
    Real.sqrt ((x' - 3)^2 + (y' - 1)^2) + y' ≥ min := by
  sorry

end min_value_parabola_l926_92611


namespace evaluate_expression_l926_92677

theorem evaluate_expression : 6 - 8 * (9 - 4^2) * 5 = 286 := by sorry

end evaluate_expression_l926_92677


namespace sum_of_three_pentagons_l926_92608

/-- The value of a square -/
def square_value : ℚ := sorry

/-- The value of a pentagon -/
def pentagon_value : ℚ := sorry

/-- First equation: 3 squares + 2 pentagons = 27 -/
axiom eq1 : 3 * square_value + 2 * pentagon_value = 27

/-- Second equation: 2 squares + 3 pentagons = 25 -/
axiom eq2 : 2 * square_value + 3 * pentagon_value = 25

/-- Theorem: The sum of three pentagons equals 63/5 -/
theorem sum_of_three_pentagons : 3 * pentagon_value = 63 / 5 := by sorry

end sum_of_three_pentagons_l926_92608


namespace stretch_circle_to_ellipse_l926_92686

/-- Given a circle A and a stretch transformation, prove the equation of the resulting curve C -/
theorem stretch_circle_to_ellipse (x y x' y' : ℝ) :
  (x^2 + y^2 = 1) →  -- Circle A equation
  (x' = 2*x) →       -- Stretch transformation for x
  (y' = 3*y) →       -- Stretch transformation for y
  (x'^2 / 4 + y'^2 / 9 = 1) -- Resulting curve C equation
:= by sorry

end stretch_circle_to_ellipse_l926_92686


namespace intersection_empty_implies_m_leq_neg_one_l926_92612

def M (m : ℝ) : Set ℝ := {x | x - m < 0}

def N : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 2*x}

theorem intersection_empty_implies_m_leq_neg_one (m : ℝ) :
  M m ∩ N = ∅ → m ≤ -1 := by
  sorry

end intersection_empty_implies_m_leq_neg_one_l926_92612


namespace birds_in_tree_l926_92641

theorem birds_in_tree (initial_birds : ℝ) (birds_flew_away : ℝ) (birds_remaining : ℝ) : 
  birds_flew_away = 14.0 → birds_remaining = 7 → initial_birds = birds_flew_away + birds_remaining :=
by sorry

end birds_in_tree_l926_92641


namespace comparison_of_special_points_l926_92672

theorem comparison_of_special_points (a b c : Real) 
  (ha : 0 < a ∧ a < Real.pi / 2)
  (hb : 0 < b ∧ b < Real.pi / 2)
  (hc : 0 < c ∧ c < Real.pi / 2)
  (eq_a : a = Real.cos a)
  (eq_b : b = Real.sin (Real.cos b))
  (eq_c : c = Real.cos (Real.sin c)) :
  b < a ∧ a < c :=
by sorry

end comparison_of_special_points_l926_92672


namespace greatest_rational_root_of_quadratic_l926_92643

theorem greatest_rational_root_of_quadratic (a b c : ℕ) 
  (ha : a ≤ 100) (hb : b ≤ 100) (hc : c ≤ 100) (ha_pos : a > 0) :
  ∃ (x : ℚ), x = -1/99 ∧ 
    (∀ (y : ℚ), y ≠ x → a * y^2 + b * y + c = 0 → y < x) ∧
    a * x^2 + b * x + c = 0 :=
sorry

end greatest_rational_root_of_quadratic_l926_92643


namespace quadratic_equation_roots_l926_92635

theorem quadratic_equation_roots (a b : ℝ) (h1 : a ≠ 0) :
  (∃ x : ℝ, a * x^2 = b ∧ x = 2) → (∃ y : ℝ, a * y^2 = b ∧ y = -2) :=
by sorry

end quadratic_equation_roots_l926_92635


namespace vector_calculation_l926_92684

/-- Given vectors a, b, c, and e in a vector space, 
    where a = 5e, b = -3e, and c = 4e,
    prove that 2a - 3b + c = 23e -/
theorem vector_calculation 
  (V : Type*) [AddCommGroup V] [Module ℝ V] 
  (e : V) 
  (a b c : V) 
  (ha : a = 5 • e) 
  (hb : b = -3 • e) 
  (hc : c = 4 • e) : 
  2 • a - 3 • b + c = 23 • e := by
  sorry

end vector_calculation_l926_92684


namespace bus_car_speed_problem_l926_92628

/-- Proves that given the conditions of the problem, the bus speed is 50 km/h and the car speed is 75 km/h -/
theorem bus_car_speed_problem (distance : ℝ) (delay : ℝ) (speed_ratio : ℝ) 
  (h1 : distance = 50)
  (h2 : delay = 1/3)
  (h3 : speed_ratio = 1.5)
  (h4 : ∀ (bus_speed : ℝ), bus_speed > 0 → 
    distance / bus_speed - distance / (speed_ratio * bus_speed) = delay) :
  ∃ (bus_speed car_speed : ℝ),
    bus_speed = 50 ∧ 
    car_speed = 75 ∧
    car_speed = speed_ratio * bus_speed :=
by sorry

end bus_car_speed_problem_l926_92628


namespace trigonometric_fraction_equals_one_l926_92666

theorem trigonometric_fraction_equals_one : 
  (Real.sin (22 * π / 180) * Real.cos (8 * π / 180) + 
   Real.cos (158 * π / 180) * Real.cos (98 * π / 180)) / 
  (Real.sin (23 * π / 180) * Real.cos (7 * π / 180) + 
   Real.cos (157 * π / 180) * Real.cos (97 * π / 180)) = 1 := by
sorry

end trigonometric_fraction_equals_one_l926_92666


namespace parallel_lines_m_l926_92632

/-- Two lines are parallel if their slopes are equal -/
def parallel (a b c d e f : ℝ) : Prop :=
  a * e = b * d ∧ a * f ≠ b * c

/-- The problem statement -/
theorem parallel_lines_m (m : ℝ) :
  parallel m 1 (-1) 9 m (-(2 * m + 3)) → m = 3 := by
  sorry

end parallel_lines_m_l926_92632


namespace abs_a_minus_three_l926_92667

theorem abs_a_minus_three (a : ℝ) (h : a < 3) : |a - 3| = 3 - a := by
  sorry

end abs_a_minus_three_l926_92667


namespace prove_c_value_l926_92660

theorem prove_c_value (c : ℕ) : 
  (5 ^ 5) * (9 ^ 3) = c * (15 ^ 5) → c = 3 → c = 3 := by
  sorry

end prove_c_value_l926_92660


namespace no_solution_condition_l926_92621

theorem no_solution_condition (k : ℝ) : 
  (∀ x : ℝ, x ≠ 3 ∧ x ≠ 4 → (x - 1) / (x - 3) ≠ (x - k) / (x - 4)) ↔ k = 2 :=
by sorry

end no_solution_condition_l926_92621


namespace one_distinct_real_root_l926_92696

theorem one_distinct_real_root :
  ∃! x : ℝ, x ≠ 0 ∧ (abs x - 4 / x = 3 * abs x / x) :=
by sorry

end one_distinct_real_root_l926_92696


namespace mindy_income_multiple_l926_92656

/-- Proves that Mindy earned 3 times more than Mork given their tax rates and combined tax rate -/
theorem mindy_income_multiple (mork_rate mindy_rate combined_rate : ℚ) : 
  mork_rate = 40/100 →
  mindy_rate = 30/100 →
  combined_rate = 325/1000 →
  ∃ k : ℚ, k = 3 ∧ 
    (mork_rate + k * mindy_rate) / (1 + k) = combined_rate :=
by sorry

end mindy_income_multiple_l926_92656


namespace sixth_term_equals_five_l926_92690

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The 6th term of the sequence equals 5 given the conditions -/
theorem sixth_term_equals_five (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_sum : a 2 + a 6 + a 10 = 15) : 
  a 6 = 5 := by
sorry

end sixth_term_equals_five_l926_92690


namespace min_expression_leq_one_l926_92675

theorem min_expression_leq_one (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_eq_three : x + y + z = 3) : 
  min (x * (x + y - z)) (min (y * (y + z - x)) (z * (z + x - y))) ≤ 1 := by
  sorry

end min_expression_leq_one_l926_92675


namespace books_loaned_out_special_collection_loaned_books_l926_92688

/-- Proves that the number of books loaned out during the month is 20 --/
theorem books_loaned_out 
  (initial_books : ℕ) 
  (final_books : ℕ) 
  (return_rate : ℚ) : ℕ :=
  let loaned_books := (initial_books - final_books) / (1 - return_rate)
  20

/-- Given conditions --/
def initial_books : ℕ := 75
def final_books : ℕ := 68
def return_rate : ℚ := 65 / 100

/-- Main theorem --/
theorem special_collection_loaned_books : 
  books_loaned_out initial_books final_books return_rate = 20 := by
  sorry

end books_loaned_out_special_collection_loaned_books_l926_92688


namespace sum_congruent_to_6_mod_9_l926_92655

def sum : ℕ := 2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999

theorem sum_congruent_to_6_mod_9 : sum % 9 = 6 := by
  sorry

end sum_congruent_to_6_mod_9_l926_92655


namespace median_length_l926_92661

/-- A tetrahedron with vertex D at the origin and right angles at D -/
structure RightTetrahedron where
  A : ℝ × ℝ × ℝ
  B : ℝ × ℝ × ℝ
  C : ℝ × ℝ × ℝ
  right_angles : sorry
  DA_length : ‖A‖ = 1
  DB_length : ‖B‖ = 2
  DC_length : ‖C‖ = 3

/-- The median of a tetrahedron from vertex D -/
def tetrahedron_median (t : RightTetrahedron) : ℝ := sorry

/-- Theorem: The length of the median from D in the specified tetrahedron is √6/3 -/
theorem median_length (t : RightTetrahedron) : 
  tetrahedron_median t = Real.sqrt 6 / 3 := by sorry

end median_length_l926_92661


namespace lecture_hall_rows_l926_92613

/-- Represents the number of seats in a row of the lecture hall. -/
def seatsInRow (n : ℕ) : ℕ := 12 + 2 * (n - 1)

/-- Represents the total number of seats in the first n rows of the lecture hall. -/
def totalSeats (n : ℕ) : ℕ := n * (seatsInRow 1 + seatsInRow n) / 2

/-- States that the number of rows in the lecture hall is 16, given the conditions. -/
theorem lecture_hall_rows :
  ∃ (n : ℕ),
    n > 0 ∧
    totalSeats n > 400 ∧
    totalSeats n ≤ 440 ∧
    seatsInRow 1 = 12 ∧
    ∀ (i : ℕ), i > 0 → seatsInRow (i + 1) = seatsInRow i + 2 ∧
    n = 16 :=
  sorry

end lecture_hall_rows_l926_92613


namespace correct_calculation_l926_92630

theorem correct_calculation (x : ℝ) (h : 21 * x = 63) : x + 40 = 43 := by
  sorry

end correct_calculation_l926_92630


namespace election_ratio_l926_92631

theorem election_ratio (X Y : ℝ) 
  (h1 : 0.74 * X + 0.5000000000000002 * Y = 0.66 * (X + Y)) 
  (h2 : X > 0) 
  (h3 : Y > 0) : 
  X / Y = 2 := by
sorry

end election_ratio_l926_92631


namespace goldfish_count_l926_92627

theorem goldfish_count (daily_food_per_fish : ℝ) (special_food_percentage : ℝ) 
  (special_food_cost_per_ounce : ℝ) (total_special_food_cost : ℝ) 
  (h1 : daily_food_per_fish = 1.5)
  (h2 : special_food_percentage = 0.2)
  (h3 : special_food_cost_per_ounce = 3)
  (h4 : total_special_food_cost = 45) : 
  ∃ (total_fish : ℕ), total_fish = 50 := by
  sorry

end goldfish_count_l926_92627


namespace wage_percentage_proof_l926_92625

def company_finances (revenue : ℝ) (num_employees : ℕ) (tax_rate : ℝ) 
  (marketing_rate : ℝ) (operational_rate : ℝ) (employee_wage : ℝ) : Prop :=
  let after_tax := revenue * (1 - tax_rate)
  let after_marketing := after_tax * (1 - marketing_rate)
  let after_operational := after_marketing * (1 - operational_rate)
  let total_wages := num_employees * employee_wage
  total_wages / after_operational = 0.15

theorem wage_percentage_proof :
  company_finances 400000 10 0.10 0.05 0.20 4104 := by
  sorry

end wage_percentage_proof_l926_92625


namespace monday_bonnets_count_l926_92665

/-- Represents the number of bonnets made on each day of the week --/
structure BonnetProduction where
  monday : ℕ
  tuesday_wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Calculates the total number of bonnets produced --/
def total_bonnets (bp : BonnetProduction) : ℕ :=
  bp.monday + bp.tuesday_wednesday + bp.thursday + bp.friday

theorem monday_bonnets_count :
  ∃ (bp : BonnetProduction),
    bp.tuesday_wednesday = 2 * bp.monday ∧
    bp.thursday = bp.monday + 5 ∧
    bp.friday = bp.thursday - 5 ∧
    total_bonnets bp = 11 * 5 ∧
    bp.monday = 10 := by
  sorry

end monday_bonnets_count_l926_92665


namespace ben_catch_count_l926_92617

/-- The number of fish caught by each family member (except Ben) --/
def family_catch : Fin 4 → ℕ
| 0 => 1  -- Judy
| 1 => 3  -- Billy
| 2 => 2  -- Jim
| 3 => 5  -- Susie

/-- The total number of filets they will have --/
def total_filets : ℕ := 24

/-- The number of fish thrown back --/
def thrown_back : ℕ := 3

/-- The number of filets per fish --/
def filets_per_fish : ℕ := 2

theorem ben_catch_count :
  ∃ (ben_catch : ℕ),
    ben_catch = total_filets / filets_per_fish + thrown_back - (family_catch 0 + family_catch 1 + family_catch 2 + family_catch 3) ∧
    ben_catch = 4 := by
  sorry

end ben_catch_count_l926_92617


namespace math_score_difference_l926_92629

def regression_equation (x : ℝ) : ℝ := 6 + 0.4 * x

theorem math_score_difference (x₁ x₂ : ℝ) (h : x₂ - x₁ = 50) :
  regression_equation x₂ - regression_equation x₁ = 20 := by
  sorry

end math_score_difference_l926_92629


namespace first_day_exceeding_150_fungi_l926_92601

def fungi_growth (n : ℕ) : ℕ := 4 * 2^n

theorem first_day_exceeding_150_fungi : 
  (∃ n : ℕ, fungi_growth n > 150) ∧ 
  (∀ m : ℕ, m < 6 → fungi_growth m ≤ 150) ∧
  (fungi_growth 6 > 150) :=
by sorry

end first_day_exceeding_150_fungi_l926_92601


namespace multiply_and_simplify_l926_92650

theorem multiply_and_simplify (x : ℝ) : (x^4 + 12*x^2 + 144) * (x^2 - 12) = x^6 - 1728 := by
  sorry

end multiply_and_simplify_l926_92650


namespace henry_book_count_l926_92646

/-- Calculates the number of books Henry has after donating and picking up new books -/
def final_book_count (initial_books : ℕ) (box_count : ℕ) (books_per_box : ℕ) 
  (room_books : ℕ) (coffee_table_books : ℕ) (kitchen_books : ℕ) (new_books : ℕ) : ℕ :=
  initial_books - (box_count * books_per_box + room_books + coffee_table_books + kitchen_books) + new_books

/-- Theorem stating that Henry ends up with 23 books -/
theorem henry_book_count : 
  final_book_count 99 3 15 21 4 18 12 = 23 := by
  sorry

end henry_book_count_l926_92646


namespace triangle_angle_c_60_degrees_l926_92634

theorem triangle_angle_c_60_degrees
  (A B C : Real)
  (triangle_sum : A + B + C = Real.pi)
  (tan_condition : Real.tan A + Real.tan B + Real.sqrt 3 = Real.sqrt 3 * Real.tan A * Real.tan B) :
  C = Real.pi / 3 :=
sorry

end triangle_angle_c_60_degrees_l926_92634


namespace negative_nine_plus_sixteen_y_squared_equals_seven_y_squared_l926_92642

theorem negative_nine_plus_sixteen_y_squared_equals_seven_y_squared (y : ℝ) : 
  -9 * y^2 + 16 * y^2 = 7 * y^2 := by
  sorry

end negative_nine_plus_sixteen_y_squared_equals_seven_y_squared_l926_92642


namespace rectangular_field_area_l926_92668

/-- Represents a rectangular field with a given length, breadth, and perimeter. -/
structure RectangularField where
  length : ℝ
  breadth : ℝ
  perimeter : ℝ

/-- The area of a rectangular field. -/
def area (field : RectangularField) : ℝ :=
  field.length * field.breadth

/-- Theorem: For a rectangular field where the breadth is 60% of the length
    and the perimeter is 800 m, the area of the field is 37,500 square meters. -/
theorem rectangular_field_area :
  ∀ (field : RectangularField),
    field.breadth = 0.6 * field.length →
    field.perimeter = 800 →
    area field = 37500 := by
  sorry

end rectangular_field_area_l926_92668
