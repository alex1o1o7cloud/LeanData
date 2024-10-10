import Mathlib

namespace Q_always_perfect_square_l2185_218569

theorem Q_always_perfect_square (x : ℤ) : ∃ (b : ℤ), x^4 + 4*x^3 + 8*x^2 + 6*x + 9 = b^2 := by
  sorry

end Q_always_perfect_square_l2185_218569


namespace min_value_expression_l2185_218508

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4 * y = 2) :
  (x + 40 * y + 4) / (3 * x * y) ≥ 18 :=
by sorry

end min_value_expression_l2185_218508


namespace sally_payment_l2185_218535

/-- Calculates the amount Sally needs to pay out of pocket for books -/
def sally_out_of_pocket (given_amount : ℕ) (book_cost : ℕ) (num_students : ℕ) : ℕ :=
  max 0 (book_cost * num_students - given_amount)

/-- Proves that Sally needs to pay $205 out of pocket -/
theorem sally_payment : sally_out_of_pocket 320 15 35 = 205 := by
  sorry

end sally_payment_l2185_218535


namespace point_quadrant_l2185_218515

/-- If a point A(a,b) is in the first quadrant, then the point B(a,-b) is in the fourth quadrant. -/
theorem point_quadrant (a b : ℝ) (h : a > 0 ∧ b > 0) : a > 0 ∧ -b < 0 := by
  sorry

end point_quadrant_l2185_218515


namespace k_range_l2185_218501

/-- The function y = |log₂ x| is meaningful and not monotonic in the interval (k-1, k+1) -/
def is_meaningful_and_not_monotonic (k : ℝ) : Prop :=
  (k - 1 > 0) ∧ (1 ∈ Set.Ioo (k - 1) (k + 1))

/-- The theorem stating the range of k -/
theorem k_range :
  ∀ k : ℝ, is_meaningful_and_not_monotonic k ↔ k ∈ Set.Ioo 1 2 := by
  sorry

end k_range_l2185_218501


namespace rectangular_parallelepiped_volume_l2185_218556

theorem rectangular_parallelepiped_volume
  (m n p d : ℝ) 
  (h_positive : m > 0 ∧ n > 0 ∧ p > 0 ∧ d > 0) :
  ∃ (V : ℝ), V = (m * n * p * d^3) / (m^2 + n^2 + p^2)^(3/2) ∧
  ∃ (a b c : ℝ), 
    a / m = b / n ∧ 
    b / n = c / p ∧
    V = a * b * c ∧
    d^2 = a^2 + b^2 + c^2 := by
  sorry

#check rectangular_parallelepiped_volume

end rectangular_parallelepiped_volume_l2185_218556


namespace no_simultaneous_properties_l2185_218549

theorem no_simultaneous_properties : ¬∃ (star : ℤ → ℤ → ℤ),
  (∀ Z : ℤ, ∃ X Y : ℤ, star X Y = Z) ∧
  (∀ A B : ℤ, star A B = -(star B A)) ∧
  (∀ A B C : ℤ, star (star A B) C = star A (star B C)) :=
by sorry

end no_simultaneous_properties_l2185_218549


namespace boys_camp_total_l2185_218541

theorem boys_camp_total (total : ℕ) 
  (h1 : (total : ℚ) * (1 / 5) = (total : ℚ) * (20 / 100))
  (h2 : (total : ℚ) * (1 / 5) * (3 / 10) = (total : ℚ) * (1 / 5) * (30 / 100))
  (h3 : (total : ℚ) * (1 / 5) * (7 / 10) = 56) :
  total = 400 := by
sorry

end boys_camp_total_l2185_218541


namespace monitor_height_is_seven_l2185_218511

/-- Represents a rectangular monitor -/
structure RectangularMonitor where
  width : ℝ
  height : ℝ

/-- The circumference of a rectangular monitor -/
def circumference (m : RectangularMonitor) : ℝ :=
  2 * (m.width + m.height)

/-- Theorem: A rectangular monitor with width 12 cm and circumference 38 cm has a height of 7 cm -/
theorem monitor_height_is_seven :
  ∃ (m : RectangularMonitor), m.width = 12 ∧ circumference m = 38 → m.height = 7 :=
by sorry

end monitor_height_is_seven_l2185_218511


namespace truck_departure_time_l2185_218520

/-- Proves that given a car traveling at 55 mph and a truck traveling at 65 mph
    on the same road in the same direction, if it takes 6.5 hours for the truck
    to pass the car, then the truck left the station 1 hour after the car. -/
theorem truck_departure_time (car_speed truck_speed : ℝ) (passing_time : ℝ) :
  car_speed = 55 →
  truck_speed = 65 →
  passing_time = 6.5 →
  (truck_speed - car_speed) * passing_time / truck_speed = 1 :=
by sorry

end truck_departure_time_l2185_218520


namespace maaza_liters_l2185_218561

/-- The number of liters of Pepsi -/
def pepsi : ℕ := 144

/-- The number of liters of Sprite -/
def sprite : ℕ := 368

/-- The total number of cans required -/
def total_cans : ℕ := 261

/-- The capacity of each can in liters -/
def can_capacity : ℕ := Nat.gcd pepsi sprite

theorem maaza_liters : ∃ M : ℕ, 
  M + pepsi + sprite = total_cans * can_capacity ∧ 
  M = 3664 := by
  sorry

end maaza_liters_l2185_218561


namespace perfect_square_from_divisibility_l2185_218593

theorem perfect_square_from_divisibility (n p : ℕ) : 
  n > 1 → 
  Nat.Prime p → 
  (p - 1) % n = 0 → 
  (n^3 - 1) % p = 0 → 
  ∃ (k : ℕ), 4*p - 3 = k^2 :=
by
  sorry

end perfect_square_from_divisibility_l2185_218593


namespace rectangular_solid_on_sphere_l2185_218509

theorem rectangular_solid_on_sphere (a b c : ℝ) (h1 : a = 1) (h2 : b = 2) (h3 : c = 3) :
  let diagonal := Real.sqrt (a^2 + b^2 + c^2)
  let radius := diagonal / 2
  let surface_area := 4 * Real.pi * radius^2
  surface_area = 14 * Real.pi := by
  sorry

end rectangular_solid_on_sphere_l2185_218509


namespace john_total_spend_l2185_218524

-- Define the prices and quantities
def tshirt_price : ℝ := 20
def tshirt_quantity : ℕ := 3
def pants_price : ℝ := 50
def pants_quantity : ℕ := 2
def jacket_original_price : ℝ := 80
def jacket_discount : ℝ := 0.25
def hat_price : ℝ := 15
def shoes_original_price : ℝ := 60
def shoes_discount : ℝ := 0.10

-- Define the total cost function
def total_cost : ℝ :=
  (tshirt_price * tshirt_quantity) +
  (pants_price * pants_quantity) +
  (jacket_original_price * (1 - jacket_discount)) +
  hat_price +
  (shoes_original_price * (1 - shoes_discount))

-- Theorem to prove
theorem john_total_spend : total_cost = 289 := by
  sorry

end john_total_spend_l2185_218524


namespace infinite_divisibility_l2185_218565

theorem infinite_divisibility (a : ℕ) (h : a > 3) :
  ∃ (f : ℕ → ℕ), Monotone f ∧ (∀ i, (a + f i) ∣ (a^(f i) + 1)) :=
sorry

end infinite_divisibility_l2185_218565


namespace profit_percentage_calculation_l2185_218533

theorem profit_percentage_calculation (selling_price profit : ℝ) :
  selling_price = 850 →
  profit = 215 →
  let cost_price := selling_price - profit
  let profit_percentage := (profit / cost_price) * 100
  ∃ ε > 0, abs (profit_percentage - 33.86) < ε :=
by sorry

end profit_percentage_calculation_l2185_218533


namespace binary_digit_difference_l2185_218570

-- Define a function to calculate the number of digits in the binary representation of a number
def binaryDigits (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log2 n + 1

-- State the theorem
theorem binary_digit_difference : binaryDigits 950 - binaryDigits 150 = 2 := by
  sorry

end binary_digit_difference_l2185_218570


namespace equation_solution_l2185_218588

theorem equation_solution : ∃! x : ℚ, (10 - 2*x)^2 = 4*x^2 := by
  sorry

end equation_solution_l2185_218588


namespace tree_count_proof_l2185_218554

theorem tree_count_proof (total : ℕ) (pine_fraction : ℚ) (fir_percent : ℚ) 
  (h1 : total = 520)
  (h2 : pine_fraction = 1 / 3)
  (h3 : fir_percent = 25 / 100) :
  ⌊total * pine_fraction⌋ + ⌊total * fir_percent⌋ = 390 := by
  sorry

end tree_count_proof_l2185_218554


namespace smallest_alpha_inequality_l2185_218552

theorem smallest_alpha_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  ∃ (α : ℝ), α > 0 ∧ α = 1/2 ∧
  ∀ (β : ℝ), β > 0 →
    ((x + y) / 2 ≥ β * Real.sqrt (x * y) + (1 - β) * Real.sqrt ((x^2 + y^2) / 2) →
     β ≥ α) :=
by sorry

end smallest_alpha_inequality_l2185_218552


namespace min_white_fraction_is_one_eighth_l2185_218526

/-- Represents a cube constructed from smaller cubes -/
structure LargeCube where
  edge_length : ℕ
  small_cubes : ℕ
  red_cubes : ℕ
  white_cubes : ℕ

/-- Calculates the surface area of a cube -/
def surface_area (c : LargeCube) : ℕ := 6 * c.edge_length * c.edge_length

/-- Calculates the minimum number of white cubes needed to have at least one on each face -/
def min_white_cubes_on_surface : ℕ := 4

/-- Calculates the white surface area when white cubes are placed optimally -/
def white_surface_area : ℕ := min_white_cubes_on_surface * 3

/-- The theorem to be proved -/
theorem min_white_fraction_is_one_eighth (c : LargeCube) 
    (h1 : c.edge_length = 4)
    (h2 : c.small_cubes = 64)
    (h3 : c.red_cubes = 56)
    (h4 : c.white_cubes = 8) :
  (white_surface_area : ℚ) / (surface_area c : ℚ) = 1/8 := by
  sorry

end min_white_fraction_is_one_eighth_l2185_218526


namespace second_number_value_l2185_218537

theorem second_number_value (x y z : ℚ) 
  (sum_eq : x + y + z = 120)
  (ratio_xy : x / y = 3 / 4)
  (ratio_yz : y / z = 4 / 7) :
  y = 240 / 7 := by
sorry

end second_number_value_l2185_218537


namespace joan_games_this_year_l2185_218503

/-- The number of football games Joan went to this year -/
def games_this_year : ℕ := sorry

/-- The number of football games Joan went to last year -/
def games_last_year : ℕ := 9

/-- The total number of football games Joan went to -/
def total_games : ℕ := 13

/-- Theorem stating that the number of games Joan went to this year is 4 -/
theorem joan_games_this_year : games_this_year = 4 := by sorry

end joan_games_this_year_l2185_218503


namespace complex_magnitude_quadratic_l2185_218502

theorem complex_magnitude_quadratic (z : ℂ) : z^2 - 6*z + 25 = 0 → Complex.abs z = 5 := by
  sorry

end complex_magnitude_quadratic_l2185_218502


namespace min_cost_is_80_yuan_l2185_218595

/-- Represents the swimming trip problem -/
structure SwimmingTripProblem where
  card_cost : ℕ            -- Cost of each swim card in yuan
  students : ℕ             -- Number of students
  swims_per_student : ℕ    -- Number of swims each student needs
  bus_cost : ℕ             -- Cost of bus rental per trip in yuan

/-- Calculates the minimum cost per student for the swimming trip -/
def min_cost_per_student (problem : SwimmingTripProblem) : ℚ :=
  let total_swims := problem.students * problem.swims_per_student
  let cards := 8  -- Optimal number of cards to buy
  let trips := total_swims / cards
  let total_cost := problem.card_cost * cards + problem.bus_cost * trips
  (total_cost : ℚ) / problem.students

/-- Theorem stating that the minimum cost per student is 80 yuan -/
theorem min_cost_is_80_yuan (problem : SwimmingTripProblem) 
    (h1 : problem.card_cost = 240)
    (h2 : problem.students = 48)
    (h3 : problem.swims_per_student = 8)
    (h4 : problem.bus_cost = 40) : 
  min_cost_per_student problem = 80 := by
  sorry

#eval min_cost_per_student { card_cost := 240, students := 48, swims_per_student := 8, bus_cost := 40 }

end min_cost_is_80_yuan_l2185_218595


namespace fairCoinThreeFlipsOneHead_l2185_218551

def fairCoinProbability (n : ℕ) (k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1/2)^k * (1/2)^(n-k)

theorem fairCoinThreeFlipsOneHead :
  fairCoinProbability 3 1 = 3/8 := by
  sorry

end fairCoinThreeFlipsOneHead_l2185_218551


namespace lunch_cost_l2185_218542

theorem lunch_cost (x : ℝ) : 
  x + 0.04 * x + 0.06 * x = 110 → x = 100 := by
  sorry

end lunch_cost_l2185_218542


namespace total_cost_is_598_l2185_218543

/-- The cost of 1 kg of flour in dollars -/
def flour_cost : ℝ := 23

/-- The cost relationship between mangos and rice -/
def mango_rice_relation (mango_cost rice_cost : ℝ) : Prop :=
  10 * mango_cost = rice_cost * 10

/-- The cost relationship between flour and rice -/
def flour_rice_relation (rice_cost : ℝ) : Prop :=
  6 * flour_cost = 2 * rice_cost

/-- The total cost of the given quantities of mangos, rice, and flour -/
def total_cost (mango_cost rice_cost : ℝ) : ℝ :=
  4 * mango_cost + 3 * rice_cost + 5 * flour_cost

/-- Theorem stating the total cost is $598 given the conditions -/
theorem total_cost_is_598 (mango_cost rice_cost : ℝ) 
  (h1 : mango_rice_relation mango_cost rice_cost)
  (h2 : flour_rice_relation rice_cost) : 
  total_cost mango_cost rice_cost = 598 := by
  sorry

end total_cost_is_598_l2185_218543


namespace right_square_prism_properties_l2185_218592

/-- Right square prism -/
structure RightSquarePrism where
  base_edge : ℝ
  height : ℝ

/-- Calculates the lateral area of a right square prism -/
def lateral_area (p : RightSquarePrism) : ℝ :=
  4 * p.base_edge * p.height

/-- Calculates the volume of a right square prism -/
def volume (p : RightSquarePrism) : ℝ :=
  p.base_edge ^ 2 * p.height

theorem right_square_prism_properties :
  ∃ (p : RightSquarePrism), p.base_edge = 3 ∧ p.height = 2 ∧
    lateral_area p = 24 ∧ volume p = 18 := by
  sorry

end right_square_prism_properties_l2185_218592


namespace arithmetic_sequence_common_difference_l2185_218532

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_first_term : a 1 = 1) 
  (h_sum : a 1 + a 2 + a 3 = 12) : 
  a 2 - a 1 = 3 := by
sorry

end arithmetic_sequence_common_difference_l2185_218532


namespace power_of_p_is_one_l2185_218566

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The property of being a positive even integer with a positive units digit -/
def isPositiveEvenWithPositiveUnitsDigit (p : ℕ) : Prop :=
  p > 0 ∧ p % 2 = 0 ∧ unitsDigit p > 0

theorem power_of_p_is_one (p : ℕ) (k : ℕ) 
  (h1 : isPositiveEvenWithPositiveUnitsDigit p)
  (h2 : unitsDigit (p + 1) = 7)
  (h3 : unitsDigit (p^3) - unitsDigit (p^k) = 0) :
  k = 1 := by sorry

end power_of_p_is_one_l2185_218566


namespace divisible_by_five_problem_l2185_218559

theorem divisible_by_five_problem (n : ℕ) : 
  n % 5 = 0 ∧ n / 5 = 96 → (n + 17) * 69 = 34293 := by
  sorry

end divisible_by_five_problem_l2185_218559


namespace fourth_pentagon_dots_l2185_218547

/-- Represents the number of dots in the nth pentagon of the sequence -/
def dots (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else if n = 2 then 6
  else if n = 3 then 16
  else dots (n - 1) + 5 * (n - 1)

/-- The theorem stating that the fourth pentagon has 31 dots -/
theorem fourth_pentagon_dots : dots 4 = 31 := by
  sorry


end fourth_pentagon_dots_l2185_218547


namespace all_faces_dirty_l2185_218594

/-- Represents the state of a wise man's face -/
inductive FaceState
| Clean
| Dirty

/-- Represents a wise man -/
structure WiseMan :=
  (id : Nat)
  (faceState : FaceState)

/-- Represents the knowledge of a wise man about the others' faces -/
def Knowledge := WiseMan → FaceState

/-- Represents whether a wise man is laughing -/
def isLaughing (w : WiseMan) (k : Knowledge) : Prop :=
  ∃ (other : WiseMan), k other = FaceState.Dirty

/-- The main theorem -/
theorem all_faces_dirty 
  (men : Finset WiseMan) 
  (h_three_men : men.card = 3) 
  (k : WiseMan → Knowledge) 
  (h_correct_knowledge : ∀ (w₁ w₂ : WiseMan), w₁ ≠ w₂ → k w₁ w₂ = w₂.faceState) 
  (h_all_laughing : ∀ (w : WiseMan), w ∈ men → isLaughing w (k w)) :
  ∀ (w : WiseMan), w ∈ men → w.faceState = FaceState.Dirty :=
sorry

end all_faces_dirty_l2185_218594


namespace geometric_sequence_terms_l2185_218538

theorem geometric_sequence_terms (a₁ aₙ q : ℚ) (n : ℕ) (h₁ : a₁ = 9/8) (h₂ : aₙ = 1/3) (h₃ : q = 2/3) :
  aₙ = a₁ * q^(n - 1) → n = 4 := by
  sorry

end geometric_sequence_terms_l2185_218538


namespace same_and_different_signs_l2185_218576

theorem same_and_different_signs (a b : ℝ) : 
  (a * b > 0 ↔ (a > 0 ∧ b > 0) ∨ (a < 0 ∧ b < 0)) ∧
  (a * b < 0 ↔ (a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0)) :=
by sorry

end same_and_different_signs_l2185_218576


namespace probability_three_heads_in_eight_tosses_l2185_218527

-- Define the number of coin tosses
def num_tosses : ℕ := 8

-- Define the number of heads we're looking for
def target_heads : ℕ := 3

-- Define a function to calculate the binomial coefficient
def binomial_coefficient (n k : ℕ) : ℕ := sorry

-- Define a function to calculate the probability of getting exactly k heads in n tosses
def probability_exactly_k_heads (n k : ℕ) : ℚ :=
  (binomial_coefficient n k : ℚ) / (2 ^ n : ℚ)

-- Theorem statement
theorem probability_three_heads_in_eight_tosses :
  probability_exactly_k_heads num_tosses target_heads = 7 / 32 := by sorry

end probability_three_heads_in_eight_tosses_l2185_218527


namespace quadratic_sum_of_constants_l2185_218514

-- Define the quadratic function
def f (x : ℝ) : ℝ := 4*x^2 - 16*x - 64

-- Define the completed square form
def g (x a b c : ℝ) : ℝ := a*(x+b)^2 + c

-- Theorem statement
theorem quadratic_sum_of_constants :
  ∃ (a b c : ℝ), (∀ x, f x = g x a b c) ∧ (a + b + c = -78) := by
  sorry

end quadratic_sum_of_constants_l2185_218514


namespace median_of_consecutive_integers_with_sum_property_l2185_218522

-- Define a set of consecutive integers
def ConsecutiveIntegers (a : ℤ) (n : ℕ) := {i : ℤ | ∃ k : ℕ, k < n ∧ i = a + k}

-- Define the property of sum of nth from beginning and end being 200
def SumProperty (s : Set ℤ) : Prop :=
  ∀ a n, s = ConsecutiveIntegers a n →
    ∀ k, k < n → (a + k) + (a + (n - 1 - k)) = 200

-- Theorem statement
theorem median_of_consecutive_integers_with_sum_property (s : Set ℤ) :
  SumProperty s → ∃ a n, s = ConsecutiveIntegers a n ∧ n % 2 = 1 ∧ 
  (∃ m : ℤ, m ∈ s ∧ (∀ x ∈ s, 2 * (x - m) ≤ n - 1 ∧ 2 * (m - x) ≤ n - 1) ∧ m = 100) :=
sorry

end median_of_consecutive_integers_with_sum_property_l2185_218522


namespace smallest_k_for_square_l2185_218585

theorem smallest_k_for_square : ∃ (m : ℕ), 
  2016 * 2017 * 2018 * 2019 + 1 = m^2 ∧ 
  ∀ (k : ℕ), k < 1 → ¬∃ (n : ℕ), 2016 * 2017 * 2018 * 2019 + k = n^2 :=
by sorry

end smallest_k_for_square_l2185_218585


namespace pirate_treasure_problem_l2185_218544

theorem pirate_treasure_problem :
  let n : ℕ := 8  -- Total number of islands
  let k : ℕ := 5  -- Number of islands with treasure
  let p_treasure : ℚ := 1/6  -- Probability of an island having treasure and no traps
  let p_neither : ℚ := 2/3  -- Probability of an island having neither treasure nor traps
  
  (Nat.choose n k : ℚ) * p_treasure^k * p_neither^(n - k) = 7/3328 := by
  sorry

end pirate_treasure_problem_l2185_218544


namespace triangle_area_at_least_three_l2185_218558

/-- A type representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The area of a triangle formed by three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

/-- A set of five points in a plane -/
def FivePoints : Type := Fin 5 → Point

theorem triangle_area_at_least_three (points : FivePoints) 
  (h : ∀ (i j k : Fin 5), i ≠ j → j ≠ k → i ≠ k → 
       triangleArea (points i) (points j) (points k) ≥ 2) :
  ∃ (i j k : Fin 5), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    triangleArea (points i) (points j) (points k) ≥ 3 := by
  sorry

end triangle_area_at_least_three_l2185_218558


namespace olivia_spent_25_dollars_l2185_218523

/-- The amount Olivia spent at the supermarket -/
def amount_spent (initial_amount remaining_amount : ℕ) : ℕ :=
  initial_amount - remaining_amount

/-- Theorem stating that Olivia spent 25 dollars -/
theorem olivia_spent_25_dollars (initial_amount remaining_amount : ℕ) 
  (h1 : initial_amount = 54)
  (h2 : remaining_amount = 29) : 
  amount_spent initial_amount remaining_amount = 25 := by
  sorry

end olivia_spent_25_dollars_l2185_218523


namespace point_on_line_angle_with_x_axis_line_equation_correct_l2185_218584

/-- The equation of a line passing through (2, 2) and making a 60° angle with the x-axis -/
def line_equation (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * x - 2 * Real.sqrt 3 + 2

/-- The point (2, 2) lies on the line -/
theorem point_on_line : line_equation 2 2 := by sorry

/-- The angle between the line and the x-axis is 60° -/
theorem angle_with_x_axis : 
  Real.arctan (Real.sqrt 3) = 60 * π / 180 := by sorry

/-- The line equation is correct -/
theorem line_equation_correct (x y : ℝ) :
  line_equation x y ↔ 
    (∃ k : ℝ, y - 2 = k * (x - 2) ∧ 
              k = Real.tan (60 * π / 180)) := by sorry

end point_on_line_angle_with_x_axis_line_equation_correct_l2185_218584


namespace cubic_root_form_l2185_218583

theorem cubic_root_form : ∃ (x : ℝ), 
  16 * x^3 - 4 * x^2 - 4 * x - 1 = 0 ∧ 
  x = (Real.rpow 256 (1/3 : ℝ) + Real.rpow 16 (1/3 : ℝ) + 1) / 16 := by
  sorry

end cubic_root_form_l2185_218583


namespace transform_f_eq_g_l2185_218553

/-- The original quadratic function -/
def f (x : ℝ) : ℝ := x^2 - 2

/-- The transformation: shift 1 unit left, then 3 units up -/
def transform (g : ℝ → ℝ) : ℝ → ℝ := λ x => g (x + 1) + 3

/-- The expected result function -/
def g (x : ℝ) : ℝ := (x + 1)^2 + 1

/-- Theorem stating that the transformation of f equals g -/
theorem transform_f_eq_g : transform f = g := by sorry

end transform_f_eq_g_l2185_218553


namespace dice_roll_sum_l2185_218563

theorem dice_roll_sum (a b c d : ℕ) : 
  1 ≤ a ∧ a ≤ 6 ∧
  1 ≤ b ∧ b ≤ 6 ∧
  1 ≤ c ∧ c ≤ 6 ∧
  1 ≤ d ∧ d ≤ 6 ∧
  a * b * c * d = 360 →
  a + b + c + d ≠ 20 := by
sorry

end dice_roll_sum_l2185_218563


namespace cricket_team_members_l2185_218555

/-- The number of members in a cricket team satisfying specific age conditions. -/
theorem cricket_team_members : ∃ (n : ℕ),
  n > 0 ∧
  let captain_age : ℕ := 26
  let keeper_age : ℕ := captain_age + 5
  let team_avg_age : ℚ := 24
  let remaining_avg_age : ℚ := team_avg_age - 1
  n * team_avg_age = (n - 2) * remaining_avg_age + (captain_age + keeper_age) ∧
  n = 11 := by
sorry

end cricket_team_members_l2185_218555


namespace average_weight_problem_l2185_218571

theorem average_weight_problem (A B C : ℝ) 
  (h1 : (A + B) / 2 = 40)
  (h2 : (B + C) / 2 = 43)
  (h3 : B = 31) :
  (A + B + C) / 3 = 45 := by
  sorry

end average_weight_problem_l2185_218571


namespace percentage_both_correct_l2185_218562

theorem percentage_both_correct (p_first : ℝ) (p_second : ℝ) (p_neither : ℝ) 
  (h1 : p_first = 0.75)
  (h2 : p_second = 0.70)
  (h3 : p_neither = 0.20) :
  p_first + p_second - (1 - p_neither) = 0.65 := by
  sorry

end percentage_both_correct_l2185_218562


namespace hyperbola_focal_length_l2185_218589

theorem hyperbola_focal_length (x y : ℝ) :
  x^2 / 7 - y^2 / 3 = 1 → ∃ (f : ℝ), f = 2 * Real.sqrt 10 :=
sorry

end hyperbola_focal_length_l2185_218589


namespace parabola_translation_theorem_l2185_218568

/-- Represents a parabola of the form y = ax² -/
structure Parabola where
  a : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a translation in 2D space -/
structure Translation where
  right : ℝ
  up : ℝ

/-- Returns true if the given equation represents the parabola after translation -/
def is_translated_parabola (p : Parabola) (t : Translation) (eq : ℝ → ℝ) : Prop :=
  ∀ x, eq x = p.a * (x - t.right)^2 + t.up

/-- Returns true if the given point satisfies the equation -/
def satisfies_equation (pt : Point) (eq : ℝ → ℝ) : Prop :=
  eq pt.x = pt.y

theorem parabola_translation_theorem (p : Parabola) (t : Translation) (pt : Point) :
  is_translated_parabola p t (fun x => -4 * (x - 2)^2 + 3) ∧
  satisfies_equation pt (fun x => -4 * (x - 2)^2 + 3) ∧
  t.right = 2 ∧ t.up = 3 ∧ pt.x = 3 ∧ pt.y = -1 :=
sorry

end parabola_translation_theorem_l2185_218568


namespace max_third_side_of_triangle_l2185_218590

/-- Given a triangle with two sides of 5 cm and 10 cm, 
    the maximum integer length of the third side is 14 cm. -/
theorem max_third_side_of_triangle (a b c : ℕ) : 
  a = 5 → b = 10 → c ≤ 14 → a + b > c → a + c > b → b + c > a → c ≤ a + b - 1 :=
by sorry

end max_third_side_of_triangle_l2185_218590


namespace final_single_stone_piles_l2185_218516

/-- Represents the state of the game -/
structure GameState where
  piles : List Nat
  deriving Repr

/-- Initial game state -/
def initialState : GameState :=
  { piles := List.range 10 |>.map (· + 1) }

/-- Combines two piles and adds 2 stones -/
def combinePiles (state : GameState) (i j : Nat) : GameState :=
  sorry

/-- Splits a pile into two after removing 2 stones -/
def splitPile (state : GameState) (i : Nat) (split : Nat) : GameState :=
  sorry

/-- Checks if the game has ended -/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- Counts the number of piles with one stone -/
def countSingleStonePiles (state : GameState) : Nat :=
  sorry

/-- The main theorem to prove -/
theorem final_single_stone_piles (finalState : GameState) :
  isGameOver finalState → countSingleStonePiles finalState = 23 := by
  sorry

end final_single_stone_piles_l2185_218516


namespace imaginary_part_of_complex_fraction_l2185_218557

theorem imaginary_part_of_complex_fraction (i : ℂ) :
  i * i = -1 →
  Complex.im ((1 + i) / (1 - i)) = 1 := by
sorry

end imaginary_part_of_complex_fraction_l2185_218557


namespace inverse_of_A_l2185_218510

def A : Matrix (Fin 2) (Fin 2) ℚ := !![4, 5; -2, 9]

theorem inverse_of_A :
  let A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![9/46, -5/46; 1/23, 2/23]
  A * A_inv = 1 ∧ A_inv * A = 1 :=
by sorry

end inverse_of_A_l2185_218510


namespace fruit_sales_problem_l2185_218548

/-- Fruit sales problem -/
theorem fruit_sales_problem 
  (cost_price : ℝ) 
  (base_price : ℝ) 
  (base_sales : ℝ) 
  (price_increment : ℝ) 
  (sales_decrement : ℝ) 
  (min_sales : ℝ) 
  (max_price : ℝ) :
  cost_price = 8 →
  base_price = 10 →
  base_sales = 300 →
  price_increment = 1 →
  sales_decrement = 50 →
  min_sales = 250 →
  max_price = 13 →
  ∃ (sales_function : ℝ → ℝ) (max_profit : ℝ) (donation_range : Set ℝ),
    -- 1. Sales function
    (∀ x, sales_function x = -50 * x + 800) ∧
    -- 2. Maximum profit
    max_profit = 750 ∧
    -- 3. Donation range
    donation_range = {a : ℝ | 2 ≤ a ∧ a ≤ 2.5} :=
by
  sorry

end fruit_sales_problem_l2185_218548


namespace sqrt_meaningful_range_l2185_218545

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x + 1) ↔ x ≥ -1 :=
sorry

end sqrt_meaningful_range_l2185_218545


namespace parallel_vectors_tan_l2185_218587

theorem parallel_vectors_tan (x : ℝ) : 
  let a : ℝ × ℝ := (Real.sin x, Real.cos x)
  let b : ℝ × ℝ := (2, -3)
  (a.1 * b.2 = a.2 * b.1) → Real.tan x = -2/3 := by
sorry

end parallel_vectors_tan_l2185_218587


namespace balloon_distribution_l2185_218521

theorem balloon_distribution (red white green chartreuse : ℕ) 
  (h1 : red = 24)
  (h2 : white = 38)
  (h3 : green = 68)
  (h4 : chartreuse = 75)
  (friends : ℕ)
  (h5 : friends = 10) :
  (red + white + green + chartreuse) % friends = 5 := by
  sorry

end balloon_distribution_l2185_218521


namespace expansion_equality_l2185_218506

theorem expansion_equality (x : ℝ) : (x + 6) * (x - 1) = x^2 + 5*x - 6 := by sorry

end expansion_equality_l2185_218506


namespace magazine_publication_theorem_l2185_218517

/-- Represents a magazine issue -/
structure Issue :=
  (year : ℕ)
  (month : ℕ)
  (exercisePosition : ℕ)
  (problemPosition : ℕ)

/-- The publication schedule of the magazine -/
def publicationSchedule : 
  (exercisesPerIssue : ℕ) → 
  (problemsPerIssue : ℕ) → 
  (issuesPerYear : ℕ) → 
  (startYear : ℕ) → 
  (lastExerciseNumber : ℕ) → 
  (lastProblemNumber : ℕ) → 
  (Prop) :=
  λ exercisesPerIssue problemsPerIssue issuesPerYear startYear lastExerciseNumber lastProblemNumber =>
    ∃ (exerciseIssue problemIssue : Issue),
      -- The exercise issue is in 1979, 3rd month, 2nd exercise
      exerciseIssue.year = 1979 ∧
      exerciseIssue.month = 3 ∧
      exerciseIssue.exercisePosition = 2 ∧
      -- The problem issue is in 1973, 5th month, 5th problem
      problemIssue.year = 1973 ∧
      problemIssue.month = 5 ∧
      problemIssue.problemPosition = 5 ∧
      -- The serial numbers match the respective years
      (lastExerciseNumber + (exerciseIssue.year - startYear) * exercisesPerIssue * issuesPerYear + 
       (exerciseIssue.month - 1) * exercisesPerIssue + exerciseIssue.exercisePosition = exerciseIssue.year) ∧
      (lastProblemNumber + (problemIssue.year - startYear) * problemsPerIssue * issuesPerYear + 
       (problemIssue.month - 1) * problemsPerIssue + problemIssue.problemPosition = problemIssue.year)

theorem magazine_publication_theorem :
  publicationSchedule 8 8 9 1967 1169 1576 :=
by
  sorry


end magazine_publication_theorem_l2185_218517


namespace sector_to_inscribed_circle_area_ratio_l2185_218573

/-- Given a sector with a central angle of 120° and its inscribed circle,
    the ratio of the area of the sector to the area of the inscribed circle
    is (7 + 4√3) / 9. -/
theorem sector_to_inscribed_circle_area_ratio :
  ∀ (R r : ℝ), R > 0 → r > 0 →
  (2 * π / 3 : ℝ) = 2 * Real.arcsin (r / R) →
  (π * R^2 * (2 * π / 3) / (2 * π)) / (π * r^2) = (7 + 4 * Real.sqrt 3) / 9 := by
  sorry

end sector_to_inscribed_circle_area_ratio_l2185_218573


namespace max_points_at_least_sqrt2_max_points_greater_sqrt2_l2185_218586

-- Define a point on a unit sphere
def PointOnUnitSphere := ℝ × ℝ × ℝ

-- Distance function between two points on a unit sphere
def sphereDistance (p q : PointOnUnitSphere) : ℝ := sorry

-- Theorem for part a
theorem max_points_at_least_sqrt2 :
  ∀ (n : ℕ) (points : Fin n → PointOnUnitSphere),
    (∀ (i j : Fin n), i ≠ j → sphereDistance (points i) (points j) ≥ Real.sqrt 2) →
    n ≤ 6 :=
sorry

-- Theorem for part b
theorem max_points_greater_sqrt2 :
  ∀ (n : ℕ) (points : Fin n → PointOnUnitSphere),
    (∀ (i j : Fin n), i ≠ j → sphereDistance (points i) (points j) > Real.sqrt 2) →
    n ≤ 4 :=
sorry

end max_points_at_least_sqrt2_max_points_greater_sqrt2_l2185_218586


namespace price_increase_proof_l2185_218579

theorem price_increase_proof (x : ℝ) : 
  (1 + x)^2 = 1.44 → x = 0.2 := by sorry

end price_increase_proof_l2185_218579


namespace delivery_cost_fraction_l2185_218528

/-- Proves that the fraction of the remaining amount spent on delivery costs is 1/4 -/
theorem delivery_cost_fraction (total_cost : ℝ) (salary_fraction : ℝ) (order_cost : ℝ)
  (h1 : total_cost = 4000)
  (h2 : salary_fraction = 2/5)
  (h3 : order_cost = 1800) :
  let salary_cost := salary_fraction * total_cost
  let remaining_after_salary := total_cost - salary_cost
  let delivery_cost := remaining_after_salary - order_cost
  delivery_cost / remaining_after_salary = 1/4 := by
sorry

end delivery_cost_fraction_l2185_218528


namespace equation_solutions_l2185_218582

def equation (x : ℝ) : Prop :=
  1 / (x^2 + 10*x - 12) + 1 / (x^2 + 3*x - 12) + 1 / (x^2 - 14*x - 12) = 0

theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ x = 1 ∨ x = -21 ∨ x = 5 + Real.sqrt 37 ∨ x = 5 - Real.sqrt 37 :=
by sorry

end equation_solutions_l2185_218582


namespace division_remainder_problem_l2185_218550

theorem division_remainder_problem (smaller : ℕ) : 
  1614 - smaller = 1360 →
  1614 / smaller = 6 →
  1614 % smaller = 90 := by
sorry

end division_remainder_problem_l2185_218550


namespace equality_of_polynomials_l2185_218591

theorem equality_of_polynomials (a b : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x + 5 = (x - 2)^2 + a*(x - 2) + b) → 
  a + b = 4 := by
sorry

end equality_of_polynomials_l2185_218591


namespace tripled_base_doubled_exponent_l2185_218560

theorem tripled_base_doubled_exponent 
  (c y : ℝ) (d : ℝ) (h_d : d ≠ 0) :
  let s := (3 * c) ^ (2 * d)
  s = c^d / y^d →
  y = 1 / (9 * c) := by
sorry

end tripled_base_doubled_exponent_l2185_218560


namespace no_solution_to_inequalities_l2185_218540

theorem no_solution_to_inequalities : ¬∃ x : ℝ, (x / 2 ≥ 1 + x) ∧ (3 + 2*x > -3 - 3*x) := by
  sorry

end no_solution_to_inequalities_l2185_218540


namespace class_mean_calculation_l2185_218525

theorem class_mean_calculation (total_students : ℕ) (first_group : ℕ) (second_group : ℕ)
  (first_mean : ℚ) (second_mean : ℚ) :
  total_students = first_group + second_group →
  first_group = 40 →
  second_group = 10 →
  first_mean = 68 / 100 →
  second_mean = 74 / 100 →
  (first_group * first_mean + second_group * second_mean) / total_students = 692 / 1000 := by
sorry

#eval (40 * (68 : ℚ) / 100 + 10 * (74 : ℚ) / 100) / 50

end class_mean_calculation_l2185_218525


namespace difference_of_squares_division_l2185_218519

theorem difference_of_squares_division : (245^2 - 205^2) / 40 = 450 := by
  sorry

end difference_of_squares_division_l2185_218519


namespace point_distance_to_line_l2185_218529

/-- The distance from a point (a, 2) to the line x - y + 3 = 0 is 1, where a > 0 -/
def distance_to_line (a : ℝ) : Prop :=
  a > 0 ∧ |a + 1| / Real.sqrt 2 = 1

/-- Theorem: If the distance from (a, 2) to the line x - y + 3 = 0 is 1, then a = √2 - 1 -/
theorem point_distance_to_line (a : ℝ) (h : distance_to_line a) : a = Real.sqrt 2 - 1 := by
  sorry

end point_distance_to_line_l2185_218529


namespace fraction_of_product_l2185_218507

theorem fraction_of_product (x : ℚ) : x * (1/2 * 2/5 * 5100) = 765.0000000000001 → x = 3/4 := by
  sorry

end fraction_of_product_l2185_218507


namespace square_cut_perimeter_sum_l2185_218598

theorem square_cut_perimeter_sum (s : Real) 
  (h1 : s > 0) 
  (h2 : ∃ (a b c d : Real), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ 
        a + b = 1 ∧ c + d = 1 ∧
        s = 2*(a+b) + 2*(c+d) + 2*(a+c) + 2*(b+d)) :
  s = 8 ∨ s = 10 := by
sorry

end square_cut_perimeter_sum_l2185_218598


namespace angle_B_is_60_degrees_side_c_and_area_l2185_218580

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def validTriangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.a < t.b ∧ t.b < t.c ∧
  Real.sqrt 3 * t.a = 2 * t.b * Real.sin t.A

-- Theorem 1: Prove that angle B is 60 degrees
theorem angle_B_is_60_degrees (t : Triangle) (h : validTriangle t) :
  t.B = Real.pi / 3 := by
  sorry

-- Theorem 2: Prove side c length and area when a = 2 and b = √7
theorem side_c_and_area (t : Triangle) (h : validTriangle t)
  (ha : t.a = 2) (hb : t.b = Real.sqrt 7) :
  t.c = 3 ∧ (1/2 * t.a * t.c * Real.sin t.B) = (3 * Real.sqrt 3) / 2 := by
  sorry

end angle_B_is_60_degrees_side_c_and_area_l2185_218580


namespace A_intersect_B_l2185_218564

def A : Set ℤ := {-2, 0, 2}

def f (x : ℤ) : ℤ := Int.natAbs x

def B : Set ℤ := f '' A

theorem A_intersect_B : A ∩ B = {0, 2} := by sorry

end A_intersect_B_l2185_218564


namespace bobbys_shoe_cost_bobbys_shoe_cost_is_968_l2185_218577

/-- Calculates the total cost of Bobby's handmade shoes -/
theorem bobbys_shoe_cost (mold_cost : ℝ) (labor_rate : ℝ) (work_hours : ℝ) 
  (labor_discount : ℝ) (materials_cost : ℝ) (tax_rate : ℝ) : ℝ :=
  let discounted_labor_cost := labor_rate * work_hours * labor_discount
  let total_before_tax := mold_cost + discounted_labor_cost + materials_cost
  let tax := total_before_tax * tax_rate
  let total_with_tax := total_before_tax + tax
  
  total_with_tax

theorem bobbys_shoe_cost_is_968 :
  bobbys_shoe_cost 250 75 8 0.8 150 0.1 = 968 := by
  sorry

end bobbys_shoe_cost_bobbys_shoe_cost_is_968_l2185_218577


namespace inscribed_pentagon_segments_l2185_218599

/-- Represents a convex pentagon with an inscribed circle -/
structure InscribedPentagon where
  -- Side lengths
  FG : ℝ
  GH : ℝ
  HI : ℝ
  IJ : ℝ
  JF : ℝ
  -- Segment lengths from vertices to tangent points
  x : ℝ
  y : ℝ
  z : ℝ
  -- Properties
  convex : Bool
  inscribed : Bool
  -- Relationships between segments and sides
  eq1 : x + y = GH
  eq2 : x + z = FG
  eq3 : y + z = JF

/-- Theorem: Given the specific side lengths, the segment lengths are determined -/
theorem inscribed_pentagon_segments
  (p : InscribedPentagon)
  (h1 : p.FG = 7)
  (h2 : p.GH = 8)
  (h3 : p.HI = 8)
  (h4 : p.IJ = 8)
  (h5 : p.JF = 9)
  (h6 : p.convex)
  (h7 : p.inscribed) :
  p.x = 3 ∧ p.y = 5 ∧ p.z = 4 := by
  sorry

#check inscribed_pentagon_segments

end inscribed_pentagon_segments_l2185_218599


namespace arithmetic_sequence_before_negative_seventeen_l2185_218574

/-- 
Given an arithmetic sequence with first term 88 and common difference -3,
prove that the number of terms that appear before -17 is 35.
-/
theorem arithmetic_sequence_before_negative_seventeen :
  let a : ℕ → ℤ := λ n => 88 - 3 * (n - 1)
  ∃ k : ℕ, a k = -17 ∧ k - 1 = 35 := by
  sorry

end arithmetic_sequence_before_negative_seventeen_l2185_218574


namespace greatest_multiple_of_30_l2185_218500

/-- A function that checks if a list of digits represents a valid arrangement
    according to the problem conditions -/
def is_valid_arrangement (digits : List Nat) : Prop :=
  digits.length = 6 ∧
  digits.toFinset = {1, 3, 4, 6, 8, 9} ∧
  (digits.foldl (fun acc d => acc * 10 + d) 0) % 30 = 0

/-- The claim that 986310 is the greatest possible number satisfying the conditions -/
theorem greatest_multiple_of_30 :
  ∀ (digits : List Nat),
    is_valid_arrangement digits →
    (digits.foldl (fun acc d => acc * 10 + d) 0) ≤ 986310 :=
by sorry

end greatest_multiple_of_30_l2185_218500


namespace smallest_sum_of_sequence_l2185_218505

theorem smallest_sum_of_sequence (A B C D : ℤ) : 
  A > 0 → B > 0 → C > 0 →  -- A, B, C are positive integers
  (∃ r : ℤ, C - B = B - A) →  -- A, B, C form an arithmetic sequence
  (∃ q : ℚ, C = B * q ∧ D = C * q) →  -- B, C, D form a geometric sequence
  C = (7 * B) / 4 →  -- C/B = 7/4
  (∀ A' B' C' D' : ℤ, 
    A' > 0 → B' > 0 → C' > 0 → 
    (∃ r : ℤ, C' - B' = B' - A') → 
    (∃ q : ℚ, C' = B' * q ∧ D' = C' * q) → 
    C' = (7 * B') / 4 → 
    A + B + C + D ≤ A' + B' + C' + D') →
  A + B + C + D = 97 := by
sorry

end smallest_sum_of_sequence_l2185_218505


namespace dentist_age_fraction_l2185_218518

/-- Given a dentist's current age A and a fraction F, proves that F = 1/10 when A = 32 and (1/6) * (A - 8) = F * (A + 8) -/
theorem dentist_age_fraction (A : ℕ) (F : ℚ) 
  (h1 : A = 32) 
  (h2 : (1/6 : ℚ) * ((A : ℚ) - 8) = F * ((A : ℚ) + 8)) : 
  F = 1/10 := by sorry

end dentist_age_fraction_l2185_218518


namespace remainder_101_pow_36_mod_100_l2185_218531

theorem remainder_101_pow_36_mod_100 : 101^36 % 100 = 1 := by sorry

end remainder_101_pow_36_mod_100_l2185_218531


namespace flour_mass_acceptance_l2185_218534

-- Define the labeled mass and uncertainty
def labeled_mass : ℝ := 35
def uncertainty : ℝ := 0.25

-- Define the acceptable range
def min_acceptable : ℝ := labeled_mass - uncertainty
def max_acceptable : ℝ := labeled_mass + uncertainty

-- Define the masses of the flour bags
def mass_A : ℝ := 34.70
def mass_B : ℝ := 34.80
def mass_C : ℝ := 35.30
def mass_D : ℝ := 35.51

-- Theorem to prove
theorem flour_mass_acceptance :
  (min_acceptable ≤ mass_B ∧ mass_B ≤ max_acceptable) ∧
  (mass_A < min_acceptable ∨ mass_A > max_acceptable) ∧
  (mass_C < min_acceptable ∨ mass_C > max_acceptable) ∧
  (mass_D < min_acceptable ∨ mass_D > max_acceptable) := by
  sorry

end flour_mass_acceptance_l2185_218534


namespace job_completion_times_l2185_218567

/-- Represents the productivity of a worker -/
structure Productivity where
  rate : ℝ
  rate_pos : rate > 0

/-- Represents a worker -/
structure Worker where
  productivity : Productivity

/-- Represents a job with three workers -/
structure Job where
  worker1 : Worker
  worker2 : Worker
  worker3 : Worker
  total_work : ℝ
  total_work_pos : total_work > 0
  third_worker_productivity : worker3.productivity.rate = (worker1.productivity.rate + worker2.productivity.rate) / 2
  work_condition : 48 * worker3.productivity.rate + 10 * worker1.productivity.rate = 
                   48 * worker3.productivity.rate + 15 * worker2.productivity.rate

/-- The theorem to be proved -/
theorem job_completion_times (job : Job) :
  let time1 := job.total_work / job.worker1.productivity.rate
  let time2 := job.total_work / job.worker2.productivity.rate
  let time3 := job.total_work / job.worker3.productivity.rate
  (time1 = 50 ∧ time2 = 75 ∧ time3 = 60) := by
  sorry

end job_completion_times_l2185_218567


namespace karen_homework_paragraphs_l2185_218597

/-- Represents the homework assignment structure -/
structure HomeworkAssignment where
  shortAnswerTime : ℕ
  paragraphTime : ℕ
  essayTime : ℕ
  essayCount : ℕ
  shortAnswerCount : ℕ
  totalTime : ℕ

/-- Calculates the number of paragraphs in the homework assignment -/
def calculateParagraphs (hw : HomeworkAssignment) : ℕ :=
  (hw.totalTime - hw.essayCount * hw.essayTime - hw.shortAnswerCount * hw.shortAnswerTime) / hw.paragraphTime

/-- Theorem stating that Karen's homework assignment results in 5 paragraphs -/
theorem karen_homework_paragraphs :
  let hw : HomeworkAssignment := {
    shortAnswerTime := 3,
    paragraphTime := 15,
    essayTime := 60,
    essayCount := 2,
    shortAnswerCount := 15,
    totalTime := 240
  }
  calculateParagraphs hw = 5 := by sorry

end karen_homework_paragraphs_l2185_218597


namespace books_distribution_l2185_218513

/-- Number of ways to distribute books among students -/
def distribute_books (n_books : ℕ) (n_students : ℕ) : ℕ :=
  -- Definition goes here
  sorry

/-- Theorem: Distributing 5 books among 3 students results in 90 different methods -/
theorem books_distribution :
  distribute_books 5 3 = 90 :=
by
  sorry

end books_distribution_l2185_218513


namespace inverse_f_75_l2185_218546

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^3 - 6

-- State the theorem
theorem inverse_f_75 : f⁻¹ 75 = 3 := by sorry

end inverse_f_75_l2185_218546


namespace smallest_value_expression_l2185_218536

theorem smallest_value_expression (a b : ℤ) (h1 : a = 3 * b) (h2 : b ≠ 0) :
  (((a + b) / (a - b)) ^ 2 + ((a - b) / (a + b)) ^ 2 : ℝ) = 4.25 := by sorry

end smallest_value_expression_l2185_218536


namespace weight_of_almonds_l2185_218539

/-- Given the total weight of nuts and the weight of pecans, 
    calculate the weight of almonds. -/
theorem weight_of_almonds 
  (total_weight : ℝ) 
  (pecan_weight : ℝ) 
  (h1 : total_weight = 0.52) 
  (h2 : pecan_weight = 0.38) : 
  total_weight - pecan_weight = 0.14 := by
sorry

end weight_of_almonds_l2185_218539


namespace ellipse_eccentricity_l2185_218512

theorem ellipse_eccentricity (k : ℝ) :
  (∃ x y : ℝ, x^2 / 9 + y^2 / (4 + k) = 1) →
  (∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c / a = 4/5 ∧
    ((a^2 = 9 ∧ b^2 = 4 + k) ∨ (a^2 = 4 + k ∧ b^2 = 9)) ∧
    c^2 = a^2 - b^2) →
  k = -19/25 ∨ k = 21 :=
by sorry

end ellipse_eccentricity_l2185_218512


namespace carls_cupcakes_l2185_218578

/-- Carl's cupcake selling problem -/
theorem carls_cupcakes (days : ℕ) (cupcakes_per_day : ℕ) (cupcakes_for_bonnie : ℕ) : 
  days = 2 → cupcakes_per_day = 60 → cupcakes_for_bonnie = 24 →
  days * cupcakes_per_day + cupcakes_for_bonnie = 144 := by
  sorry

end carls_cupcakes_l2185_218578


namespace ancient_chinese_algorithm_is_successive_subtraction_l2185_218504

/-- An ancient Chinese mathematical algorithm developed during the Song and Yuan dynasties -/
structure AncientChineseAlgorithm where
  name : String
  period : String
  comparable_to_euclidean : Bool

/-- The method of successive subtraction -/
def successive_subtraction : AncientChineseAlgorithm :=
  { name := "Method of Successive Subtraction",
    period := "Song and Yuan dynasties",
    comparable_to_euclidean := true }

/-- Theorem stating that the ancient Chinese algorithm comparable to the Euclidean algorithm
    of division is the method of successive subtraction -/
theorem ancient_chinese_algorithm_is_successive_subtraction :
  ∃ (a : AncientChineseAlgorithm), 
    a.period = "Song and Yuan dynasties" ∧ 
    a.comparable_to_euclidean = true ∧ 
    a = successive_subtraction :=
by
  sorry

end ancient_chinese_algorithm_is_successive_subtraction_l2185_218504


namespace cloth_sale_calculation_l2185_218575

/-- Proves that the number of metres of cloth sold is 500 given the conditions -/
theorem cloth_sale_calculation (total_selling_price : ℕ) (loss_per_metre : ℕ) (cost_price_per_metre : ℕ)
  (h1 : total_selling_price = 18000)
  (h2 : loss_per_metre = 5)
  (h3 : cost_price_per_metre = 41) :
  total_selling_price / (cost_price_per_metre - loss_per_metre) = 500 := by
  sorry

end cloth_sale_calculation_l2185_218575


namespace x_cube_x_x_square_l2185_218572

theorem x_cube_x_x_square (x : ℝ) (h : -1 < x ∧ x < 0) : x^3 < x ∧ x < x^2 := by
  sorry

end x_cube_x_x_square_l2185_218572


namespace roof_ratio_l2185_218596

theorem roof_ratio (length width : ℝ) 
  (area_eq : length * width = 676)
  (diff_eq : length - width = 39) :
  length / width = 4 :=
by sorry

end roof_ratio_l2185_218596


namespace general_admission_price_general_admission_price_is_20_l2185_218530

/-- Calculates the price of a general admission ticket given the total number of tickets sold,
    total revenue, VIP ticket price, and the difference between general and VIP tickets sold. -/
theorem general_admission_price 
  (total_tickets : ℕ) 
  (total_revenue : ℝ) 
  (vip_price : ℝ) 
  (ticket_difference : ℕ) : ℝ :=
  let general_tickets := (total_tickets + ticket_difference) / 2
  let vip_tickets := total_tickets - general_tickets
  let general_price := (total_revenue - vip_price * vip_tickets) / general_tickets
  general_price

/-- The price of a general admission ticket is $20 given the specific conditions. -/
theorem general_admission_price_is_20 : 
  general_admission_price 320 7500 40 212 = 20 := by
  sorry

end general_admission_price_general_admission_price_is_20_l2185_218530


namespace square_root_of_nine_l2185_218581

theorem square_root_of_nine : Real.sqrt 9 = 3 := by
  sorry

end square_root_of_nine_l2185_218581
