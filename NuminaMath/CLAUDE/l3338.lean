import Mathlib

namespace find_first_number_l3338_333811

theorem find_first_number (x : ℝ) : 
  let set1 := [10, 70, 19]
  let set2 := [x, 40, 60]
  (List.sum set2 / 3 : ℝ) = (List.sum set1 / 3 : ℝ) + 7 → x = 20 := by
  sorry

end find_first_number_l3338_333811


namespace polynomial_transformation_l3338_333802

theorem polynomial_transformation (x : ℝ) (hx : x ≠ 0) :
  let z := x - 1 / x
  x^4 - 3*x^3 - 2*x^2 + 3*x + 1 = x^2 * (z^2 - 3*z) := by sorry

end polynomial_transformation_l3338_333802


namespace set_equality_l3338_333837

theorem set_equality : {p : ℝ × ℝ | p.1 + p.2 = 5 ∧ 2 * p.1 - p.2 = 1} = {(2, 3)} := by
  sorry

end set_equality_l3338_333837


namespace triathlete_average_speed_l3338_333822

-- Define the problem parameters
def total_distance : Real := 8
def running_distance : Real := 4
def swimming_distance : Real := 4
def running_speed : Real := 10
def swimming_speed : Real := 6

-- Define the theorem
theorem triathlete_average_speed :
  let running_time := running_distance / running_speed
  let swimming_time := swimming_distance / swimming_speed
  let total_time := running_time + swimming_time
  let average_speed_mph := total_distance / total_time
  let average_speed_mpm := average_speed_mph / 60
  average_speed_mpm = 0.125 := by sorry

end triathlete_average_speed_l3338_333822


namespace three_valid_pairs_l3338_333885

/-- The number of ordered pairs (a, b) satisfying the floor painting conditions -/
def num_valid_pairs : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ =>
    let a := p.1
    let b := p.2
    b > a ∧ (a - 4) * (b - 4) = a * b / 3
  ) (Finset.product (Finset.range 100) (Finset.range 100))).card

/-- Theorem stating that there are exactly 3 valid pairs -/
theorem three_valid_pairs : num_valid_pairs = 3 := by
  sorry

end three_valid_pairs_l3338_333885


namespace class_size_problem_l3338_333877

theorem class_size_problem (n : ℕ) 
  (h1 : 20 ≤ n ∧ n ≤ 30) 
  (h2 : ∃ x y : ℕ, x < n ∧ y < n ∧ x ≠ y ∧ 2 * x + 1 = n - x ∧ 3 * y + 1 = n - y) :
  n = 25 := by
sorry

end class_size_problem_l3338_333877


namespace celia_running_time_l3338_333820

/-- Given that Celia runs twice as fast as Lexie, and Lexie takes 20 minutes to run a mile,
    prove that Celia will take 300 minutes to run 30 miles. -/
theorem celia_running_time :
  ∀ (lexie_speed celia_speed : ℝ),
  celia_speed = 2 * lexie_speed →
  lexie_speed * 20 = 1 →
  celia_speed * 300 = 30 :=
by
  sorry

end celia_running_time_l3338_333820


namespace fifty_paise_coins_count_l3338_333823

/-- Represents the types of coins in the bag -/
inductive CoinType
  | OneRupee
  | FiftyPaise
  | TwentyFivePaise

/-- Represents the bag of coins -/
structure CoinBag where
  numCoins : CoinType → ℕ
  totalValue : ℚ
  equalCoins : ∀ (c1 c2 : CoinType), numCoins c1 = numCoins c2

def coinValue : CoinType → ℚ
  | CoinType.OneRupee => 1
  | CoinType.FiftyPaise => 1/2
  | CoinType.TwentyFivePaise => 1/4

theorem fifty_paise_coins_count (bag : CoinBag) 
  (h1 : bag.totalValue = 105)
  (h2 : bag.numCoins CoinType.OneRupee = 60) :
  bag.numCoins CoinType.FiftyPaise = 60 := by
  sorry

end fifty_paise_coins_count_l3338_333823


namespace sphere_to_great_circle_area_ratio_l3338_333854

/-- The ratio of the area of a sphere to the area of its great circle is 4 -/
theorem sphere_to_great_circle_area_ratio :
  ∀ (R : ℝ), R > 0 →
  (4 * π * R^2) / (π * R^2) = 4 :=
by sorry

end sphere_to_great_circle_area_ratio_l3338_333854


namespace geometric_sequence_second_term_l3338_333848

theorem geometric_sequence_second_term (a₁ a₃ : ℝ) (h₁ : a₁ = 180) (h₃ : a₃ = 75 / 32) :
  ∃ b : ℝ, b > 0 ∧ b^2 = 421.875 ∧ ∃ r : ℝ, a₁ * r = b ∧ b * r = a₃ :=
sorry

end geometric_sequence_second_term_l3338_333848


namespace rectangle_packing_l3338_333875

/-- Represents the maximum number of non-overlapping 2-by-3 rectangles
    that can be placed in an m-by-n rectangle -/
def max_rectangles (m n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the maximum number of 2-by-3 rectangles
    that can be placed in an m-by-n rectangle is at least ⌊mn/6⌋ -/
theorem rectangle_packing (m n : ℕ) (hm : m > 1) (hn : n > 1) :
  max_rectangles m n ≥ (m * n) / 6 :=
sorry

end rectangle_packing_l3338_333875


namespace regular_decagon_interior_angle_regular_decagon_interior_angle_is_144_l3338_333824

/-- The measure of each interior angle of a regular decagon is 144 degrees. -/
theorem regular_decagon_interior_angle : ℝ :=
  let n : ℕ := 10  -- number of sides in a decagon
  let sum_of_angles : ℝ := (n - 2) * 180  -- sum of interior angles formula
  let angle_measure : ℝ := sum_of_angles / n  -- measure of each angle (sum divided by number of sides)
  angle_measure

/-- Proof that the measure of each interior angle of a regular decagon is 144 degrees. -/
theorem regular_decagon_interior_angle_is_144 : 
  regular_decagon_interior_angle = 144 := by
  sorry


end regular_decagon_interior_angle_regular_decagon_interior_angle_is_144_l3338_333824


namespace simplify_fraction_l3338_333894

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by
  sorry

end simplify_fraction_l3338_333894


namespace apple_crate_weight_l3338_333866

/-- The weight of one original box of apples in kilograms. -/
def original_box_weight : ℝ := 35

/-- The number of crates in the original set. -/
def num_crates : ℕ := 7

/-- The amount of apples removed from each crate in kilograms. -/
def removed_weight : ℝ := 20

/-- The number of original crates that equal the weight of all crates after removal. -/
def equivalent_crates : ℕ := 3

theorem apple_crate_weight :
  num_crates * (original_box_weight - removed_weight) = equivalent_crates * original_box_weight :=
sorry

end apple_crate_weight_l3338_333866


namespace isosceles_triangle_triangle_area_l3338_333807

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Part 1
theorem isosceles_triangle (t : Triangle) (h : t.a * Real.sin t.A = t.b * Real.sin t.B) :
  t.a = t.b := by
  sorry

-- Part 2
theorem triangle_area (t : Triangle) 
  (h1 : t.a + t.b = t.a * t.b)
  (h2 : t.c = 2)
  (h3 : t.C = π / 3) :
  (1 / 2) * t.a * t.b * Real.sin t.C = Real.sqrt 3 := by
  sorry

end isosceles_triangle_triangle_area_l3338_333807


namespace solution_set_equivalence_l3338_333843

theorem solution_set_equivalence (x : ℝ) :
  (Real.log (|x - π/3|) / Real.log (1/2) ≥ Real.log (π/2) / Real.log (1/2)) ↔
  (-π/6 ≤ x ∧ x ≤ 5*π/6 ∧ x ≠ π/3) :=
sorry

end solution_set_equivalence_l3338_333843


namespace infinite_special_numbers_l3338_333852

theorem infinite_special_numbers (k : ℕ) :
  let n := 250 * 3^(6*k)
  ∃ (a b c d : ℕ), 
    n = a^2 + b^2 ∧ 
    n = c^3 + d^3 ∧ 
    ¬∃ (x y : ℕ), n = x^6 + y^6 := by
  sorry

end infinite_special_numbers_l3338_333852


namespace kylies_towels_l3338_333829

theorem kylies_towels (daughters_towels husband_towels machine_capacity loads : ℕ) 
  (h1 : daughters_towels = 6)
  (h2 : husband_towels = 3)
  (h3 : machine_capacity = 4)
  (h4 : loads = 3) : 
  ∃ k : ℕ, k = loads * machine_capacity - daughters_towels - husband_towels ∧ k = 3 := by
  sorry

end kylies_towels_l3338_333829


namespace unique_solution_to_equation_l3338_333806

theorem unique_solution_to_equation (x : ℝ) : (x^2 + 4*x - 5)^0 = x^2 - 5*x + 5 ↔ x = 1 := by
  sorry

end unique_solution_to_equation_l3338_333806


namespace complex_fraction_sum_l3338_333845

theorem complex_fraction_sum (a b : ℝ) : 
  (1 + 2*I) / (1 + I) = a + b*I → a + b = 2 := by
  sorry

end complex_fraction_sum_l3338_333845


namespace city_distance_min_city_distance_l3338_333857

def is_valid_distance (S : ℕ) : Prop :=
  (∀ x : ℕ, x ≤ S → (Nat.gcd x (S - x) = 1 ∨ Nat.gcd x (S - x) = 3 ∨ Nat.gcd x (S - x) = 13)) ∧
  (∃ x : ℕ, x ≤ S ∧ Nat.gcd x (S - x) = 1) ∧
  (∃ x : ℕ, x ≤ S ∧ Nat.gcd x (S - x) = 3) ∧
  (∃ x : ℕ, x ≤ S ∧ Nat.gcd x (S - x) = 13)

theorem city_distance : 
  ∀ S : ℕ, is_valid_distance S → S ≥ 39 :=
by sorry

theorem min_city_distance :
  is_valid_distance 39 :=
by sorry

end city_distance_min_city_distance_l3338_333857


namespace triangle_inequality_l3338_333828

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (a^2 / (b + c - a)) + (b^2 / (c + a - b)) + (c^2 / (a + b - c)) ≥ a + b + c := by
  sorry

end triangle_inequality_l3338_333828


namespace sum_reciprocals_lower_bound_l3338_333890

theorem sum_reciprocals_lower_bound 
  (a b c d m : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (hm : m > 0)
  (eq1 : 1/a = (a + b + c + d + m)/a)
  (eq2 : 1/b = (a + b + c + d + m)/b)
  (eq3 : 1/c = (a + b + c + d + m)/c)
  (eq4 : 1/d = (a + b + c + d + m)/d)
  (eq5 : 1/m = (a + b + c + d + m)/m) :
  1/a + 1/b + 1/c + 1/d + 1/m ≥ 25 := by
sorry

end sum_reciprocals_lower_bound_l3338_333890


namespace ab_plus_cd_value_l3338_333815

theorem ab_plus_cd_value (a b c d : ℝ) 
  (eq1 : a + b + c = 5)
  (eq2 : a + b + d = 8)
  (eq3 : a + c + d = 20)
  (eq4 : b + c + d = 15) :
  a * b + c * d = 84 := by
sorry

end ab_plus_cd_value_l3338_333815


namespace no_solution_iff_m_eq_seven_l3338_333899

theorem no_solution_iff_m_eq_seven (m : ℝ) : 
  (∀ x : ℝ, x ≠ 4 ∧ x ≠ 8 → (x - 3) / (x - 4) ≠ (x - m) / (x - 8)) ↔ m = 7 :=
by sorry

end no_solution_iff_m_eq_seven_l3338_333899


namespace remainder_problem_l3338_333804

theorem remainder_problem (N : ℕ) : 
  (N / 5 = 5) ∧ (N % 5 = 0) → N % 11 = 3 := by
sorry

end remainder_problem_l3338_333804


namespace program_requires_eight_sessions_l3338_333898

/-- Calculates the number of seating sessions required for a group -/
def sessionsRequired (groupSize : ℕ) (capacity : ℕ) : ℕ :=
  (groupSize + capacity - 1) / capacity

/-- Represents the seating program -/
structure SeatingProgram where
  totalParents : ℕ
  totalPupils : ℕ
  capacity : ℕ
  parentsMorning : ℕ
  parentsAfternoon : ℕ
  pupilsMorning : ℕ
  pupilsMidDay : ℕ
  pupilsEvening : ℕ

/-- Calculates the total number of seating sessions required -/
def totalSessions (program : SeatingProgram) : ℕ :=
  sessionsRequired program.parentsMorning program.capacity +
  sessionsRequired program.parentsAfternoon program.capacity +
  sessionsRequired program.pupilsMorning program.capacity +
  sessionsRequired program.pupilsMidDay program.capacity +
  sessionsRequired program.pupilsEvening program.capacity

/-- Theorem stating that the given program requires 8 seating sessions -/
theorem program_requires_eight_sessions (program : SeatingProgram)
  (h1 : program.totalParents = 61)
  (h2 : program.totalPupils = 177)
  (h3 : program.capacity = 44)
  (h4 : program.parentsMorning = 35)
  (h5 : program.parentsAfternoon = 26)
  (h6 : program.pupilsMorning = 65)
  (h7 : program.pupilsMidDay = 57)
  (h8 : program.pupilsEvening = 55)
  : totalSessions program = 8 := by
  sorry


end program_requires_eight_sessions_l3338_333898


namespace trains_meet_time_l3338_333889

/-- Two trains moving towards each other on a straight track meet at 10 a.m. -/
theorem trains_meet_time :
  -- Define the distance between stations P and Q
  let distance_PQ : ℝ := 110

  -- Define the speed of the first train
  let speed_train1 : ℝ := 20

  -- Define the speed of the second train
  let speed_train2 : ℝ := 25

  -- Define the time difference between the starts of the two trains (in hours)
  let time_diff : ℝ := 1

  -- Define the start time of the second train
  let start_time_train2 : ℝ := 8

  -- The time when the trains meet (in hours after midnight)
  let meet_time : ℝ := start_time_train2 + 
    (distance_PQ - speed_train1 * time_diff) / (speed_train1 + speed_train2)

  -- Prove that the meet time is 10 a.m.
  meet_time = 10 := by sorry

end trains_meet_time_l3338_333889


namespace benjamins_dinner_cost_l3338_333842

-- Define the prices of items
def burger_price : ℕ := 5
def fries_price : ℕ := 2
def salad_price : ℕ := 3 * fries_price

-- Define the quantities of items
def burger_quantity : ℕ := 1
def fries_quantity : ℕ := 2
def salad_quantity : ℕ := 1

-- Define the total cost function
def total_cost : ℕ := 
  burger_price * burger_quantity + 
  fries_price * fries_quantity + 
  salad_price * salad_quantity

-- Theorem statement
theorem benjamins_dinner_cost : total_cost = 15 := by
  sorry

end benjamins_dinner_cost_l3338_333842


namespace solution_set_inequality_l3338_333867

/-- Given that the solution set of ax² - bx + c < 0 is (-2, 3), 
    prove that the solution set of bx² + ax + c < 0 is (-3, 2) -/
theorem solution_set_inequality (a b c : ℝ) : 
  (∀ x, ax^2 - b*x + c < 0 ↔ -2 < x ∧ x < 3) →
  (∀ x, b*x^2 + a*x + c < 0 ↔ -3 < x ∧ x < 2) :=
sorry

end solution_set_inequality_l3338_333867


namespace product_evaluation_l3338_333860

theorem product_evaluation (n : ℕ) (h : n = 3) :
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) * (n + 3) = 720 := by
  sorry

end product_evaluation_l3338_333860


namespace cafe_meal_cost_l3338_333884

theorem cafe_meal_cost (s c k : ℝ) : 
  (2 * s + 5 * c + 2 * k = 6.50) → 
  (3 * s + 8 * c + 3 * k = 10.20) → 
  (s + c + k = 1.90) :=
by
  sorry

end cafe_meal_cost_l3338_333884


namespace eggs_solution_l3338_333855

def eggs_problem (dozen_count : ℕ) (price_per_egg : ℚ) (tax_rate : ℚ) (discount_rate : ℚ) : ℚ :=
  let total_eggs := dozen_count * 12
  let original_cost := total_eggs * price_per_egg
  let discounted_cost := original_cost * (1 - discount_rate)
  let tax_amount := discounted_cost * tax_rate
  discounted_cost + tax_amount

theorem eggs_solution :
  eggs_problem 3 (1/2) (5/100) (10/100) = 1701/100 := by
  sorry

end eggs_solution_l3338_333855


namespace range_of_k_equation_of_l_when_OB_twice_OA_l3338_333840

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 6)^2 + y^2 = 20

-- Define the line l
def line_l (k x y : ℝ) : Prop := y = k * x

-- Define the condition that line l intersects circle C at two distinct points
def intersects_at_two_points (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧ line_l k x₁ y₁ ∧ line_l k x₂ y₂

-- Define the condition OB = 2OA
def OB_twice_OA (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₂^2 + y₂^2 = 4 * (x₁^2 + y₁^2)

-- Theorem for the range of k
theorem range_of_k (k : ℝ) :
  intersects_at_two_points k ↔ -Real.sqrt 5 / 2 < k ∧ k < Real.sqrt 5 / 2 :=
sorry

-- Theorem for the equation of line l when OB = 2OA
theorem equation_of_l_when_OB_twice_OA (k : ℝ) :
  (intersects_at_two_points k ∧
   ∃ (x₁ y₁ x₂ y₂ : ℝ), circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧ line_l k x₁ y₁ ∧ line_l k x₂ y₂ ∧ OB_twice_OA x₁ y₁ x₂ y₂)
  → k = 1 ∨ k = -1 :=
sorry

end range_of_k_equation_of_l_when_OB_twice_OA_l3338_333840


namespace cos_two_thirds_pi_plus_two_alpha_l3338_333864

theorem cos_two_thirds_pi_plus_two_alpha (α : Real) 
  (h : Real.sin (π / 6 - α) = 1 / 3) : 
  Real.cos (2 * π / 3 + 2 * α) = -7 / 9 := by
  sorry

end cos_two_thirds_pi_plus_two_alpha_l3338_333864


namespace terrys_total_spending_l3338_333861

/-- Terry's spending over three days --/
def terrys_spending (monday_amount : ℝ) : ℝ :=
  let tuesday_amount := 2 * monday_amount
  let wednesday_amount := 2 * (monday_amount + tuesday_amount)
  monday_amount + tuesday_amount + wednesday_amount

/-- Theorem: Terry's total spending is $54 --/
theorem terrys_total_spending : terrys_spending 6 = 54 := by
  sorry

end terrys_total_spending_l3338_333861


namespace dvd_book_capacity_l3338_333814

/-- Represents the capacity of a DVD book -/
structure DVDBook where
  current : ℕ  -- Number of DVDs currently in the book
  remaining : ℕ  -- Number of additional DVDs that can be added

/-- Calculates the total capacity of a DVD book -/
def totalCapacity (book : DVDBook) : ℕ :=
  book.current + book.remaining

/-- Theorem: The total capacity of the given DVD book is 126 -/
theorem dvd_book_capacity : 
  ∀ (book : DVDBook), book.current = 81 → book.remaining = 45 → totalCapacity book = 126 :=
by
  sorry


end dvd_book_capacity_l3338_333814


namespace password_digit_l3338_333874

theorem password_digit (n : ℕ) : 
  n = 5678 * 6789 → 
  ∃ (a b c d e f g h i : ℕ),
    n = a * 10^8 + b * 10^7 + c * 10^6 + d * 10^5 + e * 10^4 + f * 10^3 + g * 10^2 + h * 10 + i ∧
    a = 3 ∧ b = 8 ∧ c = 5 ∧ d = 4 ∧ f = 9 ∧ g = 4 ∧ h = 2 ∧
    e = 7 :=
sorry

end password_digit_l3338_333874


namespace negation_of_universal_positive_square_plus_x_l3338_333803

theorem negation_of_universal_positive_square_plus_x (P : ℝ → Prop) : 
  (¬ ∀ x : ℝ, x > 0 → x^2 + x > 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 + x ≤ 0) :=
by sorry

end negation_of_universal_positive_square_plus_x_l3338_333803


namespace tan_alpha_value_l3338_333887

theorem tan_alpha_value (α : Real) 
  (h : (Real.sin α - 2 * Real.cos α) / (3 * Real.sin α + 5 * Real.cos α) = -5) :
  Real.tan α = -23/16 := by
  sorry

end tan_alpha_value_l3338_333887


namespace complement_P_union_Q_l3338_333841

def U : Set ℝ := Set.univ
def P : Set ℝ := {x | x > 1}
def Q : Set ℝ := {x | x * (x - 2) < 0}

theorem complement_P_union_Q : 
  (U \ (P ∪ Q)) = {x : ℝ | x ≤ 0} := by sorry

end complement_P_union_Q_l3338_333841


namespace milk_packet_content_l3338_333893

theorem milk_packet_content 
  (num_packets : ℕ) 
  (oz_to_ml : ℝ) 
  (total_oz : ℝ) 
  (h1 : num_packets = 150)
  (h2 : oz_to_ml = 30)
  (h3 : total_oz = 1250) :
  (total_oz * oz_to_ml) / num_packets = 250 := by
sorry

end milk_packet_content_l3338_333893


namespace ellipse_equation_l3338_333809

/-- Given an ellipse C with equation x²/a² + y²/b² = 1, where a > b > 0,
    focal length = 4, and passing through point P(√2, √3),
    prove that the equation of the ellipse is x²/8 + y²/4 = 1. -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (∃ c : ℝ, c = 2 ∧ a^2 - b^2 = c^2) →
  (2 / a^2 + 3 / b^2 = 1) →
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 8 + y^2 / 4 = 1) :=
by sorry

end ellipse_equation_l3338_333809


namespace min_value_geometric_sequence_l3338_333868

/-- Given a geometric sequence with first term b₁ = 2, 
    the minimum value of 3b₂ + 4b₃ is -9/8 -/
theorem min_value_geometric_sequence (b₁ b₂ b₃ : ℝ) :
  b₁ = 2 →
  (∃ r : ℝ, b₂ = b₁ * r ∧ b₃ = b₂ * r) →
  (∀ c₂ c₃ : ℝ, (∃ s : ℝ, c₂ = 2 * s ∧ c₃ = 2 * s^2) → 
    3 * b₂ + 4 * b₃ ≤ 3 * c₂ + 4 * c₃) →
  3 * b₂ + 4 * b₃ = -9/8 :=
by sorry

end min_value_geometric_sequence_l3338_333868


namespace truncated_pyramid_lateral_area_l3338_333880

/-- Represents a regular quadrangular pyramid -/
structure RegularQuadPyramid where
  base_side : ℝ
  height : ℝ

/-- Represents a truncated regular quadrangular pyramid -/
structure TruncatedRegularQuadPyramid where
  base_side : ℝ
  height : ℝ
  cut_height : ℝ

/-- Calculates the lateral surface area of a truncated regular quadrangular pyramid -/
def lateral_surface_area (t : TruncatedRegularQuadPyramid) : ℝ :=
  sorry

theorem truncated_pyramid_lateral_area :
  let p : RegularQuadPyramid := { base_side := 6, height := 4 }
  let t : TruncatedRegularQuadPyramid := { base_side := 6, height := 4, cut_height := 1 }
  lateral_surface_area t = 26.25 := by
  sorry

end truncated_pyramid_lateral_area_l3338_333880


namespace f_max_values_l3338_333882

noncomputable def f (x θ : Real) : Real :=
  Real.sin x ^ 2 + Real.sqrt 3 * Real.tan θ * Real.cos x + (Real.sqrt 3 / 8) * Real.tan θ - 3/2

theorem f_max_values (θ : Real) (h : θ ∈ Set.Icc 0 (Real.pi / 3)) :
  (∃ (x : Real), f x (Real.pi / 3) ≤ f x (Real.pi / 3) ∧ f x (Real.pi / 3) = 15/8) ∧
  (∃ (θ' : Real) (h' : θ' ∈ Set.Icc 0 (Real.pi / 3)), 
    (∃ (x : Real), ∀ (y : Real), f y θ' ≤ f x θ' ∧ f x θ' = -1/8) ∧ 
    θ' = Real.pi / 6) :=
by sorry

end f_max_values_l3338_333882


namespace incorrect_representation_of_roots_l3338_333833

theorem incorrect_representation_of_roots : ∃ x : ℝ, x^2 - 3*x = 0 ∧ ¬(x = x ∧ x = 2*x) :=
by sorry

end incorrect_representation_of_roots_l3338_333833


namespace grape_juice_mixture_proof_l3338_333862

/-- Proves that adding 10 gallons of grape juice to 40 gallons of a mixture 
    containing 10% grape juice results in a new mixture with 28.000000000000004% grape juice. -/
theorem grape_juice_mixture_proof : 
  let initial_mixture : ℝ := 40
  let initial_concentration : ℝ := 0.1
  let added_juice : ℝ := 10
  let final_concentration : ℝ := 0.28000000000000004
  (initial_mixture * initial_concentration + added_juice) / (initial_mixture + added_juice) = final_concentration := by
  sorry

end grape_juice_mixture_proof_l3338_333862


namespace solution_set_inequality_l3338_333801

theorem solution_set_inequality (x : ℝ) : 
  (x - 1) * (2 - x) ≥ 0 ↔ 1 ≤ x ∧ x ≤ 2 := by
  sorry

end solution_set_inequality_l3338_333801


namespace max_m_value_l3338_333844

/-- Given a > 0, proves that the maximum value of m is e^(1/2) when the tangents of 
    y = x²/2 + ax and y = 2a²ln(x) + m coincide at their intersection point. -/
theorem max_m_value (a : ℝ) (h_a : a > 0) : 
  let C₁ : ℝ → ℝ := λ x => x^2 / 2 + a * x
  let C₂ : ℝ → ℝ → ℝ := λ x m => 2 * a^2 * Real.log x + m
  let tangent_C₁ : ℝ → ℝ := λ x => x + a
  let tangent_C₂ : ℝ → ℝ := λ x => 2 * a^2 / x
  ∃ x₀ m, C₁ x₀ = C₂ x₀ m ∧ tangent_C₁ x₀ = tangent_C₂ x₀ ∧ 
    (∀ m', C₁ x₀ = C₂ x₀ m' ∧ tangent_C₁ x₀ = tangent_C₂ x₀ → m' ≤ m) ∧
    m = Real.exp (1/2 : ℝ) := by
  sorry

end max_m_value_l3338_333844


namespace vakha_always_wins_l3338_333891

/-- Represents a point on the circle -/
structure Point where
  index : Fin 99

/-- Represents a color (Red or Blue) -/
inductive Color
  | Red
  | Blue

/-- Represents the game state -/
structure GameState where
  coloredPoints : Fin 99 → Option Color

/-- Represents a player (Bjorn or Vakha) -/
inductive Player
  | Bjorn
  | Vakha

/-- Checks if three points form an equilateral triangle -/
def isEquilateralTriangle (p1 p2 p3 : Point) : Prop :=
  (p2.index - p1.index) % 33 = 0 ∧
  (p3.index - p2.index) % 33 = 0 ∧
  (p1.index - p3.index) % 33 = 0

/-- Checks if a monochromatic equilateral triangle exists in the game state -/
def existsMonochromaticTriangle (state : GameState) : Prop :=
  ∃ (p1 p2 p3 : Point) (c : Color),
    isEquilateralTriangle p1 p2 p3 ∧
    state.coloredPoints p1.index = some c ∧
    state.coloredPoints p2.index = some c ∧
    state.coloredPoints p3.index = some c

/-- Represents a valid move in the game -/
def validMove (state : GameState) (p : Point) (c : Color) : Prop :=
  state.coloredPoints p.index = none ∧
  (∃ (q : Point), state.coloredPoints q.index ≠ none ∧ (q.index + 1 = p.index ∨ q.index = p.index + 1))

/-- Represents a winning strategy for Vakha -/
def hasWinningStrategy (player : Player) : Prop :=
  ∀ (initialState : GameState),
    ∃ (finalState : GameState),
      (∀ (p : Point) (c : Color), validMove initialState p c → 
        ∃ (nextState : GameState), validMove nextState p c) ∧
      existsMonochromaticTriangle finalState

/-- The main theorem: Vakha always has a winning strategy -/
theorem vakha_always_wins : hasWinningStrategy Player.Vakha := by
  sorry

end vakha_always_wins_l3338_333891


namespace geometric_body_volume_l3338_333835

/-- The volume of a geometric body composed of two tetrahedra --/
theorem geometric_body_volume :
  let side_length : ℝ := 1
  let height : ℝ := Real.sqrt 3 / 2
  let tetrahedron_volume : ℝ := (1 / 3) * ((Real.sqrt 3 / 4) * side_length ^ 2) * height
  let total_volume : ℝ := 2 * tetrahedron_volume
  total_volume = 1 / 4 := by
  sorry

end geometric_body_volume_l3338_333835


namespace wage_increase_proof_l3338_333818

/-- The original daily wage of a worker -/
def original_wage : ℝ := 20

/-- The percentage increase in the worker's wage -/
def wage_increase_percent : ℝ := 40

/-- The new daily wage after the increase -/
def new_wage : ℝ := 28

/-- Theorem stating that the original wage increased by 40% equals the new wage -/
theorem wage_increase_proof : 
  original_wage * (1 + wage_increase_percent / 100) = new_wage := by
  sorry

end wage_increase_proof_l3338_333818


namespace max_loquat_wholesale_l3338_333817

-- Define the fruit types
inductive Fruit
| Loquat
| Cherries
| Apples

-- Define the wholesale and retail prices
def wholesale_price (f : Fruit) : ℝ :=
  match f with
  | Fruit.Loquat => 8
  | Fruit.Cherries => 36
  | Fruit.Apples => 12

def retail_price (f : Fruit) : ℝ :=
  match f with
  | Fruit.Loquat => 10
  | Fruit.Cherries => 42
  | Fruit.Apples => 16

-- Define the theorem
theorem max_loquat_wholesale (x : ℝ) :
  -- Conditions
  (wholesale_price Fruit.Cherries = wholesale_price Fruit.Loquat + 28) →
  (80 * wholesale_price Fruit.Loquat + 120 * wholesale_price Fruit.Cherries = 4960) →
  (∃ y : ℝ, x * wholesale_price Fruit.Loquat + 
            (160 - x) * wholesale_price Fruit.Apples + 
            y * wholesale_price Fruit.Cherries = 5280) →
  (x * (retail_price Fruit.Loquat - wholesale_price Fruit.Loquat) +
   (160 - x) * (retail_price Fruit.Apples - wholesale_price Fruit.Apples) +
   ((5280 - x * wholesale_price Fruit.Loquat - (160 - x) * wholesale_price Fruit.Apples) / 
    wholesale_price Fruit.Cherries) * 
   (retail_price Fruit.Cherries - wholesale_price Fruit.Cherries) ≥ 1120) →
  -- Conclusion
  x ≤ 60 :=
by sorry


end max_loquat_wholesale_l3338_333817


namespace new_cards_for_500_l3338_333886

/-- Given a total number of cards, calculate the number of new cards received
    when trading one-fifth of the duplicate cards, where duplicates are one-fourth
    of the total. -/
def new_cards_received (total : ℕ) : ℕ :=
  (total / 4) / 5

/-- Theorem stating that given 500 total cards, the number of new cards
    received is 25. -/
theorem new_cards_for_500 : new_cards_received 500 = 25 := by
  sorry

end new_cards_for_500_l3338_333886


namespace subtraction_difference_l3338_333892

theorem subtraction_difference : 
  let total : ℝ := 7000
  let one_tenth : ℝ := 1 / 10
  let one_tenth_percent : ℝ := 1 / 1000
  (one_tenth * total) - (one_tenth_percent * total) = 693 := by
  sorry

end subtraction_difference_l3338_333892


namespace customers_not_buying_coffee_l3338_333849

theorem customers_not_buying_coffee (total_customers : ℕ) (coffee_fraction : ℚ) : 
  total_customers = 25 → coffee_fraction = 3/5 → 
  total_customers - (coffee_fraction * total_customers).floor = 10 :=
by sorry

end customers_not_buying_coffee_l3338_333849


namespace inequality_of_reciprocal_logs_l3338_333819

theorem inequality_of_reciprocal_logs (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) :
  1 / Real.log a > 1 / Real.log b :=
by sorry

end inequality_of_reciprocal_logs_l3338_333819


namespace intersection_of_M_and_N_l3338_333856

def M : Set ℤ := {0, 1, 2}
def N : Set ℤ := {x | -1 ≤ x ∧ x ≤ 1}

theorem intersection_of_M_and_N : M ∩ N = {0, 1} := by sorry

end intersection_of_M_and_N_l3338_333856


namespace geometric_sequence_common_ratio_l3338_333872

/-- Given a geometric sequence with positive terms and common ratio q,
    where S_n denotes the sum of the first n terms, prove that
    if 2^10 * S_30 + S_10 = (2^10 + 1) * S_20, then q = 1/2 -/
theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_positive : ∀ n, a n > 0)
  (h_geometric : ∀ n, a (n + 1) = q * a n)
  (S : ℕ → ℝ)
  (h_sum : ∀ n, S n = (a 0) * (1 - q^n) / (1 - q))
  (h_equation : 2^10 * S 30 + S 10 = (2^10 + 1) * S 20) :
  q = 1/2 :=
sorry

end geometric_sequence_common_ratio_l3338_333872


namespace xiao_ming_math_grade_l3338_333821

/-- Calculates a student's semester math grade based on component scores and weights -/
def semesterMathGrade (routineStudyScore midTermScore finalExamScore : ℝ) : ℝ :=
  0.3 * routineStudyScore + 0.3 * midTermScore + 0.4 * finalExamScore

/-- Xiao Ming's semester math grade is 92.4 points -/
theorem xiao_ming_math_grade :
  semesterMathGrade 90 90 96 = 92.4 := by sorry

end xiao_ming_math_grade_l3338_333821


namespace record_4800_steps_l3338_333826

/-- The standard number of steps per day -/
def standard : ℕ := 5000

/-- Function to calculate the recorded steps -/
def recordedSteps (actualSteps : ℕ) : ℤ :=
  (actualSteps : ℤ) - standard

/-- Theorem stating that 4800 steps should be recorded as -200 -/
theorem record_4800_steps :
  recordedSteps 4800 = -200 := by sorry

end record_4800_steps_l3338_333826


namespace range_of_x_l3338_333808

theorem range_of_x (x : ℝ) 
  (hP : x^2 - 2*x - 3 ≥ 0)
  (hQ : |1 - x/2| ≥ 1) :
  x ≥ 4 ∨ x ≤ -1 := by
sorry

end range_of_x_l3338_333808


namespace min_sum_with_linear_constraint_l3338_333830

theorem min_sum_with_linear_constraint (a b : ℕ) (h : 23 * a - 13 * b = 1) :
  ∃ (a' b' : ℕ), 23 * a' - 13 * b' = 1 ∧ a' + b' ≤ a + b ∧ a' + b' = 11 :=
sorry

end min_sum_with_linear_constraint_l3338_333830


namespace vector_decomposition_l3338_333846

/-- Prove that the given vector x can be decomposed in terms of vectors p, q, and r -/
theorem vector_decomposition (x p q r : ℝ × ℝ × ℝ) : 
  x = (15, -20, -1) → 
  p = (0, 2, 1) → 
  q = (0, 1, -1) → 
  r = (5, -3, 2) → 
  x = (-6 : ℝ) • p + (1 : ℝ) • q + (3 : ℝ) • r :=
by sorry

end vector_decomposition_l3338_333846


namespace function_inequality_l3338_333858

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, x * (deriv f x) + f x > 0) (a b : ℝ) (hab : a > b) : 
  a * f a > b * f b := by sorry

end function_inequality_l3338_333858


namespace circle_equation_proof_l3338_333873

/-- The equation of a circle with center (h, k) and radius r is (x - h)² + (y - k)² = r² -/
def is_circle_equation (h k r : ℝ) (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y, f x y = (x - h)^2 + (y - k)^2 - r^2

/-- A point (x, y) is on a line ax + by + c = 0 if it satisfies the equation -/
def point_on_line (a b c : ℝ) (x y : ℝ) : Prop :=
  a * x + b * y + c = 0

/-- A point (x, y) is on a circle if it satisfies the circle's equation -/
def point_on_circle (f : ℝ → ℝ → ℝ) (x y : ℝ) : Prop :=
  f x y = 0

theorem circle_equation_proof (f : ℝ → ℝ → ℝ) :
  is_circle_equation 1 1 2 f →
  (∀ x y, point_on_line 1 1 (-2) x y → point_on_circle f x y) →
  point_on_circle f 1 (-1) →
  point_on_circle f (-1) 1 →
  ∀ x y, f x y = (x - 1)^2 + (y - 1)^2 - 4 := by
  sorry

end circle_equation_proof_l3338_333873


namespace female_employees_with_advanced_degrees_l3338_333865

theorem female_employees_with_advanced_degrees
  (total_employees : ℕ)
  (total_females : ℕ)
  (total_advanced_degrees : ℕ)
  (males_college_only : ℕ)
  (h1 : total_employees = 200)
  (h2 : total_females = 120)
  (h3 : total_advanced_degrees = 100)
  (h4 : males_college_only = 40) :
  total_advanced_degrees - (total_employees - total_females - males_college_only) = 60 :=
by sorry

end female_employees_with_advanced_degrees_l3338_333865


namespace bookshop_revenue_l3338_333895

-- Define book types and their prices
structure BookType where
  name : String
  price : Nat

-- Define a day's transactions
structure DayTransactions where
  novels_sold : Nat
  comics_sold : Nat
  biographies_sold : Nat
  novels_returned : Nat
  comics_returned : Nat
  biographies_returned : Nat
  discount : Nat  -- Discount percentage (0 for no discount)

def calculate_revenue (novel : BookType) (comic : BookType) (biography : BookType) 
                      (monday : DayTransactions) (tuesday : DayTransactions) 
                      (wednesday : DayTransactions) (thursday : DayTransactions) 
                      (friday : DayTransactions) : Nat :=
  sorry  -- Proof to be implemented

theorem bookshop_revenue : 
  let novel : BookType := { name := "Novel", price := 10 }
  let comic : BookType := { name := "Comic", price := 5 }
  let biography : BookType := { name := "Biography", price := 15 }
  
  let monday : DayTransactions := {
    novels_sold := 30, comics_sold := 20, biographies_sold := 25,
    novels_returned := 1, comics_returned := 5, biographies_returned := 0,
    discount := 0
  }
  
  let tuesday : DayTransactions := {
    novels_sold := 20, comics_sold := 10, biographies_sold := 20,
    novels_returned := 0, comics_returned := 0, biographies_returned := 0,
    discount := 20
  }
  
  let wednesday : DayTransactions := {
    novels_sold := 30, comics_sold := 20, biographies_sold := 14,
    novels_returned := 5, comics_returned := 0, biographies_returned := 3,
    discount := 0
  }
  
  let thursday : DayTransactions := {
    novels_sold := 40, comics_sold := 25, biographies_sold := 13,
    novels_returned := 0, comics_returned := 0, biographies_returned := 0,
    discount := 10
  }
  
  let friday : DayTransactions := {
    novels_sold := 55, comics_sold := 40, biographies_sold := 40,
    novels_returned := 2, comics_returned := 5, biographies_returned := 3,
    discount := 0
  }
  
  calculate_revenue novel comic biography monday tuesday wednesday thursday friday = 3603 :=
by sorry


end bookshop_revenue_l3338_333895


namespace log_equation_holds_l3338_333888

theorem log_equation_holds (x : ℝ) (h1 : x > 0) (h2 : x ≠ 1) :
  (Real.log x / Real.log 3) * (Real.log 5 / Real.log x) = Real.log 5 / Real.log 3 := by
  sorry


end log_equation_holds_l3338_333888


namespace jackson_money_l3338_333836

/-- Proves that given two people where one has 5 times more money than the other, 
    and together they have $150, the person with more money has $125. -/
theorem jackson_money (williams_money : ℝ) 
  (h1 : williams_money + 5 * williams_money = 150) : 
  5 * williams_money = 125 := by
  sorry

end jackson_money_l3338_333836


namespace largest_common_term_l3338_333813

def sequence1 (n : ℕ) : ℤ := 2 + 4 * (n - 1)
def sequence2 (n : ℕ) : ℤ := 5 + 6 * (n - 1)

def is_common_term (x : ℤ) : Prop :=
  ∃ (n m : ℕ), sequence1 n = x ∧ sequence2 m = x

def is_in_range (x : ℤ) : Prop := 1 ≤ x ∧ x ≤ 200

theorem largest_common_term :
  ∃ (x : ℤ), is_common_term x ∧ is_in_range x ∧
  ∀ (y : ℤ), is_common_term y ∧ is_in_range y → y ≤ x ∧
  x = 190 :=
sorry

end largest_common_term_l3338_333813


namespace arithmetic_progression_of_primes_l3338_333850

theorem arithmetic_progression_of_primes (a : ℕ → ℕ) (d : ℕ) :
  (∀ i ∈ Finset.range 15, Nat.Prime (a i)) →
  (∀ i ∈ Finset.range 14, a (i + 1) = a i + d) →
  d > 0 →
  a 0 > 15 →
  d > 30000 := by
sorry

end arithmetic_progression_of_primes_l3338_333850


namespace cube_sum_and_reciprocal_l3338_333825

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = 7) : x^3 + 1/x^3 = 322 := by
  sorry

end cube_sum_and_reciprocal_l3338_333825


namespace sum_greater_than_four_l3338_333810

theorem sum_greater_than_four (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y > x + y) : x + y > 4 := by
  sorry

end sum_greater_than_four_l3338_333810


namespace cone_volume_from_sector_l3338_333870

/-- Given a cone whose lateral surface develops into a sector with central angle 4π/3 and area 6π,
    the volume of the cone is (4√5/3)π. -/
theorem cone_volume_from_sector (θ r l h V : ℝ) : 
  θ = (4 / 3) * Real.pi →  -- Central angle of the sector
  (1 / 2) * l^2 * θ = 6 * Real.pi →  -- Area of the sector
  2 * Real.pi * r = θ * l →  -- Circumference of base equals arc length of sector
  h^2 + r^2 = l^2 →  -- Pythagorean theorem for cone dimensions
  V = (1 / 3) * Real.pi * r^2 * h →  -- Volume formula for cone
  V = (4 * Real.sqrt 5 / 3) * Real.pi := by
sorry

end cone_volume_from_sector_l3338_333870


namespace apple_banana_cost_l3338_333827

/-- The total cost of buying apples and bananas -/
def total_cost (a b : ℝ) : ℝ := 3 * a + 2 * b

/-- Theorem stating that the total cost of buying 3 kg of apples at 'a' yuan/kg
    and 2 kg of bananas at 'b' yuan/kg is (3a + 2b) yuan -/
theorem apple_banana_cost (a b : ℝ) :
  total_cost a b = 3 * a + 2 * b := by sorry

end apple_banana_cost_l3338_333827


namespace circle_equation_l3338_333832

/-- Given two circles C1 and C2 where:
    1. C1 has equation (x-1)^2 + (y-1)^2 = 1
    2. The coordinate axes are common tangents of C1 and C2
    3. The distance between the centers of C1 and C2 is 3√2
    Then the equation of C2 must be one of:
    (x-4)^2 + (y-4)^2 = 16
    (x+2)^2 + (y+2)^2 = 4
    (x-2√2)^2 + (y+2√2)^2 = 8
    (x+2√2)^2 + (y-2√2)^2 = 8 -/
theorem circle_equation (C1 C2 : Set (ℝ × ℝ)) : 
  (∀ x y, (x-1)^2 + (y-1)^2 = 1 ↔ (x, y) ∈ C1) →
  (∀ x, (x, 0) ∈ C1 → (x, 0) ∈ C2) →
  (∀ y, (0, y) ∈ C1 → (0, y) ∈ C2) →
  (∃ x₁ y₁ x₂ y₂, (x₁, y₁) ∈ C1 ∧ (x₂, y₂) ∈ C2 ∧ (x₁ - x₂)^2 + (y₁ - y₂)^2 = 18) →
  (∀ x y, (x, y) ∈ C2 ↔ 
    ((x-4)^2 + (y-4)^2 = 16) ∨
    ((x+2)^2 + (y+2)^2 = 4) ∨
    ((x-2*Real.sqrt 2)^2 + (y+2*Real.sqrt 2)^2 = 8) ∨
    ((x+2*Real.sqrt 2)^2 + (y-2*Real.sqrt 2)^2 = 8)) :=
by sorry

end circle_equation_l3338_333832


namespace hat_price_after_discounts_l3338_333871

/-- The final price of an item after two successive discounts --/
def finalPrice (originalPrice : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  originalPrice * (1 - discount1) * (1 - discount2)

/-- Theorem stating that a $20 item with 20% and 25% successive discounts results in a $12 final price --/
theorem hat_price_after_discounts :
  finalPrice 20 0.2 0.25 = 12 := by
  sorry

end hat_price_after_discounts_l3338_333871


namespace kenya_peanuts_l3338_333883

theorem kenya_peanuts (jose_peanuts : ℕ) (kenya_additional_peanuts : ℕ) 
  (h1 : jose_peanuts = 85)
  (h2 : kenya_additional_peanuts = 48) :
  jose_peanuts + kenya_additional_peanuts = 133 :=
by sorry

end kenya_peanuts_l3338_333883


namespace election_result_l3338_333812

/-- Represents an election with five candidates -/
structure Election :=
  (total_votes : ℕ)
  (votes_A : ℕ)
  (votes_B : ℕ)
  (votes_C : ℕ)
  (votes_D : ℕ)
  (votes_E : ℕ)

/-- Conditions for the election -/
def ElectionConditions (e : Election) : Prop :=
  e.votes_A = (30 * e.total_votes) / 100 ∧
  e.votes_B = (25 * e.total_votes) / 100 ∧
  e.votes_C = (20 * e.total_votes) / 100 ∧
  e.votes_D = (15 * e.total_votes) / 100 ∧
  e.votes_E = e.total_votes - (e.votes_A + e.votes_B + e.votes_C + e.votes_D) ∧
  e.votes_A = e.votes_B + 1200

theorem election_result (e : Election) (h : ElectionConditions e) :
  e.total_votes = 24000 ∧ e.votes_E = 2400 := by
  sorry

end election_result_l3338_333812


namespace triangle_side_length_l3338_333879

theorem triangle_side_length (B C BDC : Real) (BD : Real) :
  B = π/6 → -- 30°
  C = π/4 → -- 45°
  BDC = 5*π/6 → -- 150°
  BD = 5 →
  ∃ (AB : Real), AB = 5 * Real.sqrt 3 := by
  sorry

end triangle_side_length_l3338_333879


namespace existence_of_nth_root_l3338_333834

theorem existence_of_nth_root (n b : ℕ) (hn : n > 1) (hb : b > 1)
  (h : ∀ k : ℕ, k > 1 → ∃ a : ℤ, (k : ℤ) ∣ b - a^n) :
  ∃ A : ℤ, b = A^n := by
sorry

end existence_of_nth_root_l3338_333834


namespace dolls_count_l3338_333851

/-- Given that Hannah has 5 times as many dolls as her sister, and her sister has 8 dolls,
    prove that they have 48 dolls altogether. -/
theorem dolls_count (hannah_dolls : ℕ) (sister_dolls : ℕ) : 
  hannah_dolls = 5 * sister_dolls → sister_dolls = 8 → hannah_dolls + sister_dolls = 48 := by
  sorry

end dolls_count_l3338_333851


namespace am_gm_for_even_sum_l3338_333847

theorem am_gm_for_even_sum (a b : ℕ) (ha : a > 0) (hb : b > 0) (hsum : Even (a + b)) :
  (a + b : ℝ) / 2 ≥ Real.sqrt (a * b) := by
  sorry

end am_gm_for_even_sum_l3338_333847


namespace ratio_x_to_y_l3338_333863

theorem ratio_x_to_y (x y : ℚ) (h : (12 * x - 5 * y) / (16 * x - 3 * y) = 4 / 7) :
  x / y = 23 / 20 := by
  sorry

end ratio_x_to_y_l3338_333863


namespace max_area_rectangular_pen_l3338_333897

/-- The maximum area of a rectangular pen with a perimeter of 60 feet -/
theorem max_area_rectangular_pen :
  let perimeter : ℝ := 60
  let area (x : ℝ) : ℝ := x * (perimeter / 2 - x)
  ∀ x, 0 < x → x < perimeter / 2 → area x ≤ 225 :=
by sorry

end max_area_rectangular_pen_l3338_333897


namespace perpendicular_lines_b_value_l3338_333859

/-- 
Given two lines in the form of linear equations:
  3y - 2x - 6 = 0 and 4y + bx - 5 = 0
If these lines are perpendicular, then b = 6.
-/
theorem perpendicular_lines_b_value (b : ℝ) : 
  (∀ x y, 3 * y - 2 * x - 6 = 0 → 
           4 * y + b * x - 5 = 0 → 
           (2 / 3) * (-b / 4) = -1) → 
  b = 6 := by
sorry

end perpendicular_lines_b_value_l3338_333859


namespace total_weekly_revenue_l3338_333839

def normal_price : ℝ := 5

def monday_sales : ℕ := 9
def tuesday_sales : ℕ := 12
def wednesday_sales : ℕ := 18
def thursday_sales : ℕ := 14
def friday_sales : ℕ := 16
def saturday_sales : ℕ := 20
def sunday_sales : ℕ := 11

def wednesday_discount : ℝ := 0.1
def friday_discount : ℝ := 0.05

def daily_revenue (sales : ℕ) (discount : ℝ) : ℝ :=
  (sales : ℝ) * normal_price * (1 - discount)

theorem total_weekly_revenue :
  daily_revenue monday_sales 0 +
  daily_revenue tuesday_sales 0 +
  daily_revenue wednesday_sales wednesday_discount +
  daily_revenue thursday_sales 0 +
  daily_revenue friday_sales friday_discount +
  daily_revenue saturday_sales 0 +
  daily_revenue sunday_sales 0 = 487 := by
  sorry

end total_weekly_revenue_l3338_333839


namespace simple_interest_problem_l3338_333878

/-- Given a principal amount and an unknown interest rate, 
    if increasing the rate by 8% for 15 years results in 2,750 more interest,
    then the principal amount is 2,291.67 -/
theorem simple_interest_problem (P R : ℝ) : 
  (P * R * 15 / 100 + 2750 = P * (R + 8) * 15 / 100) → 
  P = 2291.67 := by
  sorry

end simple_interest_problem_l3338_333878


namespace shares_distribution_l3338_333838

/-- Proves that if 120 rs are divided among three people (a, b, c) such that a's share is 20 rs more than b's and 20 rs less than c's, then b's share is 20 rs. -/
theorem shares_distribution (a b c : ℕ) : 
  (a + b + c = 120) →  -- Total amount is 120 rs
  (a = b + 20) →       -- a's share is 20 rs more than b's
  (c = a + 20) →       -- c's share is 20 rs more than a's
  b = 20 :=            -- b's share is 20 rs
by sorry


end shares_distribution_l3338_333838


namespace cos_squared_alpha_plus_pi_fourth_l3338_333831

theorem cos_squared_alpha_plus_pi_fourth (α : ℝ) (h : Real.sin (2 * α) = 2 / 3) :
  Real.cos (α + π / 4) ^ 2 = 1 / 6 := by
  sorry

end cos_squared_alpha_plus_pi_fourth_l3338_333831


namespace mappings_count_l3338_333800

theorem mappings_count (A B : Finset Char) :
  A = {('a' : Char), 'b'} →
  B = {('c' : Char), 'd'} →
  Fintype.card (A → B) = 4 := by
  sorry

end mappings_count_l3338_333800


namespace debate_club_election_l3338_333816

def election_ways (n m k : ℕ) : ℕ :=
  (n - k).factorial / ((n - k - m).factorial * m.factorial) +
  k.factorial * (n - k).choose (m - k)

theorem debate_club_election :
  election_ways 30 5 4 = 6378720 :=
sorry

end debate_club_election_l3338_333816


namespace perpendicular_lines_l3338_333869

theorem perpendicular_lines (a : ℝ) : 
  (∀ x y : ℝ, 2 * y + x + 3 = 0 ∧ 3 * y + a * x + 2 = 0 → 
    ((-1/2) * (-a/3) = -1)) → 
  a = -6 := by
sorry

end perpendicular_lines_l3338_333869


namespace gorilla_exhibit_visitors_l3338_333896

def visitors_per_hour : ℕ := 50
def open_hours : ℕ := 8
def gorilla_exhibit_percentage : ℚ := 4/5

theorem gorilla_exhibit_visitors :
  (visitors_per_hour * open_hours : ℚ) * gorilla_exhibit_percentage = 320 := by
  sorry

end gorilla_exhibit_visitors_l3338_333896


namespace min_sticks_for_13_triangles_l3338_333876

/-- The minimum number of sticks needed to form n equilateral triangles -/
def min_sticks (n : ℕ) : ℕ := 2 * n + 1

/-- Theorem: Given the conditions for forming 1, 2, and 3 equilateral triangles,
    the minimum number of sticks required to form 13 equilateral triangles is 27 -/
theorem min_sticks_for_13_triangles :
  (min_sticks 1 = 3) →
  (min_sticks 2 = 5) →
  (min_sticks 3 = 7) →
  min_sticks 13 = 27 := by
  sorry

end min_sticks_for_13_triangles_l3338_333876


namespace pool_ground_area_l3338_333881

theorem pool_ground_area (length width : ℝ) (h1 : length = 5) (h2 : width = 4) :
  length * width = 20 := by
  sorry

end pool_ground_area_l3338_333881


namespace trapezoid_area_l3338_333853

-- Define the trapezoid ABCD
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  height : ℝ

-- Define the circle γ
structure Circle where
  radius : ℝ
  center_in_trapezoid : Bool
  tangent_to_AB_BC_DA : Bool
  arc_angle : ℝ

-- Define the problem
def trapezoid_circle_problem (ABCD : Trapezoid) (γ : Circle) : Prop :=
  ABCD.AB = 10 ∧
  ABCD.CD = 15 ∧
  γ.radius = 6 ∧
  γ.center_in_trapezoid = true ∧
  γ.tangent_to_AB_BC_DA = true ∧
  γ.arc_angle = 120

-- Theorem statement
theorem trapezoid_area (ABCD : Trapezoid) (γ : Circle) :
  trapezoid_circle_problem ABCD γ →
  (ABCD.AB + ABCD.CD) * ABCD.height / 2 = 225 / 2 := by
  sorry

end trapezoid_area_l3338_333853


namespace lateral_surface_is_parallelogram_l3338_333805

-- Define the types for our geometric objects
inductive PrismType
| Right
| Oblique

-- Define the shapes we're considering
inductive Shape
| Rectangle
| Parallelogram

-- Define a function that returns the possible shapes of a prism's lateral surface
def lateralSurfaceShape (p : PrismType) : Set Shape :=
  match p with
  | PrismType.Right => {Shape.Rectangle}
  | PrismType.Oblique => {Shape.Rectangle, Shape.Parallelogram}

-- Theorem statement
theorem lateral_surface_is_parallelogram :
  ∀ (p : PrismType), ∃ (s : Shape), s ∈ lateralSurfaceShape p → s = Shape.Parallelogram := by
  sorry

#check lateral_surface_is_parallelogram

end lateral_surface_is_parallelogram_l3338_333805
