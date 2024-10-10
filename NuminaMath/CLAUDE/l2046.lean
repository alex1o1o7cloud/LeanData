import Mathlib

namespace equal_cost_at_20_minutes_unique_solution_l2046_204646

/-- The base rate for United Telephone service -/
def united_base_rate : ℝ := 11

/-- The per-minute charge for United Telephone -/
def united_per_minute : ℝ := 0.25

/-- The base rate for Atlantic Call service -/
def atlantic_base_rate : ℝ := 12

/-- The per-minute charge for Atlantic Call -/
def atlantic_per_minute : ℝ := 0.20

/-- The total cost for United Telephone service for m minutes -/
def united_cost (m : ℝ) : ℝ := united_base_rate + united_per_minute * m

/-- The total cost for Atlantic Call service for m minutes -/
def atlantic_cost (m : ℝ) : ℝ := atlantic_base_rate + atlantic_per_minute * m

/-- Theorem stating that the costs are equal at 20 minutes -/
theorem equal_cost_at_20_minutes : 
  united_cost 20 = atlantic_cost 20 :=
by sorry

/-- Theorem stating that 20 minutes is the unique solution -/
theorem unique_solution (m : ℝ) :
  united_cost m = atlantic_cost m ↔ m = 20 :=
by sorry

end equal_cost_at_20_minutes_unique_solution_l2046_204646


namespace grasshopper_can_return_to_start_l2046_204604

/-- Represents the position of the grasshopper on a 2D plane -/
structure Position where
  x : Int
  y : Int

/-- Represents a single jump of the grasshopper -/
structure Jump where
  distance : Nat
  direction : Nat  -- 0: right, 1: up, 2: left, 3: down

/-- Applies a jump to a position -/
def applyJump (pos : Position) (jump : Jump) : Position :=
  match jump.direction % 4 with
  | 0 => ⟨pos.x + jump.distance, pos.y⟩
  | 1 => ⟨pos.x, pos.y + jump.distance⟩
  | 2 => ⟨pos.x - jump.distance, pos.y⟩
  | _ => ⟨pos.x, pos.y - jump.distance⟩

/-- Generates the nth jump -/
def nthJump (n : Nat) : Jump :=
  ⟨n, n - 1⟩

/-- Theorem: The grasshopper can return to the starting point -/
theorem grasshopper_can_return_to_start :
  ∃ (jumps : List Jump), 
    let finalPos := jumps.foldl applyJump ⟨0, 0⟩
    finalPos.x = 0 ∧ finalPos.y = 0 :=
  sorry


end grasshopper_can_return_to_start_l2046_204604


namespace factorization_x12_minus_729_l2046_204623

theorem factorization_x12_minus_729 (x : ℝ) :
  x^12 - 729 = (x^2 + 3) * (x^4 - 3*x^2 + 9) * (x^2 - 3) * (x^4 + 3*x^2 + 9) := by
  sorry

end factorization_x12_minus_729_l2046_204623


namespace smallest_five_digit_multiple_of_9_starting_with_7_l2046_204656

def is_multiple_of_9 (n : ℕ) : Prop := ∃ k : ℕ, n = 9 * k

def starts_with_7 (n : ℕ) : Prop := ∃ m : ℕ, n = 70000 + m ∧ m < 30000

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

theorem smallest_five_digit_multiple_of_9_starting_with_7 :
  ∀ n : ℕ, is_five_digit n → starts_with_7 n → is_multiple_of_9 n → n ≥ 70002 :=
by sorry

end smallest_five_digit_multiple_of_9_starting_with_7_l2046_204656


namespace units_digit_factorial_sum_15_l2046_204658

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def factorial_sum (n : ℕ) : ℕ := (List.range n).map factorial |> List.sum

theorem units_digit_factorial_sum_15 :
  units_digit (factorial_sum 15) = 3 := by sorry

end units_digit_factorial_sum_15_l2046_204658


namespace max_value_theorem_max_value_achievable_l2046_204652

theorem max_value_theorem (x y : ℝ) : 
  (2*x + 3*y + 4) / Real.sqrt (2*x^2 + 3*y^2 + 5) ≤ Real.sqrt 17 :=
sorry

theorem max_value_achievable : 
  ∃ (x y : ℝ), (2*x + 3*y + 4) / Real.sqrt (2*x^2 + 3*y^2 + 5) = Real.sqrt 17 :=
sorry

end max_value_theorem_max_value_achievable_l2046_204652


namespace jump_rope_solution_l2046_204690

/-- The cost of jump ropes A and B satisfy the given conditions -/
def jump_rope_cost (cost_A cost_B : ℝ) : Prop :=
  10 * cost_A + 5 * cost_B = 175 ∧ 15 * cost_A + 10 * cost_B = 300

/-- The solution to the jump rope cost problem -/
theorem jump_rope_solution :
  ∃ (cost_A cost_B : ℝ), jump_rope_cost cost_A cost_B ∧ cost_A = 10 ∧ cost_B = 15 := by
  sorry

#check jump_rope_solution

end jump_rope_solution_l2046_204690


namespace sum_of_imaginary_parts_is_zero_l2046_204653

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the quadratic equation
def quadratic_equation (z : ℂ) : Prop :=
  z^2 - 3*z = 8 - 6*i

-- Theorem statement
theorem sum_of_imaginary_parts_is_zero :
  ∃ (z₁ z₂ : ℂ), quadratic_equation z₁ ∧ quadratic_equation z₂ ∧ 
  z₁ ≠ z₂ ∧ (z₁.im + z₂.im = 0) :=
sorry

end sum_of_imaginary_parts_is_zero_l2046_204653


namespace mysoon_ornament_collection_l2046_204618

theorem mysoon_ornament_collection :
  ∀ (O : ℕ), 
    (O / 6 + 10 : ℕ) = (O / 3 : ℕ) * 2 →  -- Condition 1 and 2 combined
    (O / 3 : ℕ) = O / 3 →                 -- Condition 3
    O = 20 := by
  sorry

end mysoon_ornament_collection_l2046_204618


namespace bid_probabilities_theorem_l2046_204632

/-- Represents the probability of winning a bid for a project -/
structure BidProbability where
  value : ℝ
  is_probability : 0 ≤ value ∧ value ≤ 1

/-- Represents the probabilities of winning bids for three projects -/
structure ProjectProbabilities where
  a : BidProbability
  b : BidProbability
  c : BidProbability
  a_gt_b : a.value > b.value
  c_eq_quarter : c.value = 1/4

/-- The main theorem stating the properties of the bid probabilities -/
theorem bid_probabilities_theorem (p : ProjectProbabilities) : 
  p.a.value * p.b.value * p.c.value = 1/24 ∧
  1 - (1 - p.a.value) * (1 - p.b.value) * (1 - p.c.value) = 3/4 →
  p.a.value = 1/2 ∧ p.b.value = 1/3 ∧
  p.a.value * p.b.value * (1 - p.c.value) + 
  p.a.value * (1 - p.b.value) * p.c.value + 
  (1 - p.a.value) * p.b.value * p.c.value = 5/24 := by
  sorry

end bid_probabilities_theorem_l2046_204632


namespace integer_solutions_of_equation_l2046_204626

def solution_set : Set (ℤ × ℤ) := {(-1, -1), (0, -1), (-1, 0), (0, 0), (5, 2), (-6, 2)}

theorem integer_solutions_of_equation :
  {(x, y) : ℤ × ℤ | x^2 + x = y^4 + y^3 + y^2 + y} = solution_set :=
by sorry

end integer_solutions_of_equation_l2046_204626


namespace second_exam_study_time_l2046_204614

/-- Represents the relationship between study time and test score -/
structure StudyScoreRelation where
  study_time : ℝ
  test_score : ℝ
  inverse_relation : study_time * test_score = study_time * test_score

/-- Theorem stating the required study time for the second exam -/
theorem second_exam_study_time 
  (first_exam : StudyScoreRelation)
  (h : first_exam.study_time = 6 ∧ first_exam.test_score = 60)
  (second_exam : StudyScoreRelation)
  (average_score : ℝ)
  (h_average : average_score = 90)
  (h_total_score : first_exam.test_score + second_exam.test_score = 2 * average_score) :
  second_exam.study_time = 3 ∧ second_exam.test_score = 120 := by
  sorry

#check second_exam_study_time

end second_exam_study_time_l2046_204614


namespace brinley_zoo_count_l2046_204678

/-- The number of animals Brinley counted at the San Diego Zoo --/
def total_animals (snakes arctic_foxes leopards bee_eaters cheetahs alligators : ℕ) : ℕ :=
  snakes + arctic_foxes + leopards + bee_eaters + cheetahs + alligators

/-- Theorem stating the total number of animals Brinley counted at the zoo --/
theorem brinley_zoo_count : ∃ (snakes arctic_foxes leopards bee_eaters cheetahs alligators : ℕ),
  snakes = 100 ∧
  arctic_foxes = 80 ∧
  leopards = 20 ∧
  bee_eaters = 10 * leopards ∧
  cheetahs = snakes / 2 ∧
  alligators = 2 * (arctic_foxes + leopards) ∧
  total_animals snakes arctic_foxes leopards bee_eaters cheetahs alligators = 650 :=
by
  sorry


end brinley_zoo_count_l2046_204678


namespace propositions_truth_values_l2046_204628

def proposition1 : Prop := (100 % 10 = 0) ∧ (100 % 5 = 0)

def proposition2 : Prop := (3^2 - 9 = 0) ∨ ((-3)^2 - 9 = 0)

def proposition3 : Prop := ¬(2^2 - 9 = 0)

theorem propositions_truth_values :
  proposition1 ∧ proposition2 ∧ ¬proposition3 :=
sorry

end propositions_truth_values_l2046_204628


namespace oil_quantity_function_correct_l2046_204610

/-- Represents the remaining oil quantity in liters after t minutes -/
def Q (t : ℝ) : ℝ := 20 - 0.2 * t

/-- The initial oil quantity in liters -/
def initial_quantity : ℝ := 20

/-- The outflow rate in liters per minute -/
def outflow_rate : ℝ := 0.2

theorem oil_quantity_function_correct : 
  ∀ t : ℝ, t ≥ 0 → Q t = initial_quantity - outflow_rate * t :=
sorry

end oil_quantity_function_correct_l2046_204610


namespace cubic_equation_solutions_l2046_204641

theorem cubic_equation_solutions :
  let f : ℂ → ℂ := λ x => (x^3 + 4*x^2*Real.sqrt 3 + 12*x + 8*Real.sqrt 3) + (x + 2*Real.sqrt 3)
  ∃ (z₁ z₂ z₃ : ℂ),
    z₁ = -2 * Real.sqrt 3 ∧
    z₂ = -2 * Real.sqrt 3 + Complex.I ∧
    z₃ = -2 * Real.sqrt 3 - Complex.I ∧
    (∀ z : ℂ, f z = 0 ↔ z = z₁ ∨ z = z₂ ∨ z = z₃) :=
by
  sorry

end cubic_equation_solutions_l2046_204641


namespace current_speed_l2046_204697

theorem current_speed (speed_with_current speed_against_current : ℝ) 
  (h1 : speed_with_current = 21)
  (h2 : speed_against_current = 16) :
  ∃ (man_speed current_speed : ℝ),
    man_speed + current_speed = speed_with_current ∧
    man_speed - current_speed = speed_against_current ∧
    current_speed = 2.5 := by
  sorry

end current_speed_l2046_204697


namespace complement_of_A_in_U_l2046_204620

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 > 0}

-- State the theorem
theorem complement_of_A_in_U : 
  Set.compl A = Set.Icc (-1 : ℝ) (3 : ℝ) := by sorry

end complement_of_A_in_U_l2046_204620


namespace centroid_quadrilateral_area_l2046_204622

-- Define the square ABCD
structure Square :=
  (sideLength : ℝ)

-- Define a point P inside the square
structure PointInSquare :=
  (distanceAP : ℝ)
  (distanceBP : ℝ)

-- Define the quadrilateral formed by centroids
structure CentroidQuadrilateral :=
  (diagonalLength : ℝ)

-- Define the theorem
theorem centroid_quadrilateral_area
  (s : Square)
  (p : PointInSquare)
  (q : CentroidQuadrilateral)
  (h1 : s.sideLength = 30)
  (h2 : p.distanceAP = 12)
  (h3 : p.distanceBP = 26)
  (h4 : q.diagonalLength = 20) :
  q.diagonalLength * q.diagonalLength / 2 = 200 :=
sorry

end centroid_quadrilateral_area_l2046_204622


namespace complement_A_intersect_B_l2046_204640

open Set

def A : Set ℝ := {x | |x - 1| ≥ 2}
def B : Set ℕ := {x | x < 4}

theorem complement_A_intersect_B :
  (𝒰 \ A) ∩ (coe '' B) = {0, 1, 2} := by sorry

end complement_A_intersect_B_l2046_204640


namespace circle_area_with_diameter_6_l2046_204637

theorem circle_area_with_diameter_6 (π : ℝ) (h : π > 0) :
  let diameter : ℝ := 6
  let radius : ℝ := diameter / 2
  let area : ℝ := π * radius ^ 2
  area = 9 * π := by sorry

end circle_area_with_diameter_6_l2046_204637


namespace simplify_trig_expression_l2046_204630

theorem simplify_trig_expression :
  (Real.sin (35 * π / 180))^2 - 1/2 = 
  -2 * (Real.cos (10 * π / 180) * Real.cos (80 * π / 180)) := by
  sorry

end simplify_trig_expression_l2046_204630


namespace sunny_lead_second_race_l2046_204629

/-- Represents a runner in the races -/
structure Runner where
  speed : ℝ

/-- Represents the race conditions -/
structure RaceConditions where
  raceLength : ℝ
  sunnyLeadFirstRace : ℝ
  sunnyStartBehind : ℝ
  windyDelay : ℝ

/-- Calculate the lead of Sunny at the end of the second race -/
def calculateSunnyLead (sunny : Runner) (windy : Runner) (conditions : RaceConditions) : ℝ :=
  sorry

/-- Theorem stating that Sunny finishes 56.25 meters ahead in the second race -/
theorem sunny_lead_second_race (sunny : Runner) (windy : Runner) (conditions : RaceConditions) :
  conditions.raceLength = 400 ∧
  conditions.sunnyLeadFirstRace = 50 ∧
  conditions.sunnyStartBehind = 50 ∧
  conditions.windyDelay = 10 →
  calculateSunnyLead sunny windy conditions = 56.25 :=
by
  sorry

end sunny_lead_second_race_l2046_204629


namespace pure_imaginary_condition_l2046_204698

/-- A complex number z is pure imaginary if its real part is zero and its imaginary part is non-zero. -/
def IsPureImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- The complex number z as a function of real number m. -/
def z (m : ℝ) : ℂ :=
  Complex.mk (m^2 + 3*m + 2) (m^2 - m - 6)

theorem pure_imaginary_condition (m : ℝ) :
  IsPureImaginary (z m) ↔ m = -1 := by
  sorry

end pure_imaginary_condition_l2046_204698


namespace child_tickets_sold_l2046_204613

theorem child_tickets_sold (adult_price child_price total_tickets total_revenue : ℕ) 
  (h1 : adult_price = 7)
  (h2 : child_price = 4)
  (h3 : total_tickets = 900)
  (h4 : total_revenue = 5100) :
  ∃ (adult_tickets child_tickets : ℕ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_price * adult_tickets + child_price * child_tickets = total_revenue ∧
    child_tickets = 400 := by
  sorry

end child_tickets_sold_l2046_204613


namespace possible_a3_values_l2046_204680

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h_d : d ≠ 0
  h_arithmetic : ∀ n, a (n + 1) = a n + d

/-- Theorem: Possible values of a_3 in the arithmetic sequence -/
theorem possible_a3_values (seq : ArithmeticSequence) 
  (h_a5 : seq.a 5 = 6)
  (h_a3_gt_1 : seq.a 3 > 1)
  (h_geometric : ∃ (m : ℕ → ℕ), 
    (∀ t, 5 < m t ∧ (t > 0 → m (t-1) < m t)) ∧ 
    (∀ t, ∃ r, seq.a (m t) = seq.a 3 * r^(t+1) ∧ seq.a 5 = seq.a 3 * r^2)) :
  seq.a 3 = 3 ∨ seq.a 3 = 2 ∨ seq.a 3 = 3/2 :=
sorry

end possible_a3_values_l2046_204680


namespace equation_solution_l2046_204681

theorem equation_solution :
  ∀ x : ℝ, x ≠ 0 ∧ 8*x + 3 ≠ 0 ∧ 7*x - 3 ≠ 0 →
    (2 + 5/(4*x) - 15/(4*x*(8*x+3)) = 2*(7*x+1)/(7*x-3)) ↔ x = 9 :=
by
  sorry

end equation_solution_l2046_204681


namespace circle_angle_problem_l2046_204686

theorem circle_angle_problem (x y : ℝ) : 
  y = 2 * x → 7 * x + 6 * x + 3 * x + (2 * x + y) = 360 → x = 18 := by
  sorry

end circle_angle_problem_l2046_204686


namespace joe_pocket_transfer_l2046_204634

/-- Represents the money transfer problem with Joe's pockets --/
def MoneyTransferProblem (total initial_left transfer_amount : ℚ) : Prop :=
  let initial_right := total - initial_left
  let after_quarter_left := initial_left - (initial_left / 4)
  let after_quarter_right := initial_right + (initial_left / 4)
  let final_left := after_quarter_left - transfer_amount
  let final_right := after_quarter_right + transfer_amount
  (total = 200) ∧ 
  (initial_left = 160) ∧ 
  (final_left = final_right) ∧
  (transfer_amount > 0)

theorem joe_pocket_transfer : 
  ∃ (transfer_amount : ℚ), MoneyTransferProblem 200 160 transfer_amount ∧ transfer_amount = 20 := by
  sorry

end joe_pocket_transfer_l2046_204634


namespace jaco_total_payment_l2046_204654

/-- Calculates the total amount a customer pays given item prices and a discount policy. -/
def calculateTotalWithDiscount (shoePrice sockPrice bagPrice : ℚ) : ℚ :=
  let totalBeforeDiscount := shoePrice + 2 * sockPrice + bagPrice
  let discountableAmount := max (totalBeforeDiscount - 100) 0
  let discount := discountableAmount * (1 / 10)
  totalBeforeDiscount - discount

/-- Theorem stating that Jaco will pay $118 for his purchases. -/
theorem jaco_total_payment :
  calculateTotalWithDiscount 74 2 42 = 118 := by
  sorry

#eval calculateTotalWithDiscount 74 2 42

end jaco_total_payment_l2046_204654


namespace parallel_vectors_mn_value_l2046_204691

def vector_a (m n : ℝ) : Fin 3 → ℝ := λ i =>
  match i with
  | 0 => 2
  | 1 => 2*m - 3
  | 2 => n + 2

def vector_b (m n : ℝ) : Fin 3 → ℝ := λ i =>
  match i with
  | 0 => 4
  | 1 => 2*m + 1
  | 2 => 3*n - 2

theorem parallel_vectors_mn_value (m n : ℝ) :
  (∃ (k : ℝ), ∀ (i : Fin 3), vector_a m n i = k * vector_b m n i) →
  m * n = 21 := by
sorry

end parallel_vectors_mn_value_l2046_204691


namespace yoongi_number_division_l2046_204677

theorem yoongi_number_division (x : ℤ) : 
  x - 17 = 55 → x / 9 = 8 := by
  sorry

end yoongi_number_division_l2046_204677


namespace function_properties_monotone_interval_l2046_204627

def f (a b x : ℝ) : ℝ := a * x^3 + b * x^2

theorem function_properties (a b : ℝ) :
  f a b 1 = 4 ∧ 
  (3 * a * (-2)^2 + 2 * b * (-2) = 0) →
  a = 1 ∧ b = 3 :=
sorry

def g (x : ℝ) : ℝ := x^3 + 3 * x^2

theorem monotone_interval (m : ℝ) :
  (∀ x ∈ Set.Ioo m (m + 1), MonotoneOn g (Set.Ioo m (m + 1))) →
  m ≤ -3 ∨ m ≥ 0 :=
sorry

end function_properties_monotone_interval_l2046_204627


namespace combined_molecular_weight_mixture_l2046_204607

-- Define atomic weights
def atomic_weight_Ca : ℝ := 40.08
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.01
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00

-- Define molecular weights
def molecular_weight_CaO : ℝ := atomic_weight_Ca + atomic_weight_O
def molecular_weight_CO2 : ℝ := atomic_weight_C + 2 * atomic_weight_O
def molecular_weight_HNO3 : ℝ := atomic_weight_H + atomic_weight_N + 3 * atomic_weight_O

-- Define the mixture composition
def moles_CaO : ℝ := 5
def moles_CO2 : ℝ := 3
def moles_HNO3 : ℝ := 2

-- Theorem statement
theorem combined_molecular_weight_mixture :
  moles_CaO * molecular_weight_CaO +
  moles_CO2 * molecular_weight_CO2 +
  moles_HNO3 * molecular_weight_HNO3 = 538.45 := by
  sorry

end combined_molecular_weight_mixture_l2046_204607


namespace cupboard_sale_percentage_l2046_204619

def cost_price : ℝ := 6875
def additional_amount : ℝ := 1650
def profit_percentage : ℝ := 12

theorem cupboard_sale_percentage (selling_price : ℝ) 
  (h1 : selling_price + additional_amount = cost_price * (1 + profit_percentage / 100)) :
  (cost_price - selling_price) / cost_price * 100 = profit_percentage := by
sorry

end cupboard_sale_percentage_l2046_204619


namespace present_difference_l2046_204642

/-- The number of presents Santana buys for her siblings in a year --/
def presents_count : ℕ → ℕ
| 1 => 4  -- March
| 2 => 1  -- May
| 3 => 1  -- June
| 4 => 1  -- October
| 5 => 1  -- November
| 6 => 2  -- December
| _ => 0

/-- The total number of siblings Santana has --/
def total_siblings : ℕ := 10

/-- The number of presents bought in the first half of the year --/
def first_half_presents : ℕ := presents_count 1 + presents_count 2 + presents_count 3

/-- The number of presents bought in the second half of the year --/
def second_half_presents : ℕ := 
  presents_count 4 + presents_count 5 + presents_count 6 + total_siblings + total_siblings

theorem present_difference : second_half_presents - first_half_presents = 18 := by
  sorry

#eval second_half_presents - first_half_presents

end present_difference_l2046_204642


namespace y_derivative_l2046_204675

open Real

noncomputable def y (x : ℝ) : ℝ := 2 * (cos x / sin x ^ 4) + 3 * (cos x / sin x ^ 2)

theorem y_derivative (x : ℝ) (h : sin x ≠ 0) : 
  deriv y x = 3 * (1 / sin x) - 8 * (1 / sin x) ^ 5 := by
sorry

end y_derivative_l2046_204675


namespace min_value_of_sum_l2046_204662

theorem min_value_of_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 4/y = 1) :
  x + y ≥ 9 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 1/x + 4/y = 1 ∧ x + y = 9 :=
by sorry

end min_value_of_sum_l2046_204662


namespace intersection_chord_length_l2046_204664

theorem intersection_chord_length (k : ℝ) : 
  (∀ x y : ℝ, y = k * x + 3 → (x - 3)^2 + (y - 2)^2 = 4 → 
    ∃ M N : ℝ × ℝ, 
      (M.1 - 3)^2 + (M.2 - 2)^2 = 4 ∧ 
      (N.1 - 3)^2 + (N.2 - 2)^2 = 4 ∧ 
      M.2 = k * M.1 + 3 ∧ 
      N.2 = k * N.1 + 3 ∧ 
      (M.1 - N.1)^2 + (M.2 - N.2)^2 ≥ 12) → 
  -3/4 ≤ k ∧ k ≤ 0 :=
sorry

end intersection_chord_length_l2046_204664


namespace point_outside_circle_l2046_204603

theorem point_outside_circle (a b : ℝ) (i : ℂ) : 
  i * i = -1 → 
  Complex.I = i →
  a + b * i = (2 + i) / (1 - i) → 
  a^2 + b^2 > 2 := by
  sorry

end point_outside_circle_l2046_204603


namespace ellipse_major_axis_length_l2046_204665

/-- The length of the major axis of an ellipse C, given specific conditions -/
theorem ellipse_major_axis_length : 
  ∀ (m : ℝ) (x y : ℝ → ℝ),
    (m > 0) →
    (∀ t, 2 * (x t) - (y t) + 4 = 0) →
    (∀ t, (x t)^2 / m + (y t)^2 / 2 = 1) →
    (∃ t₀, (x t₀, y t₀) = (-2, 0) ∨ (x t₀, y t₀) = (0, 4)) →
    ∃ a b : ℝ, a^2 = m ∧ b^2 = 2 ∧ 2 * max a b = 2 * Real.sqrt 6 :=
by sorry

end ellipse_major_axis_length_l2046_204665


namespace deduction_is_three_l2046_204636

/-- Calculates the deduction per idle day for a worker --/
def calculate_deduction_per_idle_day (total_days : ℕ) (pay_rate : ℕ) (total_payment : ℕ) (idle_days : ℕ) : ℕ :=
  let working_days := total_days - idle_days
  let total_earnings := working_days * pay_rate
  (total_earnings - total_payment) / idle_days

/-- Theorem: Given the conditions, the deduction per idle day is 3 --/
theorem deduction_is_three :
  calculate_deduction_per_idle_day 60 20 280 40 = 3 := by
  sorry

#eval calculate_deduction_per_idle_day 60 20 280 40

end deduction_is_three_l2046_204636


namespace inequality_system_solutions_l2046_204616

theorem inequality_system_solutions :
  let S : Set (ℝ × ℝ) := {(x, y) | 
    x^4 + 8*x^3*y + 16*x^2*y^2 + 16 ≤ 8*x^2 + 32*x*y ∧
    y^4 + 64*x^2*y^2 + 10*y^2 + 25 ≤ 16*x*y^3 + 80*x*y}
  S = {(2/Real.sqrt 11, 5/Real.sqrt 11), 
       (-2/Real.sqrt 11, -5/Real.sqrt 11),
       (2/Real.sqrt 3, 1/Real.sqrt 3), 
       (-2/Real.sqrt 3, -1/Real.sqrt 3)} := by
  sorry


end inequality_system_solutions_l2046_204616


namespace parentheses_placement_count_l2046_204621

/-- A sequence of prime numbers -/
def primeSequence : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

/-- The operation of placing parentheses in the expression -/
def parenthesesPlacement (seq : List Nat) : Nat :=
  2^(seq.length - 2)

/-- Theorem stating the number of different values obtained by placing parentheses -/
theorem parentheses_placement_count :
  parenthesesPlacement primeSequence = 256 := by
  sorry

end parentheses_placement_count_l2046_204621


namespace quadrilateral_area_l2046_204608

/-- The area of a quadrilateral with non-perpendicular diagonals -/
theorem quadrilateral_area (a b c d φ : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hφ : 0 < φ ∧ φ < π / 2) :
  let S := Real.tan φ * |a^2 + c^2 - b^2 - d^2| / 4
  ∃ (d₁ d₂ : ℝ), d₁ > 0 ∧ d₂ > 0 ∧ S = d₁ * d₂ * Real.sin φ / 2 := by
  sorry

end quadrilateral_area_l2046_204608


namespace multiply_decimals_l2046_204673

theorem multiply_decimals : (2.4 : ℝ) * 0.2 = 0.48 := by
  sorry

end multiply_decimals_l2046_204673


namespace jace_initial_earnings_l2046_204672

theorem jace_initial_earnings (debt : ℕ) (remaining : ℕ) (h1 : debt = 358) (h2 : remaining = 642) :
  debt + remaining = 1000 := by
  sorry

end jace_initial_earnings_l2046_204672


namespace average_income_l2046_204643

/-- Given the average monthly incomes of pairs of individuals and the income of one individual,
    calculate the average monthly income of the remaining pair. -/
theorem average_income (p q r : ℕ) : 
  (p + q) / 2 = 2050 →
  (p + r) / 2 = 6200 →
  p = 3000 →
  (q + r) / 2 = 5250 := by
  sorry


end average_income_l2046_204643


namespace collinear_points_theorem_l2046_204606

/-- Given vectors OA, OB, OC in R², if A, B, C are collinear, then the x-coordinate of OA is 18 -/
theorem collinear_points_theorem (k : ℝ) : 
  let OA : Fin 2 → ℝ := ![k, 12]
  let OB : Fin 2 → ℝ := ![4, 5]
  let OC : Fin 2 → ℝ := ![10, 8]
  (∃ (t : ℝ), (OC - OB) = t • (OA - OB)) → k = 18 := by
  sorry

end collinear_points_theorem_l2046_204606


namespace square_plus_one_nonzero_l2046_204611

theorem square_plus_one_nonzero : ∀ x : ℝ, x^2 + 1 ≠ 0 := by sorry

end square_plus_one_nonzero_l2046_204611


namespace tan_order_l2046_204663

theorem tan_order : 
  (1 < Real.pi / 2) → 
  (Real.pi / 2 < 2) → 
  (2 < 3) → 
  (3 < Real.pi) → 
  (∀ x y, Real.pi / 2 < x → x < y → y < Real.pi → Real.tan x < Real.tan y) →
  Real.tan 1 > Real.tan 3 ∧ Real.tan 3 > Real.tan 2 :=
by sorry

end tan_order_l2046_204663


namespace impossible_c_nine_l2046_204605

/-- An obtuse triangle with sides a, b, and c -/
structure ObtuseTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_obtuse : (a^2 + b^2 < c^2) ∨ (a^2 + c^2 < b^2) ∨ (b^2 + c^2 < a^2)
  h_triangle_inequality : (a + b > c) ∧ (a + c > b) ∧ (b + c > a)
  h_positive : a > 0 ∧ b > 0 ∧ c > 0

/-- The theorem stating that c = 9 is impossible for the given obtuse triangle -/
theorem impossible_c_nine (t : ObtuseTriangle) (h1 : t.a = 6) (h2 : t.b = 8) : t.c ≠ 9 := by
  sorry

#check impossible_c_nine

end impossible_c_nine_l2046_204605


namespace least_x_squared_divisible_by_240_l2046_204687

theorem least_x_squared_divisible_by_240 :
  ∀ x : ℕ, x > 0 → x^2 % 240 = 0 → x ≥ 60 :=
by
  sorry

end least_x_squared_divisible_by_240_l2046_204687


namespace hyperbola_iff_mn_positive_l2046_204631

-- Define the condition for a curve to be a hyperbola
def is_hyperbola (m n : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ ∀ (x y : ℝ), m * x^2 - n * y^2 = 1 ↔ (x / a)^2 - (y / b)^2 = 1

-- State the theorem
theorem hyperbola_iff_mn_positive (m n : ℝ) :
  is_hyperbola m n ↔ m * n > 0 := by sorry

end hyperbola_iff_mn_positive_l2046_204631


namespace fibonacci_divisibility_l2046_204693

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- Extractable in p-arithmetic -/
def extractable_in_p_arithmetic (x : ℝ) (p : ℕ) : Prop := sorry

theorem fibonacci_divisibility (p k : ℕ) (h_prime : Nat.Prime p) 
  (h_sqrt5 : extractable_in_p_arithmetic (Real.sqrt 5) p) :
  p^k ∣ fib (p^(k-1) * (p-1)) :=
sorry

end fibonacci_divisibility_l2046_204693


namespace selling_price_a_is_1600_l2046_204666

/-- Represents the sales and pricing information for bicycle types A and B --/
structure BikeData where
  lastYearTotalSalesA : ℕ
  priceDecreaseA : ℕ
  salesDecreasePercentage : ℚ
  purchasePriceA : ℕ
  purchasePriceB : ℕ
  sellingPriceB : ℕ

/-- Calculates the selling price of type A bikes this year --/
def calculateSellingPriceA (data : BikeData) : ℕ :=
  sorry

/-- Theorem stating that the selling price of type A bikes this year is 1600 yuan --/
theorem selling_price_a_is_1600 (data : BikeData) 
  (h1 : data.lastYearTotalSalesA = 50000)
  (h2 : data.priceDecreaseA = 400)
  (h3 : data.salesDecreasePercentage = 1/5)
  (h4 : data.purchasePriceA = 1100)
  (h5 : data.purchasePriceB = 1400)
  (h6 : data.sellingPriceB = 2000) :
  calculateSellingPriceA data = 1600 :=
sorry

end selling_price_a_is_1600_l2046_204666


namespace fourth_root_of_polynomial_l2046_204659

theorem fourth_root_of_polynomial (a b : ℝ) : 
  (∀ x : ℝ, b * x^3 + (3*b + a) * x^2 + (a - 2*b) * x + (5 - b) = 0 ↔ 
    x = -1 ∨ x = 2 ∨ x = 4 ∨ x = -8) → 
  ∃ x : ℝ, x = -8 ∧ b * x^3 + (3*b + a) * x^2 + (a - 2*b) * x + (5 - b) = 0 :=
by sorry

end fourth_root_of_polynomial_l2046_204659


namespace cricket_team_average_age_l2046_204685

/-- The average age of a cricket team given specific conditions -/
theorem cricket_team_average_age : 
  ∀ (team_size : ℕ) (captain_age : ℕ) (wicket_keeper_age_diff : ℕ) (A : ℚ),
  team_size = 11 →
  captain_age = 25 →
  wicket_keeper_age_diff = 3 →
  (team_size : ℚ) * A = 
    ((team_size - 2) : ℚ) * (A - 1) + 
    (captain_age : ℚ) + 
    ((captain_age + wicket_keeper_age_diff) : ℚ) →
  A = 31 := by
sorry

end cricket_team_average_age_l2046_204685


namespace max_value_theorem_l2046_204633

theorem max_value_theorem (x : ℝ) (h : x > 0) :
  (x^2 + 3 - Real.sqrt (x^4 + 6)) / x ≤ 36 / (2 * Real.sqrt 3 + Real.sqrt (2 * Real.sqrt 6)) :=
sorry

end max_value_theorem_l2046_204633


namespace horner_method_operations_l2046_204694

def f (x : ℝ) : ℝ := x^6 + 1

def horner_eval (x : ℝ) : ℝ := ((((((x * x + 0) * x + 0) * x + 0) * x + 0) * x + 0) * x + 1)

theorem horner_method_operations (x : ℝ) :
  (∃ (exp_count mult_count add_count : ℕ),
    horner_eval x = f x ∧
    exp_count = 0 ∧
    mult_count = 6 ∧
    add_count = 6) :=
sorry

end horner_method_operations_l2046_204694


namespace square_sum_given_diff_and_product_l2046_204699

theorem square_sum_given_diff_and_product (a b : ℝ) 
  (h1 : a - b = 3) 
  (h2 : a * b = 8) : 
  a^2 + b^2 = 25 := by
sorry

end square_sum_given_diff_and_product_l2046_204699


namespace balls_after_2010_steps_l2046_204661

/-- Converts a natural number to its base-6 representation -/
def toBase6 (n : ℕ) : List ℕ :=
  if n < 6 then [n]
  else (n % 6) :: toBase6 (n / 6)

/-- Sums the digits in a list -/
def sumDigits (digits : List ℕ) : ℕ :=
  digits.sum

theorem balls_after_2010_steps :
  sumDigits (toBase6 2010) = 10 := by
  sorry

end balls_after_2010_steps_l2046_204661


namespace smallest_prime_minister_l2046_204660

/-- A positive integer is primer if it has a prime number of distinct prime factors. -/
def isPrimer (n : ℕ+) : Prop := sorry

/-- A positive integer is primest if it has a primer number of distinct primer factors. -/
def isPrimest (n : ℕ+) : Prop := sorry

/-- A positive integer is prime-minister if it has a primest number of distinct primest factors. -/
def isPrimeMinister (n : ℕ+) : Prop := sorry

/-- The smallest prime-minister number -/
def smallestPrimeMinister : ℕ+ := 378000

theorem smallest_prime_minister :
  isPrimeMinister smallestPrimeMinister ∧
  ∀ n : ℕ+, n < smallestPrimeMinister → ¬isPrimeMinister n := by
  sorry

end smallest_prime_minister_l2046_204660


namespace polynomial_factorization_l2046_204639

theorem polynomial_factorization (x y z : ℝ) :
  2 * x^3 - x^2 * z - 4 * x^2 * y + 2 * x * y * z + 2 * x * y^2 - y^2 * z = (2 * x - z) * (x - y)^2 := by
  sorry

end polynomial_factorization_l2046_204639


namespace happy_street_traffic_happy_street_traffic_proof_l2046_204696

theorem happy_street_traffic (tuesday : ℕ) (thursday friday weekend_day : ℕ) 
  (total : ℕ) : ℕ :=
  let monday := tuesday - tuesday / 5
  let thursday_to_sunday := thursday + friday + 2 * weekend_day
  let monday_to_wednesday := total - thursday_to_sunday
  let wednesday := monday_to_wednesday - (monday + tuesday)
  wednesday - monday

#check happy_street_traffic 25 10 10 5 97 = 2

theorem happy_street_traffic_proof : 
  happy_street_traffic 25 10 10 5 97 = 2 := by
sorry

end happy_street_traffic_happy_street_traffic_proof_l2046_204696


namespace johns_roommates_multiple_l2046_204668

/-- Given that Bob has 10 roommates and John has 25 roommates, 
    prove that the multiple of Bob's roommates that John has five more than is 2. -/
theorem johns_roommates_multiple (bob_roommates john_roommates : ℕ) : 
  bob_roommates = 10 → john_roommates = 25 → 
  ∃ (x : ℕ), john_roommates = x * bob_roommates + 5 ∧ x = 2 := by
sorry

end johns_roommates_multiple_l2046_204668


namespace num_correct_statements_is_zero_l2046_204617

/-- Represents a vector in 3D space -/
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Two vectors are parallel -/
def parallel (v1 v2 : Vector3D) : Prop := sorry

/-- A vector is a unit vector -/
def is_unit_vector (v : Vector3D) : Prop := sorry

/-- Two vectors are collinear -/
def collinear (v1 v2 : Vector3D) : Prop := sorry

/-- The zero vector -/
def zero_vector : Vector3D := ⟨0, 0, 0⟩

/-- Theorem: The number of correct statements is 0 -/
theorem num_correct_statements_is_zero : 
  ¬(∀ (v1 v2 : Vector3D) (p : Point3D), is_unit_vector v1 → is_unit_vector v2 → v1.x = p.x ∧ v1.y = p.y ∧ v1.z = p.z → v2.x = p.x ∧ v2.y = p.y ∧ v2.z = p.z) ∧ 
  ¬(∀ (A B C D : Point3D), parallel ⟨B.x - A.x, B.y - A.y, B.z - A.z⟩ ⟨D.x - C.x, D.y - C.y, D.z - C.z⟩ → 
    ∃ (t : ℝ), C = ⟨A.x + t * (B.x - A.x), A.y + t * (B.y - A.y), A.z + t * (B.z - A.z)⟩) ∧
  ¬(∀ (a b c : Vector3D), parallel a b → parallel b c → b ≠ zero_vector → parallel a c) ∧
  ¬(∀ (v1 v2 : Vector3D) (A B C D : Point3D), 
    collinear v1 v2 → 
    v1.x = B.x - A.x ∧ v1.y = B.y - A.y ∧ v1.z = B.z - A.z →
    v2.x = D.x - C.x ∧ v2.y = D.y - C.y ∧ v2.z = D.z - C.z →
    A ≠ C → B ≠ D) :=
by sorry

end num_correct_statements_is_zero_l2046_204617


namespace cos_squared_sum_range_l2046_204651

theorem cos_squared_sum_range (α β : ℝ) (h : 3 * (Real.sin α)^2 + 2 * (Real.sin β)^2 - 2 * Real.sin α = 0) :
  ∃ (x : ℝ), x ∈ Set.Icc (14/9 : ℝ) 2 ∧
  x = (Real.cos α)^2 + (Real.cos β)^2 ∧
  ∀ (y : ℝ), y = (Real.cos α)^2 + (Real.cos β)^2 → y ∈ Set.Icc (14/9 : ℝ) 2 :=
sorry

end cos_squared_sum_range_l2046_204651


namespace robin_extra_gum_l2046_204612

/-- The number of extra pieces of gum Robin has -/
def extra_gum (packages : ℕ) (pieces_per_package : ℕ) (total_pieces : ℕ) : ℕ :=
  total_pieces - (packages * pieces_per_package)

/-- Theorem: Robin has 8 extra pieces of gum -/
theorem robin_extra_gum :
  extra_gum 43 23 997 = 8 := by
  sorry

end robin_extra_gum_l2046_204612


namespace point_B_coordinate_l2046_204635

def point_A : ℝ := -1

theorem point_B_coordinate (point_B : ℝ) (h : |point_B - point_A| = 3) :
  point_B = 2 ∨ point_B = -4 := by
  sorry

end point_B_coordinate_l2046_204635


namespace travel_agency_comparison_l2046_204650

/-- Represents the fee calculation for a travel agency. -/
structure TravelAgency where
  parentDiscount : ℝ  -- Discount for parents (1 means no discount)
  studentDiscount : ℝ  -- Discount for students
  basePrice : ℝ        -- Base price per person

/-- Calculate the total fee for a travel agency given the number of students. -/
def calculateFee (agency : TravelAgency) (numStudents : ℝ) : ℝ :=
  agency.basePrice * (2 * agency.parentDiscount + numStudents * agency.studentDiscount)

/-- Travel Agency A with full price for parents and 70% for students. -/
def agencyA : TravelAgency :=
  { parentDiscount := 1
  , studentDiscount := 0.7
  , basePrice := 500 }

/-- Travel Agency B with 80% price for both parents and students. -/
def agencyB : TravelAgency :=
  { parentDiscount := 0.8
  , studentDiscount := 0.8
  , basePrice := 500 }

theorem travel_agency_comparison :
  ∀ x : ℝ,
    (calculateFee agencyA x = 350 * x + 1000) ∧
    (calculateFee agencyB x = 400 * x + 800) ∧
    (0 < x ∧ x < 4 → calculateFee agencyB x < calculateFee agencyA x) ∧
    (x = 4 → calculateFee agencyA x = calculateFee agencyB x) ∧
    (x > 4 → calculateFee agencyA x < calculateFee agencyB x) :=
by sorry

end travel_agency_comparison_l2046_204650


namespace business_card_exchanges_count_l2046_204679

/-- Represents a business conference with two groups of people -/
structure BusinessConference where
  total_people : ℕ
  group1_size : ℕ
  group2_size : ℕ
  h_total : total_people = group1_size + group2_size
  h_group1 : group1_size = 25
  h_group2 : group2_size = 15

/-- Calculates the number of business card exchanges in a business conference -/
def business_card_exchanges (conf : BusinessConference) : ℕ :=
  conf.group1_size * conf.group2_size

/-- Theorem stating that the number of business card exchanges is 375 -/
theorem business_card_exchanges_count (conf : BusinessConference) :
  business_card_exchanges conf = 375 := by
  sorry

#eval business_card_exchanges ⟨40, 25, 15, rfl, rfl, rfl⟩

end business_card_exchanges_count_l2046_204679


namespace mirror_area_l2046_204670

/-- The area of a rectangular mirror fitting exactly inside a frame -/
theorem mirror_area (frame_length frame_width frame_thickness : ℕ) 
  (h1 : frame_length = 100)
  (h2 : frame_width = 130)
  (h3 : frame_thickness = 15) : 
  (frame_length - 2 * frame_thickness) * (frame_width - 2 * frame_thickness) = 7000 :=
by sorry

end mirror_area_l2046_204670


namespace direct_proportion_information_needed_l2046_204615

/-- A structure representing a direct proportion between x and y -/
structure DirectProportion where
  k : ℝ  -- Constant of proportionality
  y : ℝ → ℝ  -- Function mapping x to y
  prop : ∀ x, y x = k * x  -- Property of direct proportion

/-- The number of pieces of information needed to determine a direct proportion -/
def informationNeeded : ℕ := 2

/-- Theorem stating that exactly 2 pieces of information are needed to determine a direct proportion -/
theorem direct_proportion_information_needed :
  ∀ (dp : DirectProportion), informationNeeded = 2 :=
by sorry

end direct_proportion_information_needed_l2046_204615


namespace average_and_difference_l2046_204695

theorem average_and_difference (y : ℝ) : 
  (47 + y) / 2 = 53 → |y - 47| = 12 := by
  sorry

end average_and_difference_l2046_204695


namespace quiz_probability_l2046_204688

theorem quiz_probability (n : ℕ) (m : ℕ) (p : ℚ) : 
  n = 6 → 
  m = 4 → 
  p = 1 - (3/4)^6 → 
  p = 3367/4096 :=
by sorry

end quiz_probability_l2046_204688


namespace balloon_arrangements_l2046_204683

def word_length : Nat := 7
def repeated_letters : Nat := 2
def repetitions_per_letter : Nat := 2

theorem balloon_arrangements :
  (word_length.factorial) / (repeated_letters.factorial * repetitions_per_letter.factorial) = 1260 := by
  sorry

end balloon_arrangements_l2046_204683


namespace teachers_combined_age_l2046_204671

theorem teachers_combined_age
  (num_students : ℕ)
  (student_avg_age : ℚ)
  (num_teachers : ℕ)
  (total_avg_age : ℚ)
  (h1 : num_students = 30)
  (h2 : student_avg_age = 18)
  (h3 : num_teachers = 2)
  (h4 : total_avg_age = 19) :
  (num_students + num_teachers) * total_avg_age -
  (num_students * student_avg_age) = 68 := by
sorry

end teachers_combined_age_l2046_204671


namespace smallest_angle_representation_l2046_204625

theorem smallest_angle_representation (k : ℤ) (α : ℝ) : 
  (19 * π / 5 = 2 * k * π + α) → 
  (∀ β : ℝ, ∃ m : ℤ, 19 * π / 5 = 2 * m * π + β → |α| ≤ |β|) → 
  α = -π / 5 := by
sorry

end smallest_angle_representation_l2046_204625


namespace second_train_length_proof_l2046_204644

/-- The length of the second train given the conditions of the problem -/
def second_train_length : ℝ := 199.9760019198464

/-- The length of the first train in meters -/
def first_train_length : ℝ := 100

/-- The speed of the first train in kilometers per hour -/
def first_train_speed : ℝ := 42

/-- The speed of the second train in kilometers per hour -/
def second_train_speed : ℝ := 30

/-- The time it takes for the trains to clear each other in seconds -/
def clearing_time : ℝ := 14.998800095992321

theorem second_train_length_proof :
  second_train_length = 
    (first_train_speed + second_train_speed) * (1000 / 3600) * clearing_time - first_train_length := by
  sorry

end second_train_length_proof_l2046_204644


namespace intersection_area_l2046_204692

/-- The area of intersection between two boards of widths 5 inches and 7 inches,
    crossing at a 45-degree angle. -/
theorem intersection_area (board1_width board2_width : ℝ) (angle : ℝ) :
  board1_width = 5 →
  board2_width = 7 →
  angle = π / 4 →
  (board1_width * board2_width * Real.sin angle) = (35 * Real.sqrt 2) / 2 := by
sorry

end intersection_area_l2046_204692


namespace carl_typing_words_l2046_204645

/-- Calculates the total number of words typed given a typing speed, daily typing duration, and number of days. -/
def total_words_typed (typing_speed : ℕ) (hours_per_day : ℕ) (days : ℕ) : ℕ :=
  typing_speed * 60 * hours_per_day * days

/-- Proves that given the specified conditions, Carl types 84000 words in 7 days. -/
theorem carl_typing_words :
  total_words_typed 50 4 7 = 84000 := by
  sorry

end carl_typing_words_l2046_204645


namespace heating_pad_cost_per_use_l2046_204689

/-- The cost per use of a heating pad -/
def cost_per_use (total_cost : ℚ) (uses_per_week : ℕ) (weeks : ℕ) : ℚ :=
  total_cost / (uses_per_week * weeks)

/-- Theorem: The cost per use of a $30 heating pad used 3 times a week for 2 weeks is $5 -/
theorem heating_pad_cost_per_use :
  cost_per_use 30 3 2 = 5 := by sorry

end heating_pad_cost_per_use_l2046_204689


namespace y_coordinates_descending_l2046_204601

/-- Given a line y = -2x + b and three points on this line, prove that the y-coordinates are in descending order as x increases. -/
theorem y_coordinates_descending 
  (b : ℝ) 
  (y₁ y₂ y₃ : ℝ) 
  (h1 : y₁ = 4 + b) 
  (h2 : y₂ = 2 + b) 
  (h3 : y₃ = -2 + b) : 
  y₃ < y₂ ∧ y₂ < y₁ := by
sorry

end y_coordinates_descending_l2046_204601


namespace calculation_difference_l2046_204682

def harry_calculation : ℤ := 12 - (3 + 4 * 2)

def terry_calculation : ℤ :=
  let step1 := 12 - 3
  let step2 := step1 + 4
  step2 * 2

theorem calculation_difference :
  harry_calculation - terry_calculation = -25 := by sorry

end calculation_difference_l2046_204682


namespace largest_guaranteed_divisor_l2046_204674

def die_faces : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

def Q (s : Finset ℕ) : ℕ := s.prod id

theorem largest_guaranteed_divisor :
  ∀ s : Finset ℕ, s ⊆ die_faces → s.card = 7 → 960 ∣ Q s ∧
  ∀ n : ℕ, n > 960 → ∃ t : Finset ℕ, t ⊆ die_faces ∧ t.card = 7 ∧ ¬(n ∣ Q t) :=
by sorry

end largest_guaranteed_divisor_l2046_204674


namespace problem_solution_l2046_204684

noncomputable section

def f (a : ℝ) (x : ℝ) := a * (3 : ℝ)^x + (3 : ℝ)^(-x)

def g (m : ℝ) (x : ℝ) := (Real.log x / Real.log 2)^2 + 2 * (Real.log x / Real.log 2) + m

theorem problem_solution :
  (∀ x, f a x = f a (-x)) →
  a = 1 ∧
  (∀ x y, 0 < x → x < y → f 1 x < f 1 y) ∧
  (∃ α β, α ≠ β ∧ 1/8 ≤ α ∧ α ≤ 4 ∧ 1/8 ≤ β ∧ β ≤ 4 ∧ g m α = 0 ∧ g m β = 0) →
  -3 ≤ m ∧ m < 1 ∧ α * β = 1/4 :=
sorry

end

end problem_solution_l2046_204684


namespace transformation_matrix_correct_l2046_204676

def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![0, -1; 1, 0]
def scaling_factor : ℝ := 2

theorem transformation_matrix_correct :
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![0, -2; 2, 0]
  ∀ (v : Fin 2 → ℝ),
    M.mulVec v = scaling_factor • (rotation_matrix.mulVec v) :=
by sorry

end transformation_matrix_correct_l2046_204676


namespace xiaogong_speed_l2046_204657

/-- The speed of Xiaogong in meters per minute -/
def v_x : ℝ := 28

/-- The speed of Dachen in meters per minute -/
def v_d : ℝ := v_x + 20

/-- The total distance between points A and B in meters -/
def total_distance : ℝ := 1200

/-- The time Dachen walks before meeting Xiaogong, in minutes -/
def t_d : ℝ := 18

/-- The time Xiaogong walks before meeting Dachen, in minutes -/
def t_x : ℝ := 12

theorem xiaogong_speed :
  v_x * t_x + v_d * t_d = total_distance ∧
  v_d = v_x + 20 →
  v_x = 28 := by sorry

end xiaogong_speed_l2046_204657


namespace arithmetic_sequence_common_difference_l2046_204609

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h1 : a 2 = 9) 
  (h2 : a 5 = 33) : 
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d ∧ d = 8 :=
sorry

end arithmetic_sequence_common_difference_l2046_204609


namespace toy_purchase_with_discount_l2046_204649

theorem toy_purchase_with_discount (num_toys : ℕ) (cost_per_toy : ℝ) (discount_percent : ℝ) :
  num_toys = 5 →
  cost_per_toy = 3 →
  discount_percent = 20 →
  (num_toys * cost_per_toy) * (1 - discount_percent / 100) = 12 :=
by
  sorry

end toy_purchase_with_discount_l2046_204649


namespace wendys_brother_candy_prove_wendys_brother_candy_l2046_204624

/-- Wendy's candy problem -/
theorem wendys_brother_candy : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun (wendys_boxes : ℕ) (pieces_per_box : ℕ) (total_pieces : ℕ) (brothers_pieces : ℕ) =>
    wendys_boxes * pieces_per_box + brothers_pieces = total_pieces →
    wendys_boxes = 2 →
    pieces_per_box = 3 →
    total_pieces = 12 →
    brothers_pieces = 6

/-- Proof of Wendy's candy problem -/
theorem prove_wendys_brother_candy : wendys_brother_candy 2 3 12 6 := by
  sorry

end wendys_brother_candy_prove_wendys_brother_candy_l2046_204624


namespace citadel_school_earnings_l2046_204647

/-- Represents the total earnings for a school in the summer project. -/
def schoolEarnings (totalPayment : ℚ) (totalStudentDays : ℕ) (schoolStudentDays : ℕ) : ℚ :=
  (totalPayment / totalStudentDays) * schoolStudentDays

/-- Theorem: The earnings for Citadel school in the summer project. -/
theorem citadel_school_earnings :
  let apexDays : ℕ := 9 * 5
  let beaconDays : ℕ := 3 * 4
  let citadelDays : ℕ := 6 * 7
  let totalDays : ℕ := apexDays + beaconDays + citadelDays
  let totalPayment : ℚ := 864
  schoolEarnings totalPayment totalDays citadelDays = 864 / 99 * 42 :=
by sorry

#eval schoolEarnings 864 99 42

end citadel_school_earnings_l2046_204647


namespace max_t_value_min_y_value_equality_condition_l2046_204638

-- Define the inequality function
def f (x t : ℝ) : ℝ := |3*x + 2| + |3*x - 1| - t

-- Part 1: Maximum value of t
theorem max_t_value :
  (∀ x : ℝ, f x 3 ≥ 0) ∧ 
  (∀ t : ℝ, t > 3 → ∃ x : ℝ, f x t < 0) :=
sorry

-- Part 2: Minimum value of y
theorem min_y_value :
  ∀ m n : ℝ, m > 0 → n > 0 → 4*m + 5*n = 3 →
  1 / (m + 2*n) + 4 / (3*m + 3*n) ≥ 3 :=
sorry

-- Equality condition
theorem equality_condition :
  ∀ m n : ℝ, m > 0 → n > 0 → 4*m + 5*n = 3 →
  (1 / (m + 2*n) + 4 / (3*m + 3*n) = 3 ↔ m = 1/3 ∧ n = 1/3) :=
sorry

end max_t_value_min_y_value_equality_condition_l2046_204638


namespace candidate_vote_difference_l2046_204600

theorem candidate_vote_difference (total_votes : ℝ) (candidate_percentage : ℝ) : 
  total_votes = 10000.000000000002 →
  candidate_percentage = 0.4 →
  (total_votes * (1 - candidate_percentage) - total_votes * candidate_percentage) = 2000 := by
sorry

end candidate_vote_difference_l2046_204600


namespace basketball_match_children_l2046_204602

/-- Calculates the number of children at a basketball match given the total number of spectators,
    the number of men, and the ratio of children to women. -/
def number_of_children (total : ℕ) (men : ℕ) (child_to_woman_ratio : ℕ) : ℕ :=
  let non_men := total - men
  let women := non_men / (child_to_woman_ratio + 1)
  child_to_woman_ratio * women

/-- Theorem stating that given the specific conditions of the basketball match,
    the number of children is 2500. -/
theorem basketball_match_children :
  number_of_children 10000 7000 5 = 2500 := by
  sorry

end basketball_match_children_l2046_204602


namespace c1_c2_not_collinear_l2046_204655

/-- Given two vectors a and b in ℝ³, we define c₁ and c₂ as linear combinations of a and b.
    This theorem states that c₁ and c₂ are not collinear. -/
theorem c1_c2_not_collinear (a b : ℝ × ℝ × ℝ) 
  (h1 : a = ⟨-9, 5, 3⟩) 
  (h2 : b = ⟨7, 1, -2⟩) : 
  ¬ ∃ (k : ℝ), 2 • a - b = k • (3 • a + 5 • b) :=
sorry

end c1_c2_not_collinear_l2046_204655


namespace point_A_on_circle_point_B_on_circle_point_C_on_circle_circle_equation_unique_l2046_204648

/-- A circle passes through points A(2, 0), B(4, 0), and C(0, 2) -/
def circle_through_points (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 3)^2 = 10

/-- Point A lies on the circle -/
theorem point_A_on_circle : circle_through_points 2 0 := by sorry

/-- Point B lies on the circle -/
theorem point_B_on_circle : circle_through_points 4 0 := by sorry

/-- Point C lies on the circle -/
theorem point_C_on_circle : circle_through_points 0 2 := by sorry

/-- The equation (x - 3)² + (y - 3)² = 10 represents the unique circle 
    passing through points A(2, 0), B(4, 0), and C(0, 2) -/
theorem circle_equation_unique : 
  ∀ x y : ℝ, circle_through_points x y ↔ (x - 3)^2 + (y - 3)^2 = 10 := by sorry

end point_A_on_circle_point_B_on_circle_point_C_on_circle_circle_equation_unique_l2046_204648


namespace exactly_two_valid_multiplications_l2046_204667

def is_valid_multiplication (a b : ℕ) : Prop :=
  100 ≤ a ∧ a < 1000 ∧  -- a is a three-digit number
  a / 100 = 1 ∧  -- a starts with 1
  1 ≤ b ∧ b < 10 ∧  -- b is a single-digit number
  1000 ≤ a * b ∧ a * b < 10000 ∧  -- product is four digits
  (a * (b % 10) / 100 = 1)  -- third row starts with '100'

theorem exactly_two_valid_multiplications :
  ∃! (s : Finset ℕ), s.card = 2 ∧ ∀ a ∈ s, ∃ b, is_valid_multiplication a b :=
sorry

end exactly_two_valid_multiplications_l2046_204667


namespace max_min_on_interval_l2046_204669

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 5

theorem max_min_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-1 : ℝ) 1, f x = max) ∧
    (∀ x ∈ Set.Icc (-1 : ℝ) 1, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-1 : ℝ) 1, f x = min) ∧
    max = 5 ∧ min = 1 :=
by sorry

end max_min_on_interval_l2046_204669
