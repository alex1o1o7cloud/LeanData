import Mathlib

namespace solution_l1927_192714

def complex_number_problem (z : ℂ) : Prop :=
  (∃ (r : ℝ), z - 3 * Complex.I = r) ∧
  (∃ (t : ℝ), (z - 5 * Complex.I) / (2 - Complex.I) = t * Complex.I)

theorem solution (z : ℂ) (h : complex_number_problem z) :
  z = -1 + 3 * Complex.I ∧ Complex.abs (z / (1 - Complex.I)) = Real.sqrt 5 := by
  sorry

end solution_l1927_192714


namespace polynomial_expansion_coefficient_l1927_192767

theorem polynomial_expansion_coefficient (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, x^10 = a + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + 
            a₅*(x-1)^5 + a₆*(x-1)^6 + a₇*(x-1)^7 + a₈*(x-1)^8 + a₉*(x-1)^9 + a₁₀*(x-1)^10) →
  a₈ = 45 := by
sorry

end polynomial_expansion_coefficient_l1927_192767


namespace complex_roots_distance_l1927_192786

/-- Given three complex numbers z₁, z₂, z₃ with |zⱼ| ≤ 1 for j = 1, 2, 3, 
    and w₁, w₂ being the roots of the equation 
    (z - z₁)(z - z₂) + (z - z₂)(z - z₃) + (z - z₃)(z - z₁) = 0,
    then for j = 1, 2, 3, min{|zⱼ - w₁|, |zⱼ - w₂|} ≤ 1. -/
theorem complex_roots_distance (z₁ z₂ z₃ w₁ w₂ : ℂ) 
  (h₁ : Complex.abs z₁ ≤ 1)
  (h₂ : Complex.abs z₂ ≤ 1)
  (h₃ : Complex.abs z₃ ≤ 1)
  (hw : (w₁ - z₁) * (w₁ - z₂) + (w₁ - z₂) * (w₁ - z₃) + (w₁ - z₃) * (w₁ - z₁) = 0 ∧
        (w₂ - z₁) * (w₂ - z₂) + (w₂ - z₂) * (w₂ - z₃) + (w₂ - z₃) * (w₂ - z₁) = 0) :
  (min (Complex.abs (z₁ - w₁)) (Complex.abs (z₁ - w₂)) ≤ 1) ∧
  (min (Complex.abs (z₂ - w₁)) (Complex.abs (z₂ - w₂)) ≤ 1) ∧
  (min (Complex.abs (z₃ - w₁)) (Complex.abs (z₃ - w₂)) ≤ 1) := by
  sorry

end complex_roots_distance_l1927_192786


namespace intersection_of_A_and_B_l1927_192724

def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | -2 ≤ x ∧ x < 1}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -1 < x ∧ x < 1} := by sorry

end intersection_of_A_and_B_l1927_192724


namespace blue_balls_count_l1927_192709

theorem blue_balls_count (total : ℕ) (red : ℕ) (orange : ℕ) (pink : ℕ) 
  (h1 : total = 50)
  (h2 : red = 20)
  (h3 : orange = 5)
  (h4 : pink = 3 * orange)
  : total - (red + orange + pink) = 10 := by
  sorry

end blue_balls_count_l1927_192709


namespace left_seats_count_l1927_192722

/-- Represents the seating configuration of a bus -/
structure BusSeating where
  leftSeats : ℕ
  rightSeats : ℕ
  backSeatCapacity : ℕ
  seatCapacity : ℕ
  totalCapacity : ℕ

/-- The bus seating configuration satisfies the given conditions -/
def validBusSeating (bus : BusSeating) : Prop :=
  bus.rightSeats = bus.leftSeats - 3 ∧
  bus.backSeatCapacity = 11 ∧
  bus.seatCapacity = 3 ∧
  bus.totalCapacity = 92 ∧
  bus.totalCapacity = bus.seatCapacity * (bus.leftSeats + bus.rightSeats) + bus.backSeatCapacity

/-- The number of seats on the left side of the bus is 15 -/
theorem left_seats_count (bus : BusSeating) (h : validBusSeating bus) : bus.leftSeats = 15 := by
  sorry

end left_seats_count_l1927_192722


namespace at_least_two_solved_five_l1927_192788

/-- The number of problems in the competition -/
def num_problems : ℕ := 6

/-- The structure representing a participant in the competition -/
structure Participant where
  solved : Finset (Fin num_problems)

/-- The type of the competition -/
structure Competition where
  participants : Finset Participant
  pair_solved : ∀ (i j : Fin num_problems), i ≠ j →
    (participants.filter (λ p => i ∈ p.solved ∧ j ∈ p.solved)).card >
    (2 * participants.card) / 5
  no_all_solved : ∀ p : Participant, p ∈ participants → p.solved.card < num_problems

/-- The main theorem -/
theorem at_least_two_solved_five (comp : Competition) :
  (comp.participants.filter (λ p => p.solved.card = num_problems - 1)).card ≥ 2 := by
  sorry

end at_least_two_solved_five_l1927_192788


namespace meeting_percentage_is_25_percent_l1927_192791

/-- Represents the duration of a work day in minutes -/
def work_day_minutes : ℕ := 8 * 60

/-- Represents the duration of the first meeting in minutes -/
def first_meeting_minutes : ℕ := 30

/-- Calculates the duration of the second meeting in minutes -/
def second_meeting_minutes : ℕ := 3 * first_meeting_minutes

/-- Calculates the total time spent in meetings in minutes -/
def total_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes

/-- Represents the percentage of the work day spent in meetings -/
def meeting_percentage : ℚ := (total_meeting_minutes : ℚ) / (work_day_minutes : ℚ) * 100

theorem meeting_percentage_is_25_percent :
  meeting_percentage = 25 := by sorry

end meeting_percentage_is_25_percent_l1927_192791


namespace sequence_properties_l1927_192727

def a (n : ℕ) : ℤ := 15 * n + 2 + (15 * n - 32) * 16^(n - 1)

theorem sequence_properties :
  (∀ n : ℕ, (15^3 : ℤ) ∣ a n) ∧
  (∀ n : ℕ, (1991 : ℤ) ∣ a n ∧ (1991 : ℤ) ∣ a (n + 1) ∧ (1991 : ℤ) ∣ a (n + 2) ↔ ∃ k : ℕ, n = 89595 * k) :=
by sorry

end sequence_properties_l1927_192727


namespace fiftieth_term_l1927_192775

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) - a n = 2

theorem fiftieth_term (a : ℕ → ℕ) (h : arithmetic_sequence a) : a 50 = 99 := by
  sorry

end fiftieth_term_l1927_192775


namespace greatest_prime_producing_integer_l1927_192784

def f (x : ℤ) : ℤ := |5 * x^2 - 52 * x + 21|

def is_greatest_prime_producing_integer (n : ℤ) : Prop :=
  Nat.Prime (f n).natAbs ∧
  ∀ m : ℤ, m > n → ¬(Nat.Prime (f m).natAbs)

theorem greatest_prime_producing_integer :
  is_greatest_prime_producing_integer 10 := by sorry

end greatest_prime_producing_integer_l1927_192784


namespace sallys_score_l1927_192720

/-- Calculates the score for a math contest given the number of correct, incorrect, and unanswered questions. -/
def calculate_score (correct : ℕ) (incorrect : ℕ) (unanswered : ℕ) : ℚ :=
  (correct : ℚ) - 0.25 * (incorrect : ℚ)

/-- Proves that Sally's score in the math contest is 12.5 -/
theorem sallys_score :
  let correct := 15
  let incorrect := 10
  let unanswered := 5
  calculate_score correct incorrect unanswered = 12.5 := by
  sorry

#eval calculate_score 15 10 5

end sallys_score_l1927_192720


namespace equation_represents_hyperbola_l1927_192701

/-- Represents a conic section --/
inductive ConicSection
  | Parabola
  | Circle
  | Ellipse
  | Hyperbola
  | Point
  | Line
  | TwoLines
  | Empty

/-- Determines the type of conic section represented by the given equation --/
def determineConicSection (a b c d e f : ℝ) : ConicSection :=
  sorry

/-- The equation x^2 - 64y^2 - 12x + 16y + 36 = 0 represents a hyperbola --/
theorem equation_represents_hyperbola :
  determineConicSection 1 (-64) 0 (-12) 16 36 = ConicSection.Hyperbola :=
by sorry

end equation_represents_hyperbola_l1927_192701


namespace solution_set_l1927_192729

theorem solution_set (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  {x : ℝ | a^(2*x - 7) > a^(4*x - 1)} = {x : ℝ | x > -3} := by
  sorry

end solution_set_l1927_192729


namespace nail_polish_count_l1927_192769

theorem nail_polish_count (kim heidi karen : ℕ) : 
  kim = 12 →
  heidi = kim + 5 →
  karen = kim - 4 →
  heidi + karen = 25 := by sorry

end nail_polish_count_l1927_192769


namespace clock_hands_alignment_l1927_192759

/-- The number of whole seconds remaining in an hour when the clock hands make equal angles with the vertical -/
def remaining_seconds : ℕ := by sorry

/-- The angle (in degrees) that the hour hand and minute hand make with the vertical when they align -/
def alignment_angle : ℚ := by sorry

theorem clock_hands_alignment :
  (alignment_angle * 120 : ℚ) = (360 - alignment_angle) * 10 ∧
  remaining_seconds = 3600 - Int.floor (alignment_angle * 120) := by sorry

end clock_hands_alignment_l1927_192759


namespace tank_capacity_is_90_l1927_192737

/-- Represents a gasoline tank with a certain capacity -/
structure GasolineTank where
  capacity : ℚ
  initialFraction : ℚ
  finalFraction : ℚ
  usedAmount : ℚ

/-- Theorem stating that the tank capacity is 90 gallons given the conditions -/
theorem tank_capacity_is_90 (tank : GasolineTank)
  (h1 : tank.initialFraction = 5/6)
  (h2 : tank.finalFraction = 2/3)
  (h3 : tank.usedAmount = 15)
  (h4 : tank.initialFraction * tank.capacity - tank.finalFraction * tank.capacity = tank.usedAmount) :
  tank.capacity = 90 := by
  sorry

end tank_capacity_is_90_l1927_192737


namespace f_at_2_l1927_192708

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x + 1

-- Theorem statement
theorem f_at_2 : f 2 = 259 := by
  sorry

end f_at_2_l1927_192708


namespace rectangular_field_area_l1927_192719

theorem rectangular_field_area (length width area : ℝ) : 
  length = width + 10 →
  length = 19.13 →
  area = length * width →
  area = 174.6359 := by
  sorry

end rectangular_field_area_l1927_192719


namespace brady_dwayne_earnings_difference_l1927_192706

/-- Given that Dwayne makes $1,500 in a year and Brady and Dwayne's combined earnings are $3,450 in a year,
    prove that Brady makes $450 more than Dwayne in a year. -/
theorem brady_dwayne_earnings_difference :
  let dwayne_earnings : ℕ := 1500
  let combined_earnings : ℕ := 3450
  let brady_earnings : ℕ := combined_earnings - dwayne_earnings
  brady_earnings - dwayne_earnings = 450 := by
sorry

end brady_dwayne_earnings_difference_l1927_192706


namespace cubic_roots_sum_cubes_l1927_192700

theorem cubic_roots_sum_cubes (a b c : ℝ) : 
  (5 * a^3 - 2019 * a + 4029 = 0) →
  (5 * b^3 - 2019 * b + 4029 = 0) →
  (5 * c^3 - 2019 * c + 4029 = 0) →
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 12087/5 := by
sorry

end cubic_roots_sum_cubes_l1927_192700


namespace min_sum_of_distances_l1927_192766

/-- Represents a five-digit number -/
def FiveDigitNumber := Fin 100000

/-- The distance between two five-digit numbers -/
def distance (a b : FiveDigitNumber) : Nat :=
  sorry

/-- A permutation of all five-digit numbers -/
def Permutation := Equiv.Perm FiveDigitNumber

/-- The sum of distances between consecutive numbers in a permutation -/
def sumOfDistances (p : Permutation) : Nat :=
  sorry

/-- The minimum possible sum of distances between consecutive five-digit numbers -/
theorem min_sum_of_distances :
  ∃ (p : Permutation), sumOfDistances p = 101105 ∧
  ∀ (q : Permutation), sumOfDistances q ≥ 101105 :=
sorry

end min_sum_of_distances_l1927_192766


namespace min_both_composers_l1927_192780

theorem min_both_composers (total : ℕ) (mozart : ℕ) (beethoven : ℕ)
  (h_total : total = 120)
  (h_mozart : mozart = 95)
  (h_beethoven : beethoven = 80)
  : ∃ (both : ℕ), both ≥ mozart + beethoven - total ∧ both = 40 :=
sorry

end min_both_composers_l1927_192780


namespace positive_test_probability_l1927_192748

/-- Probability of a positive test result given the disease prevalence and test characteristics -/
theorem positive_test_probability
  (P_A : ℝ)
  (P_B_given_A : ℝ)
  (P_B_given_not_A : ℝ)
  (h1 : P_A = 0.01)
  (h2 : P_B_given_A = 0.99)
  (h3 : P_B_given_not_A = 0.1)
  (h4 : ∀ (P_A P_B_given_A P_B_given_not_A : ℝ),
    P_A ≥ 0 ∧ P_A ≤ 1 →
    P_B_given_A ≥ 0 ∧ P_B_given_A ≤ 1 →
    P_B_given_not_A ≥ 0 ∧ P_B_given_not_A ≤ 1 →
    P_B_given_A * P_A + P_B_given_not_A * (1 - P_A) ≥ 0 ∧
    P_B_given_A * P_A + P_B_given_not_A * (1 - P_A) ≤ 1) :
  P_B_given_A * P_A + P_B_given_not_A * (1 - P_A) = 0.1089 := by
  sorry


end positive_test_probability_l1927_192748


namespace appliance_sales_prediction_l1927_192730

/-- Represents the sales and cost data for an appliance -/
structure ApplianceData where
  sales : ℕ
  cost : ℕ

/-- Checks if two ApplianceData are inversely proportional -/
def inversely_proportional (a b : ApplianceData) : Prop :=
  a.sales * a.cost = b.sales * b.cost

theorem appliance_sales_prediction
  (blender_initial blender_final microwave_initial microwave_final : ApplianceData)
  (h1 : inversely_proportional blender_initial blender_final)
  (h2 : inversely_proportional microwave_initial microwave_final)
  (h3 : blender_initial.sales = 15)
  (h4 : blender_initial.cost = 300)
  (h5 : blender_final.cost = 450)
  (h6 : microwave_initial.sales = 25)
  (h7 : microwave_initial.cost = 400)
  (h8 : microwave_final.cost = 500) :
  blender_final.sales = 10 ∧ microwave_final.sales = 20 := by
  sorry

end appliance_sales_prediction_l1927_192730


namespace solve_inequality_range_of_a_l1927_192704

-- Define the function f
def f (x a : ℝ) : ℝ := |2*x - a| + |x - 1| - 5

-- Theorem for part 1
theorem solve_inequality (a : ℝ) (ha : a ≠ 0) (h2 : f 2 a = 0) :
  (a = 4 → ∀ x, f x a ≤ 10 ↔ -10/3 ≤ x ∧ x ≤ 20/3) ∧
  (a = -4 → ∀ x, f x a ≤ 10 ↔ -6 ≤ x ∧ x ≤ 4) :=
sorry

-- Theorem for part 2
theorem range_of_a (a : ℝ) (ha : a < 0) 
  (h_triangle : ∃ x₁ x₂ x₃, x₁ < x₂ ∧ x₂ < x₃ ∧ f x₁ a = 0 ∧ f x₂ a < 0 ∧ f x₃ a = 0) :
  -3 ≤ a ∧ a < 0 :=
sorry

end solve_inequality_range_of_a_l1927_192704


namespace quadratic_equation_solution_l1927_192746

theorem quadratic_equation_solution :
  ∀ y : ℂ, 4 + 3 * y^2 = 0.7 * y - 40 ↔ y = (0.1167 : ℝ) + (3.8273 : ℝ) * I ∨ y = (0.1167 : ℝ) - (3.8273 : ℝ) * I :=
by sorry

end quadratic_equation_solution_l1927_192746


namespace maximize_negative_products_l1927_192756

theorem maximize_negative_products (n : ℕ) (h : n > 0) :
  let f : ℕ → ℕ := λ k => k * (n - k)
  let max_k : ℕ := if n % 2 = 0 then n / 2 else (n - 1) / 2
  ∀ k, k ≤ n → f k ≤ f max_k ∧
    (n % 2 ≠ 0 → f k ≤ f ((n + 1) / 2)) :=
by sorry


end maximize_negative_products_l1927_192756


namespace final_sum_theorem_l1927_192773

def num_participants : ℕ := 43

def calculator_operation (n : ℕ) (initial_value : ℤ) : ℤ :=
  match initial_value with
  | 2 => 2^(2^n)
  | 1 => 1
  | -1 => (-1)^n
  | _ => initial_value

theorem final_sum_theorem :
  calculator_operation num_participants 2 +
  calculator_operation num_participants 1 +
  calculator_operation num_participants (-1) = 2^(2^num_participants) := by
  sorry

end final_sum_theorem_l1927_192773


namespace max_integers_satisfying_inequalities_l1927_192754

theorem max_integers_satisfying_inequalities :
  (∃ x : ℕ, x = 7 ∧ 50 * x < 360 ∧ ∀ y : ℕ, 50 * y < 360 → y ≤ x) ∧
  (∃ y : ℕ, y = 4 ∧ 80 * y < 352 ∧ ∀ z : ℕ, 80 * z < 352 → z ≤ y) ∧
  (∃ z : ℕ, z = 6 ∧ 70 * z < 424 ∧ ∀ w : ℕ, 70 * w < 424 → w ≤ z) ∧
  (∃ w : ℕ, w = 4 ∧ 60 * w < 245 ∧ ∀ v : ℕ, 60 * v < 245 → v ≤ w) :=
by sorry

end max_integers_satisfying_inequalities_l1927_192754


namespace well_cared_fish_lifespan_l1927_192787

/-- The average lifespan of a hamster in years -/
def hamster_lifespan : ℝ := 2.5

/-- The lifespan of a dog relative to a hamster -/
def dog_lifespan_factor : ℝ := 4

/-- The additional lifespan of a well-cared fish compared to a dog, in years -/
def fish_extra_lifespan : ℝ := 2

/-- The number of months in a year -/
def months_per_year : ℝ := 12

/-- Theorem: A well-cared fish can live 144 months -/
theorem well_cared_fish_lifespan :
  hamster_lifespan * dog_lifespan_factor * months_per_year + fish_extra_lifespan * months_per_year = 144 :=
by sorry

end well_cared_fish_lifespan_l1927_192787


namespace first_round_score_l1927_192726

def card_values : List ℕ := [2, 4, 7, 13]

theorem first_round_score (total_score : ℕ) (last_round_score : ℕ) 
  (h1 : total_score = 16)
  (h2 : last_round_score = 2)
  (h3 : card_values.sum = 26)
  (h4 : ∃ (n : ℕ), n * card_values.sum = 16 + 17 + 21 + 24)
  : ∃ (first_round_score : ℕ), 
    first_round_score ∈ card_values ∧ 
    ∃ (second_round_score : ℕ), 
      second_round_score ∈ card_values ∧ 
      first_round_score + second_round_score + last_round_score = total_score ∧
      first_round_score = 7 :=
by
  sorry

#check first_round_score

end first_round_score_l1927_192726


namespace yellow_flowers_count_l1927_192734

/-- Represents the number of flowers of each color in the garden -/
structure FlowerCounts where
  total : Nat
  green : Nat
  red : Nat
  blue : Nat
  yellow : Nat

/-- The conditions of the flower garden problem -/
def gardenConditions : FlowerCounts → Prop := fun c =>
  c.total = 96 ∧
  c.green = 9 ∧
  c.red = 3 * c.green ∧
  c.blue = c.total / 2 ∧
  c.yellow = c.total - c.green - c.red - c.blue

/-- Theorem stating that under the given conditions, there are 12 yellow flowers -/
theorem yellow_flowers_count (c : FlowerCounts) : 
  gardenConditions c → c.yellow = 12 := by
  sorry

end yellow_flowers_count_l1927_192734


namespace correct_proposition_l1927_192792

-- Define proposition p
def p : Prop := ∀ x : ℝ, x^2 ≥ 0

-- Define proposition q
def q : Prop := 1 < 0

-- Theorem to prove
theorem correct_proposition : p ∧ ¬q := by
  sorry

end correct_proposition_l1927_192792


namespace min_value_and_angle_l1927_192749

theorem min_value_and_angle (A : Real) : 
  let f := fun A => 2 * Real.sin (A / 2) - Real.cos (A / 2)
  ∃ (min_value : Real) (min_angle : Real),
    (∀ A, f A ≥ min_value) ∧
    (f min_angle = min_value) ∧
    (min_value = -1) ∧
    (min_angle = 270 * π / 180) := by
  sorry

end min_value_and_angle_l1927_192749


namespace weekend_newspaper_delivery_l1927_192762

/-- The total number of newspapers delivered on the weekend -/
def total_newspapers (saturday_papers sunday_papers : ℕ) : ℕ :=
  saturday_papers + sunday_papers

/-- Theorem: The total number of newspapers delivered on the weekend is 110 -/
theorem weekend_newspaper_delivery : total_newspapers 45 65 = 110 := by
  sorry

end weekend_newspaper_delivery_l1927_192762


namespace square_of_five_times_sqrt_three_l1927_192758

theorem square_of_five_times_sqrt_three : (5 * Real.sqrt 3) ^ 2 = 75 := by
  sorry

end square_of_five_times_sqrt_three_l1927_192758


namespace kennel_long_furred_dogs_l1927_192703

/-- Represents the number of dogs with a certain property in a kennel -/
structure DogCount where
  total : ℕ
  brown : ℕ
  neither_long_nor_brown : ℕ

/-- Calculates the number of long-furred dogs in the kennel -/
def long_furred_dogs (d : DogCount) : ℕ :=
  d.total - d.neither_long_nor_brown - d.brown

/-- Theorem stating that in a kennel with the given properties, there are 10 long-furred dogs -/
theorem kennel_long_furred_dogs :
  let d : DogCount := ⟨45, 27, 8⟩
  long_furred_dogs d = 10 := by
  sorry

#eval long_furred_dogs ⟨45, 27, 8⟩

end kennel_long_furred_dogs_l1927_192703


namespace regular_time_limit_proof_l1927_192721

/-- Represents the regular time limit in hours -/
def regular_time_limit : ℕ := 40

/-- Regular pay rate in dollars per hour -/
def regular_pay_rate : ℕ := 3

/-- Overtime pay rate in dollars per hour -/
def overtime_pay_rate : ℕ := 2 * regular_pay_rate

/-- Total pay received in dollars -/
def total_pay : ℕ := 192

/-- Overtime hours worked -/
def overtime_hours : ℕ := 12

theorem regular_time_limit_proof :
  regular_time_limit * regular_pay_rate + overtime_hours * overtime_pay_rate = total_pay :=
by sorry

end regular_time_limit_proof_l1927_192721


namespace hamburgers_served_l1927_192795

/-- Given a restaurant that made a certain number of hamburgers and had some left over,
    calculate the number of hamburgers served. -/
theorem hamburgers_served (total : ℕ) (leftover : ℕ) (h1 : total = 9) (h2 : leftover = 6) :
  total - leftover = 3 := by
  sorry

end hamburgers_served_l1927_192795


namespace stock_price_calculation_l1927_192743

/-- Calculates the price of a stock given investment details -/
theorem stock_price_calculation 
  (investment : ℝ) 
  (dividend_rate : ℝ) 
  (annual_income : ℝ) 
  (face_value : ℝ) 
  (h1 : investment = 6800)
  (h2 : dividend_rate = 0.20)
  (h3 : annual_income = 1000)
  (h4 : face_value = 100) : 
  (investment / (annual_income / dividend_rate)) * face_value = 136 := by
  sorry

end stock_price_calculation_l1927_192743


namespace distributions_five_balls_four_boxes_l1927_192738

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes -/
def totalDistributions (n k : ℕ) : ℕ := k^n

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes,
    with the condition that one specific box must contain at least one ball -/
def distributionsWithConstraint (n k : ℕ) : ℕ :=
  totalDistributions n k - totalDistributions n (k - 1)

theorem distributions_five_balls_four_boxes :
  distributionsWithConstraint 5 4 = 781 := by
  sorry

end distributions_five_balls_four_boxes_l1927_192738


namespace tiling_impossible_l1927_192797

/-- Represents a 1 × 3 strip used for tiling -/
structure Strip :=
  (length : Nat)
  (width : Nat)
  (h_length : length = 3)
  (h_width : width = 1)

/-- Represents the figure to be tiled -/
structure Figure :=
  (total_squares : Nat)
  (color1_squares : Nat)
  (color2_squares : Nat)
  (h_total : total_squares = color1_squares + color2_squares)
  (h_color1 : color1_squares = 7)
  (h_color2 : color2_squares = 8)

/-- Represents a tiling of the figure with strips -/
structure Tiling :=
  (figure : Figure)
  (strips : List Strip)
  (h_cover : ∀ s ∈ strips, s.length = 3 ∧ s.width = 1)
  (h_no_overlap : List.Nodup strips)
  (h_complete : strips.length * 3 = figure.total_squares)

/-- The main theorem stating that tiling is impossible -/
theorem tiling_impossible (f : Figure) : ¬ ∃ t : Tiling, t.figure = f := by
  sorry

end tiling_impossible_l1927_192797


namespace fraction_to_decimal_l1927_192742

theorem fraction_to_decimal : 19 / (2^2 * 5^3) = 0.095 := by
  sorry

end fraction_to_decimal_l1927_192742


namespace equation_positive_root_l1927_192731

theorem equation_positive_root (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (x + a) / (x + 3) - 2 / (x + 3) = 0) → a = 5 := by
  sorry

end equation_positive_root_l1927_192731


namespace smallest_three_digit_geometric_sequence_l1927_192796

-- Define a function to check if a number is a three-digit integer
def isThreeDigitInteger (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Define a function to check if digits are distinct
def hasDistinctDigits (n : ℕ) : Prop :=
  let digits := [n / 100, (n / 10) % 10, n % 10]
  digits.length = digits.toFinset.card

-- Define a function to check if digits form a geometric sequence
def formsGeometricSequence (n : ℕ) : Prop :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  ∃ (r : ℚ), r ≠ 0 ∧ b = a * r ∧ c = b * r

-- State the theorem
theorem smallest_three_digit_geometric_sequence :
  ∀ n : ℕ, isThreeDigitInteger n ∧ hasDistinctDigits n ∧ formsGeometricSequence n →
  124 ≤ n :=
sorry

end smallest_three_digit_geometric_sequence_l1927_192796


namespace greatest_value_quadratic_inequality_l1927_192765

theorem greatest_value_quadratic_inequality :
  ∃ (a_max : ℝ), a_max = 8 ∧
  (∀ a : ℝ, a^2 - 12*a + 32 ≤ 0 → a ≤ a_max) ∧
  (a_max^2 - 12*a_max + 32 ≤ 0) :=
sorry

end greatest_value_quadratic_inequality_l1927_192765


namespace students_walking_home_l1927_192794

theorem students_walking_home (total : ℚ) 
  (bus : ℚ) (auto : ℚ) (bike : ℚ) (metro : ℚ) :
  bus = 1/3 →
  auto = 1/5 →
  bike = 1/8 →
  metro = 1/15 →
  total = 1 →
  total - (bus + auto + bike + metro) = 11/40 := by
sorry

end students_walking_home_l1927_192794


namespace outfit_combinations_l1927_192799

/-- Represents the number of shirts -/
def num_shirts : ℕ := 8

/-- Represents the number of pants -/
def num_pants : ℕ := 6

/-- Represents the number of hats -/
def num_hats : ℕ := 6

/-- Represents the number of distinct colors -/
def num_colors : ℕ := 6

/-- Calculates the number of outfit combinations where no two items are the same color -/
def valid_outfits : ℕ := 174

/-- Theorem stating that the number of valid outfit combinations is 174 -/
theorem outfit_combinations :
  (num_shirts * num_pants * num_hats) -
  (num_colors * num_hats + num_colors * num_shirts + num_colors * num_pants - num_colors) =
  valid_outfits :=
by sorry

end outfit_combinations_l1927_192799


namespace evaluate_expression_l1927_192733

theorem evaluate_expression : 3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = 5999 := by
  sorry

end evaluate_expression_l1927_192733


namespace base6_addition_problem_l1927_192751

/-- Represents a digit in base 6 -/
def Base6Digit := Fin 6

/-- Checks if three Base6Digits are distinct -/
def are_distinct (s h e : Base6Digit) : Prop :=
  s ≠ h ∧ s ≠ e ∧ h ≠ e

/-- Converts a natural number to its base 6 representation -/
def to_base6 (n : ℕ) : ℕ :=
  sorry

/-- Adds two base 6 numbers -/
def base6_add (a b : ℕ) : ℕ :=
  sorry

/-- The main theorem -/
theorem base6_addition_problem :
  ∃ (s h e : Base6Digit),
    are_distinct s h e ∧
    0 < s.val ∧ 0 < h.val ∧ 0 < e.val ∧
    base6_add (s.val * 36 + h.val * 6 + e.val) (e.val * 36 + s.val * 6 + h.val) = s.val * 36 + h.val * 6 + s.val ∧
    s.val = 4 ∧ h.val = 2 ∧ e.val = 3 ∧
    to_base6 (s.val + h.val + e.val) = 13 :=
  sorry

end base6_addition_problem_l1927_192751


namespace sufficient_not_necessary_l1927_192789

/-- Two lines in the xy-plane --/
structure TwoLines :=
  (a : ℝ)

/-- The condition for two lines to be parallel --/
def are_parallel (lines : TwoLines) : Prop :=
  lines.a^2 - lines.a = 2

/-- The statement that a=2 is sufficient but not necessary for the lines to be parallel --/
theorem sufficient_not_necessary :
  (∃ (lines : TwoLines), lines.a = 2 → are_parallel lines) ∧
  (∃ (lines : TwoLines), are_parallel lines ∧ lines.a ≠ 2) :=
sorry

end sufficient_not_necessary_l1927_192789


namespace total_lemon_heads_eaten_l1927_192723

/-- The number of Lemon Heads in each package -/
def lemon_heads_per_package : ℕ := 6

/-- The number of whole boxes Louis finished -/
def boxes_finished : ℕ := 9

/-- Theorem: Given the conditions, Louis ate 54 Lemon Heads in total -/
theorem total_lemon_heads_eaten :
  lemon_heads_per_package * boxes_finished = 54 := by
  sorry

end total_lemon_heads_eaten_l1927_192723


namespace steak_per_member_l1927_192716

theorem steak_per_member (family_members : ℕ) (steak_size : ℕ) (steaks_needed : ℕ) :
  family_members = 5 →
  steak_size = 20 →
  steaks_needed = 4 →
  (steaks_needed * steak_size) / family_members = 16 := by
sorry

end steak_per_member_l1927_192716


namespace tinas_trip_distance_l1927_192715

/-- Tina's trip consists of three parts: highway, city, and rural roads. 
    This theorem proves that the total distance of her trip is 120 miles. -/
theorem tinas_trip_distance : ℝ → Prop :=
  fun total_distance =>
    (total_distance / 2 + 30 + total_distance / 4 = total_distance) →
    total_distance = 120

/-- Proof of the theorem -/
lemma prove_tinas_trip_distance : tinas_trip_distance 120 := by
  sorry

end tinas_trip_distance_l1927_192715


namespace competition_scores_l1927_192747

theorem competition_scores (n k : ℕ) : n ≥ 2 ∧ k ≥ 1 →
  (k * n * (n + 1) = 52 * n * (n - 1)) ↔ 
  ((n = 25 ∧ k = 2) ∨ (n = 12 ∧ k = 4) ∨ (n = 3 ∧ k = 13)) := by
sorry

end competition_scores_l1927_192747


namespace second_polygon_sides_l1927_192798

theorem second_polygon_sides (n : ℕ) (s : ℝ) : 
  s > 0 → 
  50 * (3 * s) = n * s → 
  n = 150 := by
sorry

end second_polygon_sides_l1927_192798


namespace cindy_calculation_l1927_192793

theorem cindy_calculation (x : ℝ) (h : (x + 7) * 5 = 260) : 5 * x + 7 = 232 := by
  sorry

end cindy_calculation_l1927_192793


namespace coin_distribution_theorem_l1927_192713

/-- Represents the coin distribution between Pete and Paul -/
def coin_distribution (x : ℕ) : Prop :=
  -- Paul's final coin count
  let paul_coins := x
  -- Pete's coin count using the sum formula
  let pete_coins := x * (x + 1) / 2
  -- The condition that Pete has 5 times as many coins as Paul
  pete_coins = 5 * paul_coins

/-- The total number of coins distributed -/
def total_coins (x : ℕ) : ℕ := 6 * x

theorem coin_distribution_theorem :
  ∃ x : ℕ, coin_distribution x ∧ total_coins x = 54 := by
  sorry

end coin_distribution_theorem_l1927_192713


namespace lemonade_intermission_l1927_192771

theorem lemonade_intermission (total : ℝ) (first : ℝ) (third : ℝ) (second : ℝ)
  (h_total : total = 0.92)
  (h_first : first = 0.25)
  (h_third : third = 0.25)
  (h_sum : total = first + second + third) :
  second = 0.42 := by
sorry

end lemonade_intermission_l1927_192771


namespace digit_equation_solution_l1927_192763

theorem digit_equation_solution : ∃! (Θ : ℕ), Θ > 0 ∧ Θ < 10 ∧ (476 : ℚ) / Θ = 50 + 4 * Θ :=
by sorry

end digit_equation_solution_l1927_192763


namespace quadratic_roots_sum_squares_l1927_192745

theorem quadratic_roots_sum_squares (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, 
    (8 * x₁^2 + 2 * k * x₁ + k - 1 = 0) ∧ 
    (8 * x₂^2 + 2 * k * x₂ + k - 1 = 0) ∧ 
    (x₁^2 + x₂^2 = 1) ∧
    (4 * k^2 - 32 * (k - 1) ≥ 0)) →
  k = -2 :=
by sorry

end quadratic_roots_sum_squares_l1927_192745


namespace katherine_bottle_caps_l1927_192761

def initial_bottle_caps : ℕ := 34
def eaten_bottle_caps : ℕ := 8
def remaining_bottle_caps : ℕ := 26

theorem katherine_bottle_caps :
  initial_bottle_caps = eaten_bottle_caps + remaining_bottle_caps :=
by sorry

end katherine_bottle_caps_l1927_192761


namespace system_solutions_l1927_192764

/-- The system of equations -/
def system (p x y : ℝ) : Prop :=
  p * (x^2 - y^2) = (p^2 - 1) * x * y ∧ |x - 1| + |y| = 1

/-- The system has at least three different real solutions -/
def has_three_solutions (p : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ), 
    system p x₁ y₁ ∧ 
    system p x₂ y₂ ∧ 
    system p x₃ y₃ ∧ 
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧ 
    (x₁ ≠ x₃ ∨ y₁ ≠ y₃) ∧ 
    (x₂ ≠ x₃ ∨ y₂ ≠ y₃)

/-- The main theorem -/
theorem system_solutions :
  ∀ p : ℝ, has_three_solutions p ↔ p = 1 ∨ p = -1 :=
sorry

end system_solutions_l1927_192764


namespace a_range_l1927_192740

/-- The function f(x) = x^2 - 2ax + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 3

/-- f is increasing on [1, +∞) -/
def f_increasing (a : ℝ) : Prop :=
  ∀ x y, 1 ≤ x ∧ x < y → f a x < f a y

theorem a_range (a : ℝ) (h : f_increasing a) : a ≤ 1 := by
  sorry

end a_range_l1927_192740


namespace sum_double_factorial_divisible_l1927_192774

def double_factorial (n : ℕ) : ℕ :=
  if n ≤ 1 then 1 else n * double_factorial (n - 2)

theorem sum_double_factorial_divisible :
  (double_factorial 1985 + double_factorial 1986) % 1987 = 0 := by
sorry

end sum_double_factorial_divisible_l1927_192774


namespace absolute_value_inequality_l1927_192760

theorem absolute_value_inequality (x : ℝ) :
  |x - 1| + |x + 2| ≥ 5 ↔ x ≤ -3 ∨ x ≥ 2 := by sorry

end absolute_value_inequality_l1927_192760


namespace exists_unsolvable_chessboard_l1927_192744

/-- Represents a 12x12 chessboard where each square can be black or white -/
def Chessboard := Fin 12 → Fin 12 → Bool

/-- Represents a row or column flip operation -/
inductive FlipOperation
| row (i : Fin 12)
| col (j : Fin 12)

/-- Applies a flip operation to a chessboard -/
def applyFlip (board : Chessboard) (op : FlipOperation) : Chessboard :=
  match op with
  | FlipOperation.row i => fun x y => if x = i then !board x y else board x y
  | FlipOperation.col j => fun x y => if y = j then !board x y else board x y

/-- Checks if all squares on the board are black -/
def allBlack (board : Chessboard) : Prop :=
  ∀ i j, board i j = true

/-- Theorem: There exists an initial chessboard configuration that cannot be made all black -/
theorem exists_unsolvable_chessboard : 
  ∃ (initial : Chessboard), ¬∃ (ops : List FlipOperation), allBlack (ops.foldl applyFlip initial) :=
sorry

end exists_unsolvable_chessboard_l1927_192744


namespace consecutive_integers_sum_of_squares_and_cubes_l1927_192702

theorem consecutive_integers_sum_of_squares_and_cubes :
  ∀ n : ℤ,
  (n - 1)^2 + n^2 + (n + 1)^2 = 8450 →
  n = 53 ∧ (n - 1)^3 + n^3 + (n + 1)^3 = 446949 := by
  sorry

end consecutive_integers_sum_of_squares_and_cubes_l1927_192702


namespace sum_of_digits_of_b_is_nine_l1927_192750

/-- A function that returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number has exactly 1995 digits -/
def has1995Digits (n : ℕ) : Prop := sorry

theorem sum_of_digits_of_b_is_nine (N : ℕ) 
  (h1 : has1995Digits N) 
  (h2 : N % 9 = 0) : 
  let a := sumOfDigits N
  let b := sumOfDigits a
  sumOfDigits b = 9 := by sorry

end sum_of_digits_of_b_is_nine_l1927_192750


namespace money_problem_l1927_192711

theorem money_problem (c d : ℝ) (h1 : 7 * c + d > 84) (h2 : 5 * c - d = 35) :
  c > 9.92 ∧ d > 14.58 := by
sorry

end money_problem_l1927_192711


namespace arithmetic_mean_4_16_l1927_192755

theorem arithmetic_mean_4_16 (x : ℝ) : x = (4 + 16) / 2 → x = 10 := by sorry

end arithmetic_mean_4_16_l1927_192755


namespace sallys_class_size_l1927_192776

theorem sallys_class_size (total_pens : ℕ) (pens_per_student : ℕ) (pens_home : ℕ) :
  total_pens = 342 →
  pens_per_student = 7 →
  pens_home = 17 →
  ∃ (num_students : ℕ),
    num_students * pens_per_student + 2 * pens_home + (total_pens - num_students * pens_per_student) / 2 = total_pens ∧
    num_students = 44 :=
by sorry

end sallys_class_size_l1927_192776


namespace complement_of_intersection_l1927_192770

def U : Set Nat := {1, 2, 3, 4}
def M : Set Nat := {1, 2, 3}
def N : Set Nat := {1, 3, 4}

theorem complement_of_intersection (h : U = {1, 2, 3, 4} ∧ M = {1, 2, 3} ∧ N = {1, 3, 4}) :
  (M ∩ N)ᶜ = {2, 4} := by
  sorry

end complement_of_intersection_l1927_192770


namespace choir_members_count_l1927_192717

theorem choir_members_count :
  ∃! (s : Finset ℕ), 
    (∀ n ∈ s, 100 ≤ n ∧ n ≤ 200) ∧
    (∀ n ∈ s, (n + 3) % 7 = 0 ∧ (n + 5) % 8 = 0) ∧
    s.card = 2 ∧
    123 ∈ s ∧ 179 ∈ s :=
by sorry

end choir_members_count_l1927_192717


namespace quadrilateral_is_trapezium_l1927_192790

/-- A quadrilateral with angles x°, 5x°, 2x°, and 4x° is a trapezium -/
theorem quadrilateral_is_trapezium (x : ℝ) 
  (angle_sum : x + 5*x + 2*x + 4*x = 360) : 
  ∃ (a b c d : ℝ), 
    a + b + c + d = 360 ∧ 
    a + c = 180 ∧
    (a = x ∨ a = 5*x ∨ a = 2*x ∨ a = 4*x) ∧
    (b = x ∨ b = 5*x ∨ b = 2*x ∨ b = 4*x) ∧
    (c = x ∨ c = 5*x ∨ c = 2*x ∨ c = 4*x) ∧
    (d = x ∨ d = 5*x ∨ d = 2*x ∨ d = 4*x) ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d :=
by sorry

end quadrilateral_is_trapezium_l1927_192790


namespace largest_operation_l1927_192710

theorem largest_operation : ∀ a b c d e : ℝ,
  a = 15432 + 1 / 3241 →
  b = 15432 - 1 / 3241 →
  c = 15432 * (1 / 3241) →
  d = 15432 / (1 / 3241) →
  e = 15432.3241 →
  d > a ∧ d > b ∧ d > c ∧ d > e := by
  sorry

end largest_operation_l1927_192710


namespace selection_theorem_l1927_192732

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of boys in the group -/
def num_boys : ℕ := 4

/-- The number of girls in the group -/
def num_girls : ℕ := 3

/-- The total number of people in the group -/
def total_people : ℕ := num_boys + num_girls

/-- The number of representatives to be selected -/
def num_representatives : ℕ := 3

theorem selection_theorem :
  (choose total_people num_representatives = 35) ∧
  (choose num_girls 1 * choose num_boys 2 +
   choose num_girls 2 * choose num_boys 1 +
   choose num_girls 3 = 31) ∧
  (choose total_people num_representatives -
   choose num_boys num_representatives -
   choose num_girls num_representatives = 30) := by
  sorry

end selection_theorem_l1927_192732


namespace trig_expression_equals_one_l1927_192777

theorem trig_expression_equals_one : 
  (1 - 1 / Real.cos (30 * π / 180)) * 
  (1 + 1 / Real.sin (60 * π / 180)) * 
  (1 - 1 / Real.sin (30 * π / 180)) * 
  (1 + 1 / Real.cos (60 * π / 180)) = 1 := by
  sorry

end trig_expression_equals_one_l1927_192777


namespace rugby_league_matches_l1927_192705

/-- The number of matches played in a rugby league -/
def total_matches (n : ℕ) (k : ℕ) : ℕ :=
  k * (n.choose 2)

/-- Theorem: In a league with 10 teams, where each team plays against every other team exactly 4 times, the total number of matches played is 180. -/
theorem rugby_league_matches :
  total_matches 10 4 = 180 := by
  sorry

end rugby_league_matches_l1927_192705


namespace solution_set_f_less_than_3_range_of_a_empty_solution_l1927_192753

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |2*x + 2|

-- Theorem for the solution set of f(x) < 3
theorem solution_set_f_less_than_3 :
  {x : ℝ | f x < 3} = {x : ℝ | -4/3 < x ∧ x < 0} := by sorry

-- Theorem for the range of a when f(x) < a has no solutions
theorem range_of_a_empty_solution :
  {a : ℝ | ∀ x, f x ≥ a} = {a : ℝ | a ≤ 2} := by sorry

end solution_set_f_less_than_3_range_of_a_empty_solution_l1927_192753


namespace sequence_general_term_l1927_192757

/-- Given a sequence {a_n} where S_n is the sum of the first n terms -/
def S (n : ℕ+) : ℚ := 3 * n.val ^ 2 + 8 * n.val

/-- The general term of the sequence -/
def a (n : ℕ+) : ℚ := 6 * n.val + 5

/-- Theorem stating that the given general term formula is correct for the sequence -/
theorem sequence_general_term (n : ℕ+) : a n = S n - S (n - 1) := by sorry

end sequence_general_term_l1927_192757


namespace season_length_l1927_192785

def games_per_month : ℕ := 7
def games_in_season : ℕ := 14

theorem season_length :
  games_in_season / games_per_month = 2 :=
sorry

end season_length_l1927_192785


namespace angle_measure_in_special_quadrilateral_l1927_192712

/-- In a quadrilateral EFGH where ∠E = 3∠F = 4∠G = 6∠H, the measure of ∠E is 360 * (4/7) degrees. -/
theorem angle_measure_in_special_quadrilateral :
  ∀ (E F G H : ℝ),
  E + F + G + H = 360 →
  E = 3 * F →
  E = 4 * G →
  E = 6 * H →
  E = 360 * (4/7) := by
sorry

end angle_measure_in_special_quadrilateral_l1927_192712


namespace unique_grid_solution_l1927_192783

/-- Represents a 3x3 grid with some fixed values and variables A, B, C, D -/
structure Grid :=
  (A B C D : ℕ)

/-- Checks if two numbers are adjacent in the grid -/
def adjacent (x y : ℕ) : Prop :=
  (x = 1 ∧ y = 2) ∨ (x = 1 ∧ y = 4) ∨ (x = 2 ∧ y = 3) ∨ (x = 2 ∧ y = 5) ∨
  (x = 3 ∧ y = 6) ∨ (x = 4 ∧ y = 5) ∨ (x = 4 ∧ y = 7) ∨ (x = 5 ∧ y = 6) ∨
  (x = 5 ∧ y = 8) ∨ (x = 6 ∧ y = 9) ∨ (x = 7 ∧ y = 8) ∨ (x = 8 ∧ y = 9) ∨
  (y = 1 ∧ x = 2) ∨ (y = 1 ∧ x = 4) ∨ (y = 2 ∧ x = 3) ∨ (y = 2 ∧ x = 5) ∨
  (y = 3 ∧ x = 6) ∨ (y = 4 ∧ x = 5) ∨ (y = 4 ∧ x = 7) ∨ (y = 5 ∧ x = 6) ∨
  (y = 5 ∧ x = 8) ∨ (y = 6 ∧ x = 9) ∨ (y = 7 ∧ x = 8) ∨ (y = 8 ∧ x = 9)

/-- The main theorem to prove -/
theorem unique_grid_solution :
  ∀ (g : Grid),
    (g.A ≠ 1 ∧ g.A ≠ 3 ∧ g.A ≠ 5 ∧ g.A ≠ 7 ∧ g.A ≠ 9) →
    (g.B ≠ 1 ∧ g.B ≠ 3 ∧ g.B ≠ 5 ∧ g.B ≠ 7 ∧ g.B ≠ 9) →
    (g.C ≠ 1 ∧ g.C ≠ 3 ∧ g.C ≠ 5 ∧ g.C ≠ 7 ∧ g.C ≠ 9) →
    (g.D ≠ 1 ∧ g.D ≠ 3 ∧ g.D ≠ 5 ∧ g.D ≠ 7 ∧ g.D ≠ 9) →
    (∀ (x y : ℕ), adjacent x y → x + y < 12) →
    (g.A = 8 ∧ g.B = 6 ∧ g.C = 4 ∧ g.D = 2) :=
by sorry


end unique_grid_solution_l1927_192783


namespace oviparous_produces_significant_differences_l1927_192739

-- Define the modes of reproduction
inductive ReproductionMode
  | Vegetative
  | Oviparous
  | Fission
  | Budding

-- Define the reproduction categories
inductive ReproductionCategory
  | Sexual
  | Asexual

-- Define a function to categorize reproduction modes
def categorizeReproduction : ReproductionMode → ReproductionCategory
  | ReproductionMode.Vegetative => ReproductionCategory.Asexual
  | ReproductionMode.Oviparous => ReproductionCategory.Sexual
  | ReproductionMode.Fission => ReproductionCategory.Asexual
  | ReproductionMode.Budding => ReproductionCategory.Asexual

-- Define a property for producing offspring with significant differences
def produceSignificantDifferences (mode : ReproductionMode) : Prop :=
  categorizeReproduction mode = ReproductionCategory.Sexual

theorem oviparous_produces_significant_differences :
  ∀ (mode : ReproductionMode),
    produceSignificantDifferences mode ↔ mode = ReproductionMode.Oviparous :=
by sorry

end oviparous_produces_significant_differences_l1927_192739


namespace heating_pad_cost_per_use_l1927_192718

/-- Calculates the cost per use of a heating pad. -/
def cost_per_use (total_cost : ℚ) (uses_per_week : ℕ) (num_weeks : ℕ) : ℚ :=
  total_cost / (uses_per_week * num_weeks)

/-- Theorem stating that a $30 heating pad used 3 times a week for 2 weeks costs $5 per use. -/
theorem heating_pad_cost_per_use :
  cost_per_use 30 3 2 = 5 := by
  sorry

end heating_pad_cost_per_use_l1927_192718


namespace betty_picked_15_oranges_l1927_192725

def orange_problem (betty_oranges : ℕ) : Prop :=
  let bill_oranges : ℕ := 12
  let frank_oranges : ℕ := 3 * (betty_oranges + bill_oranges)
  let seeds_planted : ℕ := 2 * frank_oranges
  let trees_grown : ℕ := seeds_planted
  let oranges_per_tree : ℕ := 5
  let total_oranges : ℕ := trees_grown * oranges_per_tree
  total_oranges = 810

theorem betty_picked_15_oranges :
  ∃ (betty_oranges : ℕ), orange_problem betty_oranges ∧ betty_oranges = 15 :=
by sorry

end betty_picked_15_oranges_l1927_192725


namespace bouncing_ball_distance_l1927_192741

/-- Calculates the total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (reboundRatio : ℝ) (bounces : ℕ) : ℝ :=
  sorry

/-- The problem statement -/
theorem bouncing_ball_distance :
  totalDistance 200 (2/3) 4 = 4200 :=
sorry

end bouncing_ball_distance_l1927_192741


namespace fraction_equalities_l1927_192768

theorem fraction_equalities : 
  (126 : ℚ) / 84 = 21 / 18 ∧ (268 : ℚ) / 335 = 4 / 5 := by sorry

end fraction_equalities_l1927_192768


namespace book_env_intersection_l1927_192735

/-- The number of people participating in both Book Club and Environmental Theme Painting --/
def intersection_book_env (total participants : ℕ) 
  (book_club fun_sports env_painting : ℕ) 
  (book_fun fun_env : ℕ) : ℕ :=
  book_club + fun_sports + env_painting - total - book_fun - fun_env

/-- Theorem stating the number of people participating in both Book Club and Environmental Theme Painting --/
theorem book_env_intersection : 
  ∀ (total participants : ℕ) 
    (book_club fun_sports env_painting : ℕ) 
    (book_fun fun_env : ℕ),
  total = 120 →
  book_club = 80 →
  fun_sports = 50 →
  env_painting = 40 →
  book_fun = 20 →
  fun_env = 10 →
  intersection_book_env total participants book_club fun_sports env_painting book_fun fun_env = 20 := by
  sorry

#check book_env_intersection

end book_env_intersection_l1927_192735


namespace diagonal_not_parallel_to_sides_l1927_192736

theorem diagonal_not_parallel_to_sides (n : ℕ) (h : n > 0) :
  n * (2 * n - 3) > 2 * n * (n - 2) := by
  sorry

end diagonal_not_parallel_to_sides_l1927_192736


namespace other_number_is_twenty_l1927_192772

theorem other_number_is_twenty (a b : ℤ) : 
  3 * a + 4 * b = 140 → (a = 20 ∨ b = 20) → (a = 20 ∧ b = 20) :=
by sorry

end other_number_is_twenty_l1927_192772


namespace class_1_wins_l1927_192778

/-- Represents the movements of the marker in a tug-of-war contest -/
def marker_movements : List ℝ := [-0.2, 0.5, -0.8, 1.4, 1.3]

/-- The winning distance in meters -/
def winning_distance : ℝ := 2

/-- Theorem stating that the sum of marker movements is at least the winning distance -/
theorem class_1_wins (movements : List ℝ := marker_movements) 
  (win_dist : ℝ := winning_distance) : 
  movements.sum ≥ win_dist := by sorry

end class_1_wins_l1927_192778


namespace extreme_value_implies_a_equals_negative_one_l1927_192707

/-- Given a function f(x) = x^3 - ax^2 - x + a, where a is a real number,
    if f(x) has an extreme value at x = -1, then a = -1 -/
theorem extreme_value_implies_a_equals_negative_one (a : ℝ) :
  let f := λ x : ℝ => x^3 - a*x^2 - x + a
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-1 - ε) (-1 + ε), f (-1) ≥ f x) ∨
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-1 - ε) (-1 + ε), f (-1) ≤ f x) →
  a = -1 :=
by sorry

end extreme_value_implies_a_equals_negative_one_l1927_192707


namespace justin_reading_requirement_l1927_192779

/-- The number of pages Justin needs to read in one week to pass his class -/
def pages_to_pass : ℕ := 130

/-- The number of pages Justin reads on the first day -/
def first_day_pages : ℕ := 10

/-- The number of remaining days in the week -/
def remaining_days : ℕ := 6

/-- The number of pages Justin reads on each of the remaining days -/
def remaining_day_pages : ℕ := 2 * first_day_pages

theorem justin_reading_requirement :
  first_day_pages + remaining_days * remaining_day_pages = pages_to_pass := by
  sorry

end justin_reading_requirement_l1927_192779


namespace fraction_sum_proof_l1927_192782

theorem fraction_sum_proof (x A B : ℚ) : 
  (5*x - 11) / (2*x^2 + x - 6) = A / (x + 2) + B / (2*x - 3) → 
  A = 3 ∧ B = -1 := by
sorry

end fraction_sum_proof_l1927_192782


namespace group_size_calculation_l1927_192752

theorem group_size_calculation (n : ℕ) 
  (h1 : n * (40 - 3) = n * 40 - 40 + 10) : n = 10 := by
  sorry

#check group_size_calculation

end group_size_calculation_l1927_192752


namespace complexity_power_of_two_no_complexity_less_than_n_l1927_192728

-- Define complexity of an integer
def complexity (n : ℕ) : ℕ := sorry

-- Theorem for part (a)
theorem complexity_power_of_two (k : ℕ) :
  ∀ m : ℕ, 2^k ≤ m → m < 2^(k+1) → complexity m ≤ k := by sorry

-- Theorem for part (b)
theorem no_complexity_less_than_n :
  ∀ n : ℕ, n > 1 → ∃ m : ℕ, n ≤ m → m < 2*n → complexity m ≥ complexity n := by sorry

end complexity_power_of_two_no_complexity_less_than_n_l1927_192728


namespace gcd_1729_867_l1927_192781

theorem gcd_1729_867 : Nat.gcd 1729 867 = 1 := by
  sorry

end gcd_1729_867_l1927_192781
