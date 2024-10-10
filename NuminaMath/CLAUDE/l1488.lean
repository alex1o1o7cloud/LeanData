import Mathlib

namespace set_intersection_problem_l1488_148812

theorem set_intersection_problem (M N : Set ℤ) (a : ℤ) 
  (hM : M = {a, 0})
  (hN : N = {1, 2})
  (hIntersection : M ∩ N = {1}) :
  a = 1 := by
  sorry

end set_intersection_problem_l1488_148812


namespace runner_lap_time_l1488_148878

/-- Proves that given a 400-meter track, a runner completing 3 laps with the first lap in 70 seconds
    and an average speed of 5 m/s for the entire run, the time for each of the second and third laps
    is 85 seconds. -/
theorem runner_lap_time (track_length : ℝ) (num_laps : ℕ) (first_lap_time : ℝ) (avg_speed : ℝ) :
  track_length = 400 →
  num_laps = 3 →
  first_lap_time = 70 →
  avg_speed = 5 →
  ∃ (second_third_lap_time : ℝ),
    second_third_lap_time = 85 ∧
    (track_length * num_laps) / avg_speed = first_lap_time + 2 * second_third_lap_time :=
by sorry

end runner_lap_time_l1488_148878


namespace saving_fraction_is_one_fourth_l1488_148868

/-- Represents the worker's monthly savings behavior -/
structure WorkerSavings where
  monthlyPay : ℝ
  savingFraction : ℝ
  monthlyPay_pos : 0 < monthlyPay
  savingFraction_range : 0 ≤ savingFraction ∧ savingFraction ≤ 1

/-- The theorem stating that the saving fraction is 1/4 given the conditions -/
theorem saving_fraction_is_one_fourth (w : WorkerSavings) 
  (h : 12 * w.savingFraction * w.monthlyPay = 
       4 * (1 - w.savingFraction) * w.monthlyPay) : 
  w.savingFraction = 1/4 := by
  sorry

end saving_fraction_is_one_fourth_l1488_148868


namespace point_in_second_quadrant_l1488_148861

theorem point_in_second_quadrant (a : ℤ) : 
  (2*a + 1 < 0) ∧ (2 + a > 0) → a = -1 :=
by sorry

end point_in_second_quadrant_l1488_148861


namespace no_integer_solutions_l1488_148888

theorem no_integer_solutions : ¬ ∃ (x : ℤ), x^2 - 9*x + 20 < 0 := by
  sorry

end no_integer_solutions_l1488_148888


namespace segment_length_in_dihedral_angle_l1488_148826

/-- Given a segment AB with ends on the faces of a dihedral angle φ, where the distances from A and B
    to the edge of the angle are a and b respectively, and the distance between the projections of A
    and B on the edge is c, the length of AB is equal to √(a² + b² + c² - 2ab cos φ). -/
theorem segment_length_in_dihedral_angle (φ a b c : ℝ) (h_φ : 0 < φ ∧ φ < π) 
    (h_a : a > 0) (h_b : b > 0) (h_c : c > 0) :
  ∃ (AB : ℝ), AB = Real.sqrt (a^2 + b^2 + c^2 - 2 * a * b * Real.cos φ) := by
  sorry

end segment_length_in_dihedral_angle_l1488_148826


namespace train_length_l1488_148849

/-- Given a train with constant speed that crosses a tree in 120 seconds
    and passes a 700m long platform in 190 seconds,
    the length of the train is 1200 meters. -/
theorem train_length (speed : ℝ) (train_length : ℝ) : 
  (train_length / 120 = speed) →
  ((train_length + 700) / 190 = speed) →
  train_length = 1200 := by
  sorry

end train_length_l1488_148849


namespace marbles_fraction_l1488_148892

theorem marbles_fraction (initial_marbles : ℕ) (fraction_taken : ℚ) (cleo_final : ℕ) : 
  initial_marbles = 30 →
  fraction_taken = 3/5 →
  cleo_final = 15 →
  (cleo_final - (fraction_taken * initial_marbles / 2)) / (initial_marbles - fraction_taken * initial_marbles) = 1/2 :=
by sorry

end marbles_fraction_l1488_148892


namespace equation_solution_l1488_148835

theorem equation_solution (x : ℝ) (some_number : ℝ) 
  (h1 : x + 1 = some_number) (h2 : x = 4) : some_number = 5 := by
  sorry

end equation_solution_l1488_148835


namespace letters_per_large_envelope_l1488_148899

theorem letters_per_large_envelope 
  (total_letters : ℕ) 
  (small_envelope_letters : ℕ) 
  (large_envelopes : ℕ) 
  (h1 : total_letters = 80) 
  (h2 : small_envelope_letters = 20) 
  (h3 : large_envelopes = 30) : 
  (total_letters - small_envelope_letters) / large_envelopes = 2 := by
  sorry

end letters_per_large_envelope_l1488_148899


namespace yacht_distance_squared_l1488_148844

theorem yacht_distance_squared (AB BC : ℝ) (angle_B : ℝ) (AC_squared : ℝ) : 
  AB = 15 → 
  BC = 25 → 
  angle_B = 150 * Real.pi / 180 →
  AC_squared = AB^2 + BC^2 - 2 * AB * BC * Real.cos angle_B →
  AC_squared = 850 - 375 * Real.sqrt 3 := by
  sorry

end yacht_distance_squared_l1488_148844


namespace triangle_reconstruction_uniqueness_l1488_148898

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a point is the incenter of a triangle -/
def is_incenter (I : Point) (t : Triangle) : Prop := sorry

/-- Checks if a point is the foot of the altitude from C to AB -/
def is_altitude_foot (H : Point) (t : Triangle) : Prop := sorry

/-- Checks if a point is the excenter opposite to C -/
def is_excenter_C (I_C : Point) (t : Triangle) : Prop := sorry

/-- Checks if the excenter touches side AB and extensions of AC and BC -/
def excenter_touches_sides (I_C : Point) (t : Triangle) : Prop := sorry

theorem triangle_reconstruction_uniqueness 
  (I H I_C : Point) : 
  ∃! t : Triangle, 
    is_incenter I t ∧ 
    is_altitude_foot H t ∧ 
    is_excenter_C I_C t ∧ 
    excenter_touches_sides I_C t :=
sorry

end triangle_reconstruction_uniqueness_l1488_148898


namespace infinite_geometric_series_ratio_l1488_148864

/-- 
For an infinite geometric series with first term a and sum S,
prove that if a = 500 and S = 3500, then the common ratio r is 6/7.
-/
theorem infinite_geometric_series_ratio 
  (a S : ℝ) 
  (h_a : a = 500) 
  (h_S : S = 3500) 
  (h_sum : S = a / (1 - r)) 
  (h_conv : |r| < 1) : 
  r = 6/7 := by
sorry

end infinite_geometric_series_ratio_l1488_148864


namespace inverse_square_theorem_l1488_148805

/-- A function representing the inverse square relationship between x and y -/
noncomputable def inverse_square_relation (k : ℝ) (y : ℝ) : ℝ := k / y^2

/-- Theorem stating that given the inverse square relationship and a known point,
    we can determine the value of x for y = 3 -/
theorem inverse_square_theorem (k : ℝ) :
  (inverse_square_relation k 4 = 0.5625) →
  (inverse_square_relation k 3 = 1) :=
by
  sorry

#check inverse_square_theorem

end inverse_square_theorem_l1488_148805


namespace family_spent_36_dollars_l1488_148895

/-- The cost of a movie ticket in dollars -/
def ticket_cost : ℚ := 5

/-- The cost of popcorn as a fraction of the ticket cost -/
def popcorn_ratio : ℚ := 4/5

/-- The cost of soda as a fraction of the popcorn cost -/
def soda_ratio : ℚ := 1/2

/-- The number of tickets bought -/
def num_tickets : ℕ := 4

/-- The number of popcorn sets bought -/
def num_popcorn : ℕ := 2

/-- The number of soda cans bought -/
def num_soda : ℕ := 4

/-- Theorem: The total amount spent by the family is $36 -/
theorem family_spent_36_dollars :
  let popcorn_cost := ticket_cost * popcorn_ratio
  let soda_cost := popcorn_cost * soda_ratio
  let total_cost := (num_tickets : ℚ) * ticket_cost +
                    (num_popcorn : ℚ) * popcorn_cost +
                    (num_soda : ℚ) * soda_cost
  total_cost = 36 := by sorry

end family_spent_36_dollars_l1488_148895


namespace deschamps_farm_l1488_148870

theorem deschamps_farm (total_animals : ℕ) (total_legs : ℕ) 
  (h1 : total_animals = 160) 
  (h2 : total_legs = 400) : ∃ (chickens cows : ℕ),
  chickens + cows = total_animals ∧ 
  2 * chickens + 4 * cows = total_legs ∧ 
  cows = 40 := by
  sorry

end deschamps_farm_l1488_148870


namespace a_is_best_l1488_148837

-- Define the structure for an athlete
structure Athlete where
  name : String
  average : ℝ
  variance : ℝ

-- Define the athletes
def athleteA : Athlete := ⟨"A", 185, 3.6⟩
def athleteB : Athlete := ⟨"B", 180, 3.6⟩
def athleteC : Athlete := ⟨"C", 185, 7.4⟩
def athleteD : Athlete := ⟨"D", 180, 8.1⟩

-- Define a function to compare athletes
def isBetterAthlete (a1 a2 : Athlete) : Prop :=
  (a1.average > a2.average) ∨ (a1.average = a2.average ∧ a1.variance < a2.variance)

-- Theorem stating that A is the best athlete
theorem a_is_best : 
  isBetterAthlete athleteA athleteB ∧ 
  isBetterAthlete athleteA athleteC ∧ 
  isBetterAthlete athleteA athleteD :=
sorry

end a_is_best_l1488_148837


namespace robot_position_difference_l1488_148875

-- Define the robot's position function
def robot_position (n : ℕ) : ℤ :=
  let full_cycles := n / 7
  let remainder := n % 7
  let cycle_progress := if remainder ≤ 4 then remainder else 4 - (remainder - 4)
  full_cycles + cycle_progress

-- State the theorem
theorem robot_position_difference : robot_position 2007 - robot_position 2011 = 0 := by
  sorry

end robot_position_difference_l1488_148875


namespace alpha_beta_inequality_l1488_148840

theorem alpha_beta_inequality (α β : ℝ) (h1 : -1 < α) (h2 : α < β) (h3 : β < 1) :
  -2 < α - β ∧ α - β < 0 := by
  sorry

end alpha_beta_inequality_l1488_148840


namespace hans_room_options_l1488_148831

/-- Represents a hotel with floors and rooms -/
structure Hotel where
  total_floors : ℕ
  rooms_per_floor : ℕ
  available_rooms_on_odd_floor : ℕ

/-- Calculates the number of available rooms in the hotel -/
def available_rooms (h : Hotel) : ℕ :=
  (h.total_floors / 2) * h.available_rooms_on_odd_floor

/-- The specific hotel in the problem -/
def problem_hotel : Hotel :=
  { total_floors := 20
    rooms_per_floor := 15
    available_rooms_on_odd_floor := 10 }

/-- Theorem stating that the number of available rooms in the problem hotel is 100 -/
theorem hans_room_options : available_rooms problem_hotel = 100 := by
  sorry

end hans_room_options_l1488_148831


namespace roots_are_irrational_l1488_148823

-- Define the quadratic equation
def quadratic (k : ℝ) (x : ℝ) : ℝ := x^2 - 4*k*x + 3*k^2 - 2

-- Define the discriminant of the quadratic equation
def discriminant (k : ℝ) : ℝ := (4*k)^2 - 4*(3*k^2 - 2)

-- Theorem statement
theorem roots_are_irrational (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ quadratic k x = 0 ∧ quadratic k y = 0 ∧ x * y = 10) →
  ∃ x y : ℝ, x ≠ y ∧ quadratic k x = 0 ∧ quadratic k y = 0 ∧ ¬(∃ q : ℚ, x = ↑q) ∧ ¬(∃ q : ℚ, y = ↑q) :=
by sorry

end roots_are_irrational_l1488_148823


namespace exponent_multiplication_l1488_148829

theorem exponent_multiplication (a : ℝ) : a^4 * a^2 = a^6 := by
  sorry

end exponent_multiplication_l1488_148829


namespace sufficient_but_not_necessary_condition_l1488_148832

theorem sufficient_but_not_necessary_condition :
  ∃ (a b : ℝ), (a > 1 ∧ b > 1 → a * b > 1) ∧
  ¬(a * b > 1 → a > 1 ∧ b > 1) :=
by sorry

end sufficient_but_not_necessary_condition_l1488_148832


namespace even_number_divisor_sum_l1488_148842

theorem even_number_divisor_sum (n : ℕ) : 
  Even n →
  (∃ (divs : Finset ℕ), divs = {d : ℕ | d ∣ n} ∧ 
    (divs.sum (λ d => (1 : ℚ) / d) = 1620 / 1003)) →
  ∃ k : ℕ, n = 2006 * k :=
by sorry

end even_number_divisor_sum_l1488_148842


namespace factorial_difference_is_perfect_square_l1488_148865

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_difference_is_perfect_square (q : ℕ) (r : ℕ) 
  (h : factorial (q + 2) - factorial (q + 1) = factorial q * r) :
  r = (q + 1) ^ 2 := by
  sorry

end factorial_difference_is_perfect_square_l1488_148865


namespace total_tape_is_870_l1488_148858

/-- Calculates the tape length for a side, including overlap -/
def tape_length (side : ℕ) : ℕ := side + 2

/-- Calculates the tape needed for a single box -/
def box_tape (length width : ℕ) : ℕ :=
  tape_length length + 2 * tape_length width

/-- The total tape needed for all boxes -/
def total_tape : ℕ :=
  5 * box_tape 30 15 +
  2 * box_tape 40 40 +
  3 * box_tape 50 20

theorem total_tape_is_870 : total_tape = 870 := by
  sorry

end total_tape_is_870_l1488_148858


namespace cycle_price_calculation_l1488_148830

/-- Proves that given a cycle sold for Rs. 1080 with a 60% gain, the original price of the cycle was Rs. 675. -/
theorem cycle_price_calculation (selling_price : ℝ) (gain_percent : ℝ) 
  (h1 : selling_price = 1080)
  (h2 : gain_percent = 60) :
  ∃ (original_price : ℝ), 
    selling_price = original_price * (1 + gain_percent / 100) ∧ 
    original_price = 675 :=
by sorry

end cycle_price_calculation_l1488_148830


namespace exists_m_composite_l1488_148856

theorem exists_m_composite (n : ℕ) : ∃ m : ℕ, ∃ k : ℕ, k > 1 ∧ k < n * m + 1 ∧ (n * m + 1) % k = 0 := by
  sorry

end exists_m_composite_l1488_148856


namespace sequence_sum_l1488_148807

theorem sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) : 
  (∀ n : ℕ, S n = n^3) → 
  (∀ n : ℕ, n ≥ 2 → a n = S n - S (n-1)) →
  a 1 = S 1 →
  a 5 + a 6 = 152 := by
sorry

end sequence_sum_l1488_148807


namespace tims_number_l1488_148890

theorem tims_number (n : ℕ) : 
  (∃ k l : ℕ, n = 9 * k - 2 ∧ n = 8 * l - 4) ∧ 
  n < 150 ∧ 
  (∀ m : ℕ, (∃ p q : ℕ, m = 9 * p - 2 ∧ m = 8 * q - 4) ∧ m < 150 → m ≤ n) →
  n = 124 := by
sorry

end tims_number_l1488_148890


namespace f_range_l1488_148845

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos (3*x/2), Real.sin (3*x/2))
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos (x/2), -Real.sin (x/2))

noncomputable def f (x : ℝ) : ℝ :=
  (a x).1 * (b x).1 + (a x).2 * (b x).2 -
  Real.sqrt (((a x).1 - (b x).1)^2 + ((a x).2 - (b x).2)^2)

theorem f_range :
  ∀ y ∈ Set.Icc (-3 : ℝ) (-1/2),
    ∃ x ∈ Set.Ico (π/6 : ℝ) (2*π/3),
      f x = y :=
sorry

end f_range_l1488_148845


namespace series_value_l1488_148863

def series_term (n : ℕ) : ℤ := n * (n + 1) - (n + 1) * (n + 2)

def series_sum : ℕ → ℤ
  | 0 => 0
  | n + 1 => series_sum n + series_term (n + 1)

theorem series_value : series_sum 2000 = 2004002 := by
  sorry

end series_value_l1488_148863


namespace correct_calculation_result_l1488_148855

theorem correct_calculation_result (x : ℤ) (h : x - 63 = 8) : x * 8 = 568 := by
  sorry

end correct_calculation_result_l1488_148855


namespace secret_spread_day_l1488_148834

/-- The number of people who know the secret after n days -/
def secret_spread (n : ℕ) : ℕ := (3^(n+1) - 1) / 2

/-- The day when 3280 students know the secret -/
theorem secret_spread_day :
  ∃ n : ℕ, secret_spread n = 3280 ∧ n = 7 :=
sorry

end secret_spread_day_l1488_148834


namespace petya_running_time_l1488_148896

theorem petya_running_time (V D : ℝ) (hV : V > 0) (hD : D > 0) : 
  let T := D / V
  let V1 := 1.25 * V
  let V2 := 0.8 * V
  let T1 := D / (2 * V1)
  let T2 := D / (2 * V2)
  let Tactual := T1 + T2
  Tactual > T :=
by sorry

end petya_running_time_l1488_148896


namespace adjacent_knights_probability_l1488_148889

def number_of_knights : ℕ := 30
def chosen_knights : ℕ := 4

def probability_adjacent_knights : ℚ :=
  1 - (number_of_knights - chosen_knights + 1) * (number_of_knights - chosen_knights - 1) * (number_of_knights - chosen_knights - 3) * (number_of_knights - chosen_knights - 5) / (number_of_knights.choose chosen_knights)

theorem adjacent_knights_probability :
  probability_adjacent_knights = 53 / 85 := by sorry

end adjacent_knights_probability_l1488_148889


namespace quadratic_roots_relation_l1488_148819

/-- Given two quadratic equations with real roots and specific conditions, prove that m = 4 --/
theorem quadratic_roots_relation (m n : ℝ) (x₁ x₂ y₁ y₂ : ℝ) : 
  n < 0 →
  x₁^2 + m^2*x₁ + n = 0 →
  x₂^2 + m^2*x₂ + n = 0 →
  y₁^2 + 5*m*y₁ + 7 = 0 →
  y₂^2 + 5*m*y₂ + 7 = 0 →
  x₁ - y₁ = 2 →
  x₂ - y₂ = 2 →
  m = 4 := by sorry

end quadratic_roots_relation_l1488_148819


namespace c_is_largest_l1488_148828

-- Define the five numbers
def a : ℚ := 7.4683
def b : ℚ := 7 + 468/1000 + 3/9990  -- 7.468̅3
def c : ℚ := 7 + 46/100 + 83/9900   -- 7.46̅83
def d : ℚ := 7 + 4/10 + 683/999     -- 7.4̅683
def e : ℚ := 7 + 4683/9999          -- 7.̅4683

-- Theorem stating that c is the largest
theorem c_is_largest : c > a ∧ c > b ∧ c > d ∧ c > e := by sorry

end c_is_largest_l1488_148828


namespace sum_of_acute_angles_not_always_obtuse_l1488_148885

-- Define what an acute angle is
def is_acute_angle (α : Real) : Prop := 0 < α ∧ α < Real.pi / 2

-- Define what an obtuse angle is
def is_obtuse_angle (α : Real) : Prop := Real.pi / 2 < α ∧ α < Real.pi

-- Theorem stating that the sum of two acute angles is not always obtuse
theorem sum_of_acute_angles_not_always_obtuse :
  ∃ (α β : Real), is_acute_angle α ∧ is_acute_angle β ∧ ¬is_obtuse_angle (α + β) :=
sorry

end sum_of_acute_angles_not_always_obtuse_l1488_148885


namespace consecutive_numbers_sum_l1488_148857

theorem consecutive_numbers_sum (n : ℕ) :
  (n + 1) + (n + 2) + (n + 3) = 2 * (n + (n - 1) + (n - 2)) →
  n + 3 = 7 ∧ (n - 2) + (n - 1) + n + (n + 1) + (n + 2) + (n + 3) = 27 :=
by sorry

end consecutive_numbers_sum_l1488_148857


namespace euler_totient_equality_l1488_148808

-- Define the Euler's totient function
def phi (n : ℕ) : ℕ := sorry

-- Define the property of being an odd number
def is_odd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

-- Theorem statement
theorem euler_totient_equality (n : ℕ) (p : ℕ) (h_p : Prime p) :
  phi n = phi (n * p) ↔ p = 2 ∧ is_odd n :=
sorry

end euler_totient_equality_l1488_148808


namespace syrup_problem_l1488_148866

/-- Represents a container with a certain volume of liquid --/
structure Container where
  syrup : ℝ
  water : ℝ

/-- The state of the three containers --/
structure ContainerState where
  a : Container
  b : Container
  c : Container

/-- Represents a pouring action --/
inductive PourAction
  | PourAll : Fin 3 → Fin 3 → PourAction
  | Equalize : Fin 3 → Fin 3 → PourAction
  | PourToSink : Fin 3 → PourAction

/-- Defines if a given sequence of actions is valid --/
def isValidActionSequence (initialState : ContainerState) (actions : List PourAction) : Prop :=
  sorry

/-- Defines if a final state has 10L of 30% syrup in one container --/
def hasTenLitersThirtyPercentSyrup (state : ContainerState) : Prop :=
  sorry

/-- The main theorem to prove --/
theorem syrup_problem (n : ℕ) :
  (∃ (actions : List PourAction),
    isValidActionSequence
      ⟨⟨3, 0⟩, ⟨0, n⟩, ⟨0, 0⟩⟩
      actions ∧
    hasTenLitersThirtyPercentSyrup
      (actions.foldl (λ state action => sorry) ⟨⟨3, 0⟩, ⟨0, n⟩, ⟨0, 0⟩⟩)) ↔
  ∃ (k : ℕ), n = 3 * k + 1 :=
sorry

end syrup_problem_l1488_148866


namespace sum_first_ten_terms_l1488_148803

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

def sum_arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem sum_first_ten_terms :
  sum_arithmetic_sequence (-3) 4 10 = 150 := by
sorry

end sum_first_ten_terms_l1488_148803


namespace cloak_change_in_silver_l1488_148838

/-- Represents the price of an invisibility cloak and the change received in different scenarios --/
structure CloakTransaction where
  silver_paid : ℕ
  gold_change : ℕ

/-- Calculates the exchange rate between silver and gold coins --/
def exchange_rate (t1 t2 : CloakTransaction) : ℚ :=
  (t1.silver_paid - t2.silver_paid : ℚ) / (t1.gold_change - t2.gold_change)

/-- Calculates the price of the cloak in gold coins --/
def cloak_price_gold (t : CloakTransaction) (rate : ℚ) : ℚ :=
  t.silver_paid / rate - t.gold_change

/-- Theorem stating the change received when buying a cloak with gold coins --/
theorem cloak_change_in_silver 
  (t1 t2 : CloakTransaction)
  (h1 : t1.silver_paid = 20 ∧ t1.gold_change = 4)
  (h2 : t2.silver_paid = 15 ∧ t2.gold_change = 1)
  (gold_paid : ℕ)
  (h3 : gold_paid = 14) :
  ∃ (silver_change : ℕ), silver_change = 10 := by
  sorry

end cloak_change_in_silver_l1488_148838


namespace power_product_square_l1488_148802

theorem power_product_square (a b : ℝ) : (a^2 * b)^2 = a^4 * b^2 := by sorry

end power_product_square_l1488_148802


namespace integral_sin4_cos2_l1488_148809

theorem integral_sin4_cos2 (x : Real) :
  let f := fun (x : Real) => (1/16) * x - (1/64) * Real.sin (4*x) - (1/48) * Real.sin (2*x)^3
  (deriv f) x = Real.sin x^4 * Real.cos x^2 := by
  sorry

end integral_sin4_cos2_l1488_148809


namespace marble_distribution_l1488_148894

theorem marble_distribution (total_marbles : ℕ) (group_size : ℕ) : 
  total_marbles = 364 →
  (total_marbles / group_size : ℚ) - (total_marbles / (group_size + 2) : ℚ) = 1 →
  group_size = 26 := by
  sorry

end marble_distribution_l1488_148894


namespace divide_multiply_problem_l1488_148887

theorem divide_multiply_problem : (2.25 / 3) * 12 = 9 := by
  sorry

end divide_multiply_problem_l1488_148887


namespace quadratic_inequality_solution_set_l1488_148817

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h : ∀ x : ℝ, -1/2 < x ∧ x < 2 ↔ f a b c x > 0) : 
  a < 0 ∧ c > 0 := by
  sorry

end quadratic_inequality_solution_set_l1488_148817


namespace three_planes_theorem_l1488_148847

-- Define the two equations
def equation_cubic (x y z : ℝ) : Prop :=
  x^3 + y^3 + z^3 = (x + y + z)^3

def equation_quintic (x y z : ℝ) : Prop :=
  x^5 + y^5 + z^5 = (x + y + z)^5

-- State the theorem
theorem three_planes_theorem :
  ∀ (x y z : ℝ),
    (equation_cubic x y z → (x + y) * (y + z) * (z + x) = 0) ∧
    (equation_quintic x y z → (x + y) * (y + z) * (z + x) = 0) :=
by sorry

end three_planes_theorem_l1488_148847


namespace rectangular_prism_volume_l1488_148873

theorem rectangular_prism_volume (a b c : ℕ) : 
  a ≥ 2 ∧ b ≥ 2 ∧ c ≥ 2 →
  2 * ((a - 2) * (b - 2) + (b - 2) * (c - 2) + (a - 2) * (c - 2)) = 24 →
  4 * ((a - 2) + (b - 2) + (c - 2)) = 28 →
  a * b * c = 60 := by
  sorry

end rectangular_prism_volume_l1488_148873


namespace stephanie_store_visits_l1488_148800

/-- Represents the number of oranges Stephanie buys per store visit -/
def oranges_per_visit : ℕ := 2

/-- Represents the total number of oranges Stephanie bought last month -/
def total_oranges : ℕ := 16

/-- Represents the number of times Stephanie went to the store -/
def store_visits : ℕ := total_oranges / oranges_per_visit

theorem stephanie_store_visits : store_visits = 8 := by
  sorry

end stephanie_store_visits_l1488_148800


namespace joan_has_three_marbles_l1488_148876

/-- The number of yellow marbles Mary has -/
def mary_marbles : ℕ := 9

/-- The total number of yellow marbles Mary and Joan have together -/
def total_marbles : ℕ := 12

/-- The number of yellow marbles Joan has -/
def joan_marbles : ℕ := total_marbles - mary_marbles

theorem joan_has_three_marbles : joan_marbles = 3 := by
  sorry

end joan_has_three_marbles_l1488_148876


namespace slope_of_line_l1488_148818

theorem slope_of_line (x₁ x₂ y₁ y₂ : ℝ) (h₁ : x₁ ≠ x₂) 
  (h₂ : 1 / x₁ + 2 / y₁ = 0) (h₃ : 1 / x₂ + 2 / y₂ = 0) :
  (y₂ - y₁) / (x₂ - x₁) = -2 :=
by sorry

end slope_of_line_l1488_148818


namespace sqrt_simplification_l1488_148871

theorem sqrt_simplification : Real.sqrt (5 - 2 * Real.sqrt 6) = Real.sqrt 3 - Real.sqrt 2 := by
  sorry

end sqrt_simplification_l1488_148871


namespace unknown_number_proof_l1488_148820

theorem unknown_number_proof : (12^1 * 6^4) / 432 = 36 := by
  sorry

end unknown_number_proof_l1488_148820


namespace min_value_of_f_l1488_148883

/-- The base-10 logarithm function -/
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

/-- The function to be minimized -/
noncomputable def f (x : ℝ) : ℝ := lg x + (Real.log 10) / (Real.log x)

theorem min_value_of_f :
  ∀ x > 1, f x ≥ 2 ∧ f 10 = 2 :=
by sorry

end min_value_of_f_l1488_148883


namespace polynomial_value_theorem_l1488_148848

theorem polynomial_value_theorem (a : ℝ) : 
  2 * a^2 + 3 * a + 1 = 6 → -6 * a^2 - 9 * a + 8 = -7 := by
  sorry

end polynomial_value_theorem_l1488_148848


namespace eraser_price_l1488_148853

/-- Proves that the price of an eraser is $1 given the problem conditions --/
theorem eraser_price (pencils_sold : ℕ) (total_earnings : ℝ) 
  (h1 : pencils_sold = 20)
  (h2 : total_earnings = 80)
  (h3 : ∀ p : ℝ, p > 0 → 
    pencils_sold * p + 2 * pencils_sold * (p / 2) = total_earnings) :
  ∃ (pencil_price : ℝ), 
    pencil_price > 0 ∧ 
    pencil_price / 2 = 1 := by
  sorry

end eraser_price_l1488_148853


namespace probability_one_black_ball_l1488_148843

/-- The probability of drawing exactly one black ball when drawing two balls without replacement from a box containing 3 white balls and 2 black balls -/
theorem probability_one_black_ball (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) 
  (h1 : total_balls = white_balls + black_balls)
  (h2 : total_balls = 5)
  (h3 : white_balls = 3)
  (h4 : black_balls = 2) : 
  (white_balls * black_balls) / ((total_balls * (total_balls - 1)) / 2) = 3 / 5 :=
sorry

end probability_one_black_ball_l1488_148843


namespace sasha_max_quarters_l1488_148881

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 25 / 100

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 5 / 100

/-- The total amount Sasha has in dollars -/
def total_amount : ℚ := 32 / 10

/-- The maximum number of quarters Sasha can have -/
def max_quarters : ℕ := 10

theorem sasha_max_quarters :
  ∀ q : ℕ,
  (q : ℚ) * (quarter_value + nickel_value) ≤ total_amount →
  q ≤ max_quarters :=
by sorry

end sasha_max_quarters_l1488_148881


namespace rectangle_area_l1488_148811

theorem rectangle_area (width : ℝ) (length : ℝ) (h1 : width = 4) (h2 : length = 3 * width) :
  width * length = 48 := by
  sorry

end rectangle_area_l1488_148811


namespace fill_time_three_pipes_l1488_148827

-- Define the tank's volume
variable (T : ℝ)

-- Define the rates at which pipes X, Y, and Z fill the tank
variable (X Y Z : ℝ)

-- Define the conditions
def condition1 : Prop := X + Y = T / 3
def condition2 : Prop := X + Z = T / 4
def condition3 : Prop := Y + Z = T / 2

-- State the theorem
theorem fill_time_three_pipes 
  (h1 : condition1 T X Y) 
  (h2 : condition2 T X Z) 
  (h3 : condition3 T Y Z) :
  1 / (X + Y + Z) = 24 / 13 := by
  sorry

end fill_time_three_pipes_l1488_148827


namespace solve_equation_l1488_148850

theorem solve_equation (b : ℚ) (h : b + b / 4 = 10 / 4) : b = 2 := by
  sorry

end solve_equation_l1488_148850


namespace units_digit_of_composite_product_l1488_148846

def first_four_composites : List Nat := [4, 6, 8, 9]

def product_of_list (l : List Nat) : Nat :=
  l.foldl (· * ·) 1

def units_digit (n : Nat) : Nat :=
  n % 10

theorem units_digit_of_composite_product :
  units_digit (product_of_list first_four_composites) = 8 := by
  sorry

end units_digit_of_composite_product_l1488_148846


namespace special_multiples_count_l1488_148877

def count_multiples (n : ℕ) (d : ℕ) : ℕ := (n / d : ℕ)

def count_special_multiples (n : ℕ) : ℕ :=
  count_multiples n 5 + count_multiples n 6 - count_multiples n 15

theorem special_multiples_count :
  count_special_multiples 3000 = 900 := by sorry

end special_multiples_count_l1488_148877


namespace rachel_chairs_l1488_148815

/-- The number of chairs Rachel bought -/
def num_chairs : ℕ := 7

/-- The number of tables Rachel bought -/
def num_tables : ℕ := 3

/-- The time spent on each piece of furniture (in minutes) -/
def time_per_furniture : ℕ := 4

/-- The total time spent (in minutes) -/
def total_time : ℕ := 40

theorem rachel_chairs :
  num_chairs = (total_time - num_tables * time_per_furniture) / time_per_furniture :=
by sorry

end rachel_chairs_l1488_148815


namespace percentage_difference_l1488_148897

theorem percentage_difference : (56 * 0.50) - (50 * 0.30) = 13 := by sorry

end percentage_difference_l1488_148897


namespace circle_center_transformation_l1488_148879

/-- Reflects a point about the line y=x --/
def reflect_about_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

/-- Translates a point by a given vector --/
def translate (p : ℝ × ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + v.1, p.2 + v.2)

/-- The main theorem --/
theorem circle_center_transformation :
  let initial_center : ℝ × ℝ := (8, -3)
  let reflected_center := reflect_about_y_eq_x initial_center
  let translation_vector : ℝ × ℝ := (4, 2)
  let final_center := translate reflected_center translation_vector
  final_center = (1, 10) := by
  sorry

end circle_center_transformation_l1488_148879


namespace max_value_theorem_l1488_148814

theorem max_value_theorem (a b : ℝ) : 
  a^2 = (1 + 2*b) * (1 - 2*b) →
  ∃ (x : ℝ), x = (2*a*b)/(|a| + 2*|b|) ∧ 
             ∀ (y : ℝ), y = (2*a*b)/(|a| + 2*|b|) → y ≤ x ∧
             x = Real.sqrt 2 / 4 :=
sorry

end max_value_theorem_l1488_148814


namespace dragon_boat_festival_visitors_scientific_notation_l1488_148804

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem dragon_boat_festival_visitors_scientific_notation :
  toScientificNotation 82600000 = ScientificNotation.mk 8.26 7 sorry := by
  sorry

end dragon_boat_festival_visitors_scientific_notation_l1488_148804


namespace intersecting_lines_k_value_l1488_148836

/-- Three lines intersecting at a single point -/
structure ThreeIntersectingLines where
  k : ℚ
  intersect_point : ℝ × ℝ
  line1 : ∀ (x y : ℝ), x + k * y = 0 → (x, y) = intersect_point
  line2 : ∀ (x y : ℝ), 2 * x + 3 * y + 8 = 0 → (x, y) = intersect_point
  line3 : ∀ (x y : ℝ), x - y - 1 = 0 → (x, y) = intersect_point

/-- If three lines intersect at a single point, then k = -1/2 -/
theorem intersecting_lines_k_value (lines : ThreeIntersectingLines) : lines.k = -1/2 := by
  sorry

end intersecting_lines_k_value_l1488_148836


namespace marble_probability_l1488_148806

theorem marble_probability : 
  let green : ℕ := 4
  let white : ℕ := 3
  let red : ℕ := 5
  let blue : ℕ := 6
  let total : ℕ := green + white + red + blue
  let favorable : ℕ := green + white
  (favorable : ℚ) / total = 7 / 18 := by
sorry

end marble_probability_l1488_148806


namespace expression_simplification_l1488_148833

theorem expression_simplification (x y : ℚ) (hx : x = -4) (hy : y = -1/2) :
  x^2 - (x^2 - 2*x*y + 3*(x*y - 1/3*y^2)) = -7/4 := by sorry

end expression_simplification_l1488_148833


namespace coin_stack_arrangements_l1488_148851

/-- Represents the number of valid arrangements for n coins where no three consecutive coins are face to face to face -/
def validArrangements : Nat → Nat
  | 0 => 1
  | 1 => 2
  | 2 => 4
  | n + 3 => validArrangements (n + 2) + validArrangements (n + 1) + validArrangements n

/-- The number of ways to choose 5 positions out of 10 for gold coins -/
def colorDistributions : Nat := Nat.choose 10 5

/-- The total number of distinguishable arrangements of 5 gold and 5 silver coins
    with the given face-to-face constraint -/
def totalArrangements : Nat := colorDistributions * validArrangements 10

theorem coin_stack_arrangements :
  totalArrangements = 69048 := by
  sorry

end coin_stack_arrangements_l1488_148851


namespace oreilly_triple_8_49_l1488_148825

/-- Definition of an O'Reilly triple -/
def is_oreilly_triple (a b x : ℕ+) : Prop :=
  (a.val : ℝ)^(1/3) + (b.val : ℝ)^(1/2) = x.val

/-- Theorem: If (8,49,x) is an O'Reilly triple, then x = 9 -/
theorem oreilly_triple_8_49 (x : ℕ+) :
  is_oreilly_triple 8 49 x → x = 9 := by
  sorry

end oreilly_triple_8_49_l1488_148825


namespace union_complement_equal_l1488_148801

def U : Finset ℕ := {1,2,3,4,5,6}
def M : Finset ℕ := {1,3,4}
def N : Finset ℕ := {3,5,6}

theorem union_complement_equal : M ∪ (U \ N) = {1,2,3,4} := by
  sorry

end union_complement_equal_l1488_148801


namespace oranges_picked_l1488_148886

theorem oranges_picked (michaela_full : ℕ) (cassandra_full : ℕ) (remaining : ℕ) : 
  michaela_full = 20 → 
  cassandra_full = 2 * michaela_full → 
  remaining = 30 → 
  michaela_full + cassandra_full + remaining = 90 := by
  sorry

end oranges_picked_l1488_148886


namespace susan_remaining_money_l1488_148874

def susan_spending (initial_amount games_multiplier snacks_cost souvenir_cost : ℕ) : ℕ :=
  initial_amount - (snacks_cost + games_multiplier * snacks_cost + souvenir_cost)

theorem susan_remaining_money :
  susan_spending 80 3 15 10 = 10 := by
  sorry

end susan_remaining_money_l1488_148874


namespace complement_of_A_in_U_l1488_148821

def U : Set ℕ := { x | (x - 1) / (5 - x) > 0 ∧ x > 0 }

def A : Set ℕ := {2, 3}

theorem complement_of_A_in_U : Set.compl A ∩ U = {4} := by sorry

end complement_of_A_in_U_l1488_148821


namespace cafeteria_apples_l1488_148854

/-- The number of apples handed out to students -/
def apples_to_students : ℕ := 27

/-- The number of pies that can be made -/
def number_of_pies : ℕ := 5

/-- The number of apples needed for each pie -/
def apples_per_pie : ℕ := 4

/-- The total number of apples in the cafeteria initially -/
def total_apples : ℕ := apples_to_students + number_of_pies * apples_per_pie

theorem cafeteria_apples : total_apples = 47 := by sorry

end cafeteria_apples_l1488_148854


namespace fitted_bowling_ball_volume_l1488_148852

/-- The volume of a fitted bowling ball -/
theorem fitted_bowling_ball_volume :
  let sphere_diameter : ℝ := 24
  let hole1_diameter : ℝ := 4
  let hole2_diameter : ℝ := 4
  let hole3_diameter : ℝ := 3
  let hole_depth : ℝ := 6
  
  let sphere_volume := (4 / 3) * π * (sphere_diameter / 2) ^ 3
  let hole1_volume := π * (hole1_diameter / 2) ^ 2 * hole_depth
  let hole2_volume := π * (hole2_diameter / 2) ^ 2 * hole_depth
  let hole3_volume := π * (hole3_diameter / 2) ^ 2 * hole_depth
  
  sphere_volume - (hole1_volume + hole2_volume + hole3_volume) = 2242.5 * π :=
by sorry


end fitted_bowling_ball_volume_l1488_148852


namespace complex_modulus_l1488_148859

theorem complex_modulus (z : ℂ) (h : z * (2 + Complex.I) = 1 + 7 * Complex.I) : 
  Complex.abs z = Real.sqrt 10 := by
  sorry

end complex_modulus_l1488_148859


namespace seventh_observation_value_l1488_148893

theorem seventh_observation_value (initial_count : Nat) (initial_average : ℝ) (new_average : ℝ) :
  initial_count = 6 →
  initial_average = 14 →
  new_average = 13 →
  (initial_count * initial_average + 7) / (initial_count + 1) = new_average →
  7 = (initial_count + 1) * new_average - initial_count * initial_average :=
by sorry

end seventh_observation_value_l1488_148893


namespace unique_integer_term_l1488_148816

def is_integer_term (n : ℕ) : Prop :=
  ∃ k : ℤ, (n^2 + 1).factorial / ((n.factorial)^(n + 2)) = k

theorem unique_integer_term :
  ∃! n : ℕ, 1 ≤ n ∧ n ≤ 100 ∧ is_integer_term n :=
sorry

end unique_integer_term_l1488_148816


namespace condo_penthouse_floors_l1488_148872

/-- Represents a condo building with regular and penthouse floors -/
structure Condo where
  total_floors : ℕ
  regular_units_per_floor : ℕ
  penthouse_units_per_floor : ℕ
  total_units : ℕ

/-- Calculates the number of penthouse floors in a condo -/
def penthouse_floors (c : Condo) : ℕ :=
  c.total_floors - (c.total_units - 2 * c.total_floors) / (c.regular_units_per_floor - c.penthouse_units_per_floor)

/-- Theorem stating that the condo with given specifications has 2 penthouse floors -/
theorem condo_penthouse_floors :
  let c : Condo := {
    total_floors := 23,
    regular_units_per_floor := 12,
    penthouse_units_per_floor := 2,
    total_units := 256
  }
  penthouse_floors c = 2 := by
  sorry

end condo_penthouse_floors_l1488_148872


namespace door_opening_proofs_l1488_148841

/-- The number of buttons on the lock -/
def num_buttons : Nat := 10

/-- The number of buttons that need to be pressed simultaneously -/
def buttons_to_press : Nat := 3

/-- Time taken for each attempt in seconds -/
def time_per_attempt : Nat := 2

/-- The total number of possible combinations -/
def total_combinations : Nat := (num_buttons.choose buttons_to_press)

/-- The maximum time needed to try all combinations in seconds -/
def max_time : Nat := total_combinations * time_per_attempt

/-- The average number of attempts needed -/
def avg_attempts : Rat := (1 + total_combinations) / 2

/-- The average time needed in seconds -/
def avg_time : Rat := avg_attempts * time_per_attempt

/-- The maximum number of attempts possible in 60 seconds -/
def max_attempts_in_minute : Nat := 60 / time_per_attempt

theorem door_opening_proofs :
  (max_time = 240) ∧
  (avg_time = 121) ∧
  (max_attempts_in_minute = 30) ∧
  ((max_attempts_in_minute - 1 : Rat) / total_combinations = 29 / 120) := by
  sorry

end door_opening_proofs_l1488_148841


namespace boys_passed_exam_l1488_148860

/-- Proves the number of boys who passed an examination given specific conditions -/
theorem boys_passed_exam (total_boys : ℕ) (overall_avg : ℚ) (pass_avg : ℚ) (fail_avg : ℚ) :
  total_boys = 120 →
  overall_avg = 36 →
  pass_avg = 39 →
  fail_avg = 15 →
  ∃ (passed_boys : ℕ),
    passed_boys = 105 ∧
    passed_boys ≤ total_boys ∧
    (passed_boys : ℚ) * pass_avg + (total_boys - passed_boys : ℚ) * fail_avg = (total_boys : ℚ) * overall_avg :=
by sorry

end boys_passed_exam_l1488_148860


namespace sufficient_but_not_necessary_l1488_148882

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)  -- subset relation for a line in a plane
variable (perpendicular : Line → Line → Prop)  -- perpendicular relation between lines
variable (perpendicularToPlane : Line → Plane → Prop)  -- perpendicular relation between a line and a plane
variable (parallel : Plane → Plane → Prop)  -- parallel relation between planes

-- State the theorem
theorem sufficient_but_not_necessary
  (a b : Line) (α β : Plane)
  (h1 : subset a α)
  (h2 : perpendicularToPlane b β)
  (h3 : parallel α β) :
  perpendicular a b ∧
  ¬(∀ (a b : Line) (α β : Plane),
    perpendicular a b →
    subset a α ∧ perpendicularToPlane b β ∧ parallel α β) :=
by sorry

end sufficient_but_not_necessary_l1488_148882


namespace two_numbers_problem_l1488_148891

theorem two_numbers_problem (x y : ℚ) : 
  (4 * y = 9 * x) → 
  (y - x = 12) → 
  y = 108 / 5 := by
sorry

end two_numbers_problem_l1488_148891


namespace quadratic_root_implies_k_l1488_148869

theorem quadratic_root_implies_k (k : ℝ) : 
  (∃ x : ℝ, 2 * x^2 + 3 * x - k = 0) ∧ (2 * 5^2 + 3 * 5 - k = 0) → k = 65 := by
  sorry

end quadratic_root_implies_k_l1488_148869


namespace zhang_income_ratio_l1488_148813

/-- Represents the per capita income of a village at a given time -/
structure Income where
  amount : ℝ

/-- Represents the state of two villages' incomes at two different times -/
structure VillageIncomes where
  li_past : Income
  li_present : Income
  zhang_past : Income
  zhang_present : Income

/-- The conditions of the problem -/
def income_conditions (v : VillageIncomes) : Prop :=
  v.zhang_past.amount = 0.4 * v.li_past.amount ∧
  v.zhang_present.amount = 0.8 * v.li_present.amount ∧
  v.li_present.amount = 3 * v.li_past.amount

/-- The theorem to be proved -/
theorem zhang_income_ratio (v : VillageIncomes) 
  (h : income_conditions v) : 
  v.zhang_present.amount / v.zhang_past.amount = 6 := by
  sorry


end zhang_income_ratio_l1488_148813


namespace triangle_theorem_l1488_148880

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.a * (Real.sqrt 3 * Real.tan t.B - 1) = 
        (t.b * Real.cos t.A / Real.cos t.B) + (t.c * Real.cos t.A / Real.cos t.C))
  (h2 : t.a + t.b + t.c = 20)
  (h3 : (1/2) * t.a * t.b * Real.sin t.C = 10 * Real.sqrt 3)
  (h4 : t.a > t.b) :
  t.C = Real.pi / 3 ∧ t.a = 8 ∧ t.b = 5 ∧ t.c = 7 := by
  sorry


end triangle_theorem_l1488_148880


namespace total_cost_for_nuggets_l1488_148867

-- Define the number of chicken nuggets ordered
def total_nuggets : ℕ := 100

-- Define the number of nuggets in a box
def nuggets_per_box : ℕ := 20

-- Define the cost of one box
def cost_per_box : ℕ := 4

-- Theorem to prove
theorem total_cost_for_nuggets : 
  (total_nuggets / nuggets_per_box) * cost_per_box = 20 := by
  sorry

end total_cost_for_nuggets_l1488_148867


namespace geometric_sequence_first_term_l1488_148810

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ ∀ n, a (n + 1) = a n * r

theorem geometric_sequence_first_term 
  (a : ℕ → ℚ) 
  (h_geometric : is_geometric_sequence a) 
  (h_third_term : a 2 = 8)
  (h_fifth_term : a 4 = 27 / 4) :
  a 0 = 256 / 27 := by
sorry

end geometric_sequence_first_term_l1488_148810


namespace hcl_moles_combined_l1488_148822

/-- The number of moles of HCl combined to produce a given amount of NH4Cl -/
theorem hcl_moles_combined 
  (nh3_moles : ℝ) 
  (nh4cl_grams : ℝ) 
  (nh4cl_molar_mass : ℝ) 
  (h1 : nh3_moles = 3)
  (h2 : nh4cl_grams = 159)
  (h3 : nh4cl_molar_mass = 53.50) :
  ∃ hcl_moles : ℝ, abs (hcl_moles - (nh4cl_grams / nh4cl_molar_mass)) < 0.001 :=
by
  sorry

#check hcl_moles_combined

end hcl_moles_combined_l1488_148822


namespace students_on_field_trip_l1488_148839

def total_budget : ℕ := 350
def bus_rental_cost : ℕ := 100
def admission_cost_per_student : ℕ := 10

theorem students_on_field_trip : 
  (total_budget - bus_rental_cost) / admission_cost_per_student = 25 := by
  sorry

end students_on_field_trip_l1488_148839


namespace twenty_multi_painted_cubes_l1488_148862

/-- Represents a cube with painted faces -/
structure PaintedCube where
  size : ℕ
  top_painted : Bool
  sides_painted : Bool
  bottom_painted : Bool

/-- Counts the number of unit cubes with at least two painted faces -/
def count_multi_painted_cubes (cube : PaintedCube) : ℕ :=
  sorry

/-- The main theorem -/
theorem twenty_multi_painted_cubes :
  let cube : PaintedCube := {
    size := 4,
    top_painted := true,
    sides_painted := true,
    bottom_painted := false
  }
  count_multi_painted_cubes cube = 20 := by
  sorry

end twenty_multi_painted_cubes_l1488_148862


namespace truck_toll_calculation_l1488_148824

/-- Calculates the toll for a truck given the number of axles -/
def toll (x : ℕ) : ℚ :=
  0.50 + 0.50 * (x - 2)

/-- Calculates the number of axles for a truck given the total number of wheels,
    number of wheels on the front axle, and number of wheels on each other axle -/
def numAxles (totalWheels frontAxleWheels otherAxleWheels : ℕ) : ℕ :=
  1 + (totalWheels - frontAxleWheels) / otherAxleWheels

theorem truck_toll_calculation :
  let x := numAxles 18 2 4
  toll x = 2 :=
by sorry

end truck_toll_calculation_l1488_148824


namespace friend_team_assignments_l1488_148884

/-- The number of ways to assign n distinguishable objects to k distinct categories -/
def assignments (n k : ℕ) : ℕ := k^n

/-- The number of friends -/
def num_friends : ℕ := 8

/-- The number of teams -/
def num_teams : ℕ := 4

/-- Theorem: The number of ways to assign 8 friends to 4 teams is 65536 -/
theorem friend_team_assignments : assignments num_friends num_teams = 65536 := by
  sorry

end friend_team_assignments_l1488_148884
